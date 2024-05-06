import matplotlib.pyplot as plt
import sys
from collections import deque
import timeit
from datetime import timedelta
from copy import deepcopy
import numpy as np
import random
from PIL import Image
from tensorboardX import SummaryWriter
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from collections import namedtuple
import collections
from tqdm import tqdm

from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from gym import spaces
import pybullet as p

# Constants
# MODE = "train"
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 200
EPS_DECAY_LAST_FRAME = 10**4
TARGET_UPDATE = 1000
LEARNING_RATE = 1e-4

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device = {device}")

class CustomKukaEnv(KukaDiverseObjectEnv):

    def _reward(self):
        """Calculates the reward for the episode.

        The reward is 1 if one of the green blocks is successfully picked up,
        0 otherwise.
        """
        reward = 0
        self._graspSuccess = 0
        for uid in self._objectUids:
            # Get the color of the object (assuming the color is stored as an attribute)
            object_color =  self.objectColors[str(uid)]
            # Check if the object is green (modify this condition according to your environment)
            if object_color[1] == 1.0 and object_color[0] == 0.0 and object_color[2] == 0.0:
                # Check if the object is above a certain height
                pos, _ = p.getBasePositionAndOrientation(uid)
                if pos[2] > 0.2:
                    self._graspSuccess += 1
                    reward = 1
                    break
        return reward

    def _randomly_place_objects(self, urdfList):
      """Randomly places the objects in the bin.

      Args:
        urdfList: The list of urdf files to place in the bin.

      Returns:
        The list of object unique ID's.
      """

      # Randomize positions of each object urdf.
      objectUids = []
      self.objectColors = {}
      self.urdfList = urdfList
      # self.greenUrdf = []
      for i, urdf_name in enumerate(urdfList):
        xpos = 0.4 + self._blockRandom * random.random()
        ypos = self._blockRandom * (random.random() - .5)
        angle = np.pi / 2 + self._blockRandom * np.pi * random.random()
        orn = p.getQuaternionFromEuler([0, 0, angle])

        urdf_path = os.path.join(self._urdfRoot, urdf_name)
        uid = p.loadURDF(urdf_path, [xpos, ypos, .15], [orn[0], orn[1], orn[2], orn[3]])

        if i == 0:
           p.changeVisualShape(uid, -1, rgbaColor=[0.0,1.0,0.0,1.0])

        objectUids.append(uid)
        self.objectColors[str(uid)] = p.getVisualShapeData(uid)[0][7]

        # Let each object fall to the tray individual, to prevent object
        # intersection.

        for _ in range(500):
          p.stepSimulation()
      return objectUids
    
env = CustomKukaEnv(renders=False, isDiscrete=True, removeHeightHack=False, maxSteps=20)
env.cid = p.connect(p.DIRECT)
env.reset()
    
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

STACK_SIZE = 5

class DuelingDQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DuelingDQN, self).__init__()  
        self.conv1 = nn.Conv2d(STACK_SIZE, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,8,4),4,2),3,1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,8,4),4,2),3,1)
        linear_input_size = convw * convh * 64
        self.advantage = nn.Linear(linear_input_size, 512)
        self.value = nn.Linear(linear_input_size, 512)
        self.advantage_head = nn.Linear(512, outputs)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        advantage = F.relu(self.advantage(x))
        value = F.relu(self.value(x))
        advantage = self.advantage_head(advantage)
        value = self.value_head(value).expand(x.size(0), n_actions)  # Use n_actions instead of self.action_space
        return value + advantage - advantage.mean()


preprocess = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen():
    global stacked_screens
    screen = env._get_observation().transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    screen = screen[1,:,:].unsqueeze(0) 
    return preprocess(screen).unsqueeze(0).to(device)

init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape
n_actions = env.action_space.n

policy_net = DuelingDQN(screen_height, screen_width, n_actions).to(device)
target_net = DuelingDQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(10000)

eps_threshold = 0

def select_action(state, i_episode):
    global steps_done
    global eps_threshold
    sample = random.random()
    eps_threshold = max(EPS_END, EPS_START - i_episode / EPS_DECAY_LAST_FRAME)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

PATH = 'dueling_dqn.pt'

num_episodes = 100000
writer = SummaryWriter()
total_rewards = []
plt_rewards = []
ten_rewards = 0
best_mean_reward = None
start_time = timeit.default_timer()
for i_episode in tqdm(range(num_episodes), desc="Training Episodes"):
    env.reset()
    state = get_screen()
    stacked_states = collections.deque(STACK_SIZE*[state],maxlen=STACK_SIZE)
    while(True):
        stacked_states_t =  torch.cat(tuple(stacked_states),dim=1)
        action = select_action(stacked_states_t, i_episode)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        next_state = get_screen()
        if not done:
            next_stacked_states = stacked_states
            next_stacked_states.append(next_state)
            next_stacked_states_t =  torch.cat(tuple(next_stacked_states),dim=1)
        else:
            next_stacked_states_t = None
        memory.push(stacked_states_t, action, next_stacked_states_t, reward)
        stacked_states = next_stacked_states
        optimize_model()
        if done:
            reward = reward.cpu().numpy().item()
            ten_rewards += reward
            total_rewards.append(reward)
            mean_reward = np.mean(total_rewards[-100:])*100
            writer.add_scalar("epsilon", eps_threshold, i_episode)
            if (best_mean_reward is None or best_mean_reward < mean_reward) and i_episode > 100:
                torch.save({
                        'policy_net_state_dict': policy_net.state_dict(),
                        'target_net_state_dict': target_net.state_dict(),
                        'optimizer_policy_net_state_dict': optimizer.state_dict()
                        }, PATH)
                if best_mean_reward is not None:
                    print("Best mean reward updated %.1f -> %.1f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward

            if i_episode % 100 == 0:
                plt_rewards.append([i_episode, mean_reward])
            break
            
    if i_episode%10 == 0:
            writer.add_scalar('ten episodes average rewards', ten_rewards/10.0, i_episode)
            ten_rewards = 0
    
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    if i_episode>=200 and mean_reward>50:
        print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode+1, mean_reward))
        break

print('Average Score: {:.2f}'.format(mean_reward))
elapsed = timeit.default_timer() - start_time
print("Elapsed time: {}".format(timedelta(seconds=elapsed)))

# plt_rewards = np.array(plt_rewards)
# plt.plot(plt_rewards[:,0], plt_rewards[:,1])
# plt.title('Rewards Over Episodes')
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.grid(True)
# plt.savefig('dueling_dqn_rewards.png')
# writer.close()
# env.close()

# episode = 10
# scores_window = collections.deque(maxlen=100) 
# env = CustomKukaEnv(renders=False, isDiscrete=True, removeHeightHack=False, maxSteps=20, isTest=True)
# env.cid = p.connect(p.GUI)
# checkpoint = torch.load(PATH)
# policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

# for i_episode in range(episode):
#     env.reset()
#     state = get_screen()
#     stacked_states = collections.deque(STACK_SIZE*[state],maxlen=STACK_SIZE)

#     video_filename = f'dueling_dqn_eval_episode_{i_episode+1}.mp4'
#     p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_filename)

#     while(True):
#         stacked_states_t =  torch.cat(tuple(stacked_states),dim=1)
#         action = policy_net(stacked_states_t).max(1)[1].view(1, 1)
#         _, reward, done, _ = env.step(action.item())
#         next_state = get_screen()
#         stacked_states.append(next_state)
#         if done:
#             break
    
#     p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)
#     print("Episode: {0:d}, reward: {1}".format(i_episode+1, reward), end="\n")
