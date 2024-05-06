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

# env = KukaDiverseObjectEnv(renders=False, isDiscrete=False, removeHeightHack=False, maxSteps=20)
# env.cid = p.connect(p.DIRECT)
# action_space = spaces.Box(low=-1, high=1, shape=(5,1))

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

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()  
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
        self.linear = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.linear(x.view(x.size(0), -1)))
        return self.head(x)
    
preprocess = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen():
    global stacked_screens
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env._get_observation().transpose((2, 0, 1))
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)

    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    screen = screen[1,:,:].unsqueeze(0) # Extract only the green channel
    # Resize, and add a batch dimension (BCHW)

    return preprocess(screen).unsqueeze(0).to(device)

# Get screen size so that we can initialize layers correctly based on shape
# returned from pybullet (48, 48, 3).  
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
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
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

PATH = 'policy_dqn.pt'

# Training
# num_episodes = 100000
# # num_episodes = 1000
# writer = SummaryWriter()
# total_rewards = []
# plt_rewards = []
# ten_rewards = 0
# best_mean_reward = None
# start_time = timeit.default_timer()
# for i_episode in tqdm(range(num_episodes), desc="Training Episodes"):
#     # Initialize the environment and state
#     env.reset()
#     print(env.objectColors)
#     state = get_screen()
#     stacked_states = collections.deque(STACK_SIZE*[state],maxlen=STACK_SIZE)
#     while(True):
#         stacked_states_t =  torch.cat(tuple(stacked_states),dim=1)
#         # Select and perform an action
#         action = select_action(stacked_states_t, i_episode)
#         _, reward, done, _ = env.step(action.item())
        
#         reward = torch.tensor([reward], device=device)

#         # Observe new state
#         next_state = get_screen()
#         if not done:
#             next_stacked_states = stacked_states
#             next_stacked_states.append(next_state)
#             next_stacked_states_t =  torch.cat(tuple(next_stacked_states),dim=1)
#         else:
#             next_stacked_states_t = None
            
#         # Store the transition in memory
#         memory.push(stacked_states_t, action, next_stacked_states_t, reward)

#         # Move to the next state
#         stacked_states = next_stacked_states
        
#         # Perform one step of the optimization (on the target network)
#         optimize_model()
#         if done:
#             reward = reward.cpu().numpy().item()
#             ten_rewards += reward
#             total_rewards.append(reward)
#             mean_reward = np.mean(total_rewards[-100:])*100
#             writer.add_scalar("epsilon", eps_threshold, i_episode)
#             if (best_mean_reward is None or best_mean_reward < mean_reward) and i_episode > 100:
#                 # For saving the model and possibly resuming training
#                 torch.save({
#                         'policy_net_state_dict': policy_net.state_dict(),
#                         'target_net_state_dict': target_net.state_dict(),
#                         'optimizer_policy_net_state_dict': optimizer.state_dict()
#                         }, PATH)
#                 if best_mean_reward is not None:
#                     print("Best mean reward updated %.1f -> %.1f, model saved" % (best_mean_reward, mean_reward))
#                 best_mean_reward = mean_reward

#             if i_episode % 100 == 0:
#                 plt_rewards.append([i_episode, mean_reward])
#             break
            
#     if i_episode%10 == 0:
#             writer.add_scalar('ten episodes average rewards', ten_rewards/10.0, i_episode)
#             ten_rewards = 0
    
#     # Update the target network, copying all weights and biases in DQN
#     if i_episode % TARGET_UPDATE == 0:
#         target_net.load_state_dict(policy_net.state_dict())

#     if i_episode>=200 and mean_reward>50:
#         print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode+1, mean_reward))
#         break

#     if i_episode % 1000 == 0 and i_episode != 0:
#         plt_rewards_arr = np.array(plt_rewards)
#         plt.plot(plt_rewards_arr[:,0], plt_rewards_arr[:,1])
#         plt.title('Rewards Over Episodes')
#         plt.xlabel('Episode')
#         plt.ylabel('Reward')
#         plt.grid(True)
#         plt.savefig('dueling_dqn_rewards.png')

# print('Average Score: {:.2f}'.format(mean_reward))
# elapsed = timeit.default_timer() - start_time
# print("Elapsed time: {}".format(timedelta(seconds=elapsed)))

# # Plot rewards over episodes

# plt_rewards = np.array(plt_rewards)
# plt.plot(plt_rewards[:,0], plt_rewards[:,1])
# plt.title('Rewards Over Episodes')
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.grid(True)
# plt.savefig('dqn_rewards.png')
# writer.close()
# env.close()

# Evaluation

episode = 10
scores_window = collections.deque(maxlen=100)  # last 100 scores
env = CustomKukaEnv(renders=False, isDiscrete=True, removeHeightHack=False, maxSteps=20)
env.cid = p.connect(p.GUI)
# env.get_green_urdf_list()
# load the model
checkpoint = torch.load(PATH)
policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

# evaluate the model
for i_episode in range(episode):
    env.reset()
    state = get_screen()
    stacked_states = collections.deque(STACK_SIZE*[state],maxlen=STACK_SIZE)

    # # Start video recording
    # video_filename = f'recordings/dqn_eval_episode_{i_episode+1}.mp4'
    # p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_filename)

    while(True):
        stacked_states_t =  torch.cat(tuple(stacked_states),dim=1)
        # Select and perform an action
        action = policy_net(stacked_states_t).max(1)[1].view(1, 1)
        _, reward, done, _ = env.step(action.item())
        # Observe new state
        next_state = get_screen()
        stacked_states.append(next_state)
        if done:
            break
    
    # Stop video recording
    # p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)

    print("Episode: {0:d}, reward: {1}".format(i_episode+1, reward), end="\n")