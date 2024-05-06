**REINFORCEMENT LEARNING BASED KUKA PICK & PLACE**

This repo consists of the implementation of DQN, Dueling DQN and PPO algorithms for picking up green blocks by a KUKA robot in a PyBullet environment and OpenAI gym.

Credits : Codes inspired from https://github.com/mahyaret/kuka_rl

https://github.com/Shyam-pi/RL-for-KUKA-pick-and-place/assets/57116285/0add6c1c-2427-49ae-9a03-8cd9e0f665ee

**Dependency Installation**

Install the necessary requirements by using the requirements.txt with the following command:

```conda create --name <env> --file requirements.txt```

where <env> is your conda environment's name of choice


**Steps to run:**

Once you are in the root folder, follow the following steps in the terminal with the conda environment as the base, according to your choice :

To run DQN algorithm training:

```python kuka_dueling_dqn.py```


To run Dueling DQN algorithm training:

```python kuka_dueling_dqn.py```


To run PPO algorithm training:

```python kuka_ppo.py```


**Note : Time complexity:**

Picking up only the green blocks ended up being a challenging problem for the RL model. It takes around 4 hrs to complete 6000 episodes of the DQN algorithm on an RTX A5000 GPU (majorly bottlenecked by the environment resetting)
