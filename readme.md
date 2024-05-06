**Steps to run:**

Navigate to the submission folder and use the following command to run:

"python kuka_dueling_dqn.py"

**Algorithm:**

The script uses an algorithm very similar to dueling DQN with minor tweaks, the implementation of which is closely inspired from : https://github.com/dxyang/DQN_pytorch

**Problems:** 

Couldn't find a working way to set this up on my local PC to have a GUI. Couldn't record a proper video when running on a headless display over the cluster.

**Time complexity:**

Picking up only the green blocks ended up being a challenging problem for the RL model. It takes around 4 hrs to complete 6000 episodes on an RTX A5000 GPU, and even then the best mean reward only climbed up to 35.0. Couldn't run it further since I couldn't avail GPU resources from the cluster for longer. 
