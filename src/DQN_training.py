import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
import os 
import sys

current_script_path = os.path.abspath(os.path.dirname(__file__))

project_dir = os.path.abspath(os.path.join(current_script_path, '..'))

if project_dir not in sys.path:
    sys.path.append(project_dir)


env = gym.make("highway-fast-v0")

model = DQN('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=5e-4,
              buffer_size=20000,
              learning_starts=200,
              batch_size=32,
              gamma=0.8,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=50,
              verbose=1,
              tensorboard_log="../logs/highway_dqn/")
model.learn(int(2e6))
model.save("../models/highway_dqn/model")

# Load and test saved model
model = DQN.load("../models/highway_dqn/model")
while True:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()