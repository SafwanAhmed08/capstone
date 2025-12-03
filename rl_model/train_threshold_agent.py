# train_threshold_agent.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_threshold_env import ThresholdEnv
import numpy as np

SAVE_DIR = "/Users/safwanahmed/Desktop/preprocessing/experts"

features = np.load(f"{SAVE_DIR}/rl_features.npy")
binary_probs = np.load(f"{SAVE_DIR}/rl_binary_probs.npy")
expert_probs = np.load(f"{SAVE_DIR}/rl_expert_probs.npy")
true_labels = np.load(f"{SAVE_DIR}/rl_true_labels.npy")



env = DummyVecEnv([lambda: ThresholdEnv(features, binary_probs, expert_probs, true_labels)])
import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"✅ Using {device.upper()} device for training")

model = PPO("MlpPolicy", env, verbose=1, device=device)
model.learn(total_timesteps=100_000)
model.save("ppo_threshold_controller.zip")
