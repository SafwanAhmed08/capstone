# rl_threshold_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ThresholdEnv(gym.Env):
    """
    Adaptive Threshold Control Environment for IDS
    Compatible with Gymnasium + Stable-Baselines3 v2
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, features, binary_probs, expert_probs, true_labels,
                 init_bin_thresh=0.5, init_expert_thresh=0.5):
        super().__init__()
        self.features = np.asarray(features)
        self.binary_probs = np.asarray(binary_probs)
        self.expert_probs = np.asarray(expert_probs)
        self.labels = np.asarray(true_labels)
        self.n_samples = len(self.features)

        # Observation = features + [binary_prob, mean_expert_prob, bin_thresh, exp_thresh]
        obs_dim = self.features.shape[1] + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Actions = threshold adjustments (-0.1 → +0.1)
        self.action_space = spaces.Discrete(5)

        self.cur_idx = 0
        self.bin_thresh = init_bin_thresh
        self.exp_thresh = init_expert_thresh

    # -----------------------
    def _get_obs(self):
        x = self.features[self.cur_idx]
        bp = np.array([self.binary_probs[self.cur_idx]])
        ep = np.array([np.mean(self.expert_probs[self.cur_idx])])
        thr = np.array([self.bin_thresh, self.exp_thresh])
        return np.concatenate([x, bp, ep, thr]).astype(np.float32)

    # -----------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.cur_idx = 0
        self.bin_thresh = 0.5
        self.exp_thresh = 0.5
        return self._get_obs(), {}  # <-- Gymnasium expects (obs, info)

    # -----------------------
    def step(self, action):
    # Adjust thresholds
        deltas = [-0.1, -0.05, 0.0, 0.05, 0.1]
        self.bin_thresh = np.clip(self.bin_thresh + deltas[action], 0.0, 1.0)
        self.exp_thresh = np.clip(self.exp_thresh + deltas[action], 0.0, 1.0)

        # Evaluate prediction
        true = self.labels[self.cur_idx]
        bin_pred = int(self.binary_probs[self.cur_idx] > self.bin_thresh)
        exp_pred = int(np.mean(self.expert_probs[self.cur_idx]) > self.exp_thresh)
        final_pred = bin_pred & exp_pred

        # Reward shaping
        TP = (final_pred == 1 and true == 1)
        TN = (final_pred == 0 and true == 0)
        FP = (final_pred == 1 and true == 0)
        FN = (final_pred == 0 and true == 1)
        reward = 2 * TP + 0.5 * TN - 1.5 * FP - 3 * FN
        reward = float(reward)

        # Increment step
        self.cur_idx += 1
        terminated = self.cur_idx >= self.n_samples
        truncated = False

        # ✅ Only call _get_obs() if still inside bounds
        if not terminated:
            obs = self._get_obs()
        else:
            # Create an array of zeros with correct shape, no _get_obs() call
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {}
        return obs, reward, terminated, truncated, info
