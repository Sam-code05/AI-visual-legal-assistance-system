import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ======================
# 1. 環境 Environment
# ======================

@dataclass
class StepResult:
    next_state: np.ndarray
    reward: float
    done: bool
    info: dict
# 有 N 個員工，每個 episode 有 T 個任務
# 每個步驟：經理觀察目前任務 + 員工能力 → 選一個員工
# 若 worker_skill >= task_difficulty 就成功，reward = 1，否則 0
class TaskAssignmentEnv:
    def __init__(self, num_workers=3, num_tasks=5):
        self.num_workers = num_workers
        self.num_tasks = num_tasks
        self.state_dim = 1 + num_workers + 1  
        self.action_dim = num_workers         # 選擇哪一位員工
        self.worker_skills = np.array([0.3, 0.6, 0.9], dtype=np.float32)
        # self.worker_skills = np.random.rand(self.num_workers)
        self.tasks = None
        self.current_task_idx = 0

    # 隨機生成員工能力與任務難度（0~1）
    def reset(self):
        # self.tasks = np.random.rand(self.num_tasks)

        levels = np.array([0.2, 0.5, 0.8], dtype=np.float32)
        self.tasks = np.random.choice(levels, size=self.num_tasks)
        self.current_task_idx = 0
        return self._get_state()

    def _get_state(self):
        current_task_diff = self.tasks[self.current_task_idx]
        idx_norm = self.current_task_idx / (self.num_tasks - 1)
        state = np.concatenate([[current_task_diff], self.worker_skills, [idx_norm]])
        return state.astype(np.float32)
    
    # action: 選擇哪一位員工 (0 ~ num_workers-1)
    def step(self, action: int) -> StepResult:
        assert 0 <= action < self.num_workers

        current_task_diff = self.tasks[self.current_task_idx]
        chosen_skill = self.worker_skills[action]

        # 成功條件：員工能力 >= 任務難度
        success = chosen_skill >= current_task_diff
        reward = 1.0 if success else 0.0

        self.current_task_idx += 1
        done = self.current_task_idx >= self.num_tasks

        if not done:
            next_state = self._get_state()
        else:
            next_state = np.zeros(self.state_dim, dtype=np.float32)

        info = {
            "success": success,
            "chosen_skill": float(chosen_skill),
            "task_difficulty": float(current_task_diff),
        }

        return StepResult(next_state, reward, done, info)

# ======================
# 2. Policy 網路（經理）
# ======================

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)
    
    # 根據目前 policy 對狀態 state 取樣一個 action
    # 回傳：action, log_prob(action)
def select_action(policy_net, state):
    state_t = torch.from_numpy(state).unsqueeze(0)  # (1, state_dim)
    logits = policy_net(state_t)                    # (1, action_dim)
    prob = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(prob)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return int(action.item()), log_prob

# ======================
# 3. REINFORCE 訓練迴圈
# ======================

def reinforce_train(num_episodes=3000,
                    gamma=0.99,
                    lr=5e-3, # 學習率微調(1e-2、5e-3、1e-3)比較
                    print_every=200):
    env = TaskAssignmentEnv(num_workers=3, num_tasks=5)
    policy_net = PolicyNet(env.state_dim, env.action_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    all_episode_rewards = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        log_probs = []
        rewards = []

        done = False
        while not done:
            action, log_prob = select_action(policy_net, state)
            step_result = env.step(action)

            log_probs.append(log_prob)
            rewards.append(step_result.reward)

            state = step_result.next_state
            done = step_result.done

        # 計算折扣回報
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # 標準化 returns，幫助穩定訓練
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs_t = torch.stack(log_probs)
        loss = - (log_probs_t * returns).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        all_episode_rewards.append(total_reward)

        if episode % print_every == 0:
            avg_reward = np.mean(all_episode_rewards[-print_every:])
            print(f"Episode {episode:4d} | "
                  f"avg reward (last {print_every}) = {avg_reward:.3f}")

    return policy_net, all_episode_rewards

# ======================
# 4. 評估與畫圖
# ======================

def plot_rewards(rewards_history, filename="training_curve.png"):
    window = 100
    moving_avg = [np.mean(rewards_history[max(0, i-window):i+1])
                  for i in range(len(rewards_history))]

    plt.figure()
    plt.plot(rewards_history, label="Episode reward")
    plt.plot(moving_avg, label=f"Moving avg ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Total reward (tasks completed)")
    plt.legend()
    plt.title("Training curve: Manager learns to assign tasks")
    plt.tight_layout()
    plt.savefig(filename)  # 存圖
    plt.show()
    print(f"Training curve saved as {filename}")

def evaluate(policy_net, num_episodes=20):
    env = TaskAssignmentEnv(num_workers=3, num_tasks=5)
    total = 0
    for _ in range(num_episodes):
        state = env.reset()
        ep_reward = 0
        done = False
        while not done:
            state_t = torch.from_numpy(state).unsqueeze(0)
            logits = policy_net(state_t)
            action = torch.argmax(logits, dim=-1).item()

            step_result = env.step(action)
            ep_reward += step_result.reward
            state = step_result.next_state
            done = step_result.done
        total += ep_reward
    avg = total / num_episodes
    print(f"[Evaluate] average tasks completed per episode = {avg:.2f} / 5")
    return avg

# ======================
# 5. main()
# ======================

def main():
    policy_net, rewards_history = reinforce_train()
    plot_rewards(rewards_history)
    evaluate(policy_net)

if __name__ == "__main__":
    main()
