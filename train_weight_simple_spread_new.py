"""
MAPPO training script for the MPE simple_spread environment.
"""

import argparse
import os
import shutil

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from pettingzoo.mpe import simple_spread_v3
except ModuleNotFoundError:
    from mpe2 import simple_spread_v3


RESULT_BASE_DIR = "results"
ENV_NAME = "simple_spread_v3"
RESULT_DIR = os.path.join(RESULT_BASE_DIR, ENV_NAME)
LOGS_DIR = os.path.join(RESULT_DIR, "logs")
PLOTS_DIR = os.path.join(RESULT_DIR, "plots")
WEIGHTS_DIR = os.path.join(RESULT_DIR, "weights")
LOG_FILE = os.path.join(LOGS_DIR, "training_log.txt")


def prepare_output_dirs():
    for directory in [LOGS_DIR, PLOTS_DIR, WEIGHTS_DIR]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)

    if os.path.exists(LOG_FILE):
        open(LOG_FILE, "w", encoding="utf-8").close()


def log_message(message):
    with open(LOG_FILE, "a", encoding="utf-8") as file_obj:
        file_obj.write(message + "\n")


def plot_all_metrics(metrics_dict, episode):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Training Metrics of {ENV_NAME} (Up to Episode {episode})", fontsize=16)
    axes = axes.flatten()

    any_metric = list(metrics_dict.values())[0]
    x_values = [50 * (i + 1) for i in range(len(any_metric))]
    window_size = min(5, len(x_values)) if x_values else 1

    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        if i >= 5:
            break

        ax = axes[i]
        values_array = np.asarray(values, dtype=np.float32)

        if len(values) > window_size:
            smoothed = np.convolve(values_array, np.ones(window_size) / window_size, mode="valid")
            std_values = np.asarray(
                [np.std(values_array[j : j + window_size]) for j in range(len(values) - window_size + 1)]
            )
            smoothed_x = x_values[window_size - 1 :]

            ax.plot(smoothed_x, smoothed, "-", linewidth=2, label="Smoothed")
            ax.scatter(x_values, values, alpha=0.3, label="Original")
            ax.fill_between(
                smoothed_x,
                smoothed - std_values,
                smoothed + std_values,
                alpha=0.2,
                label="+/-1 StdDev",
            )
        else:
            ax.plot(x_values, values, "o-", label="Data")

        ax.set_title(metric_name.replace("_", " "))
        ax.set_xlabel("Episodes")
        ax.set_ylabel(metric_name.replace("_", " "))
        ax.grid(True, alpha=0.3)
        ax.legend()

    if len(metrics_dict) < 6:
        fig.delaxes(axes[5])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(PLOTS_DIR, "training_metrics.png"))
    plt.close(fig)


def compute_entropy(probs):
    dist = torch.distributions.Categorical(probs)
    return dist.entropy().mean().item()


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().cpu().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.asarray(advantage_list), dtype=torch.float32)


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)


class CentralValueNet(torch.nn.Module):
    def __init__(self, total_state_dim, hidden_dim, team_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(total_state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, team_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MAPPO:
    def __init__(self, team_size, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, eps, gamma, device):
        self.team_size = team_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.device = device

        self.actors = [PolicyNet(state_dim, hidden_dim, action_dim).to(device) for _ in range(team_size)]
        self.critic = CentralValueNet(team_size * state_dim, hidden_dim, team_size).to(device)
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), actor_lr) for actor in self.actors]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), critic_lr)

    def save_model(self, path=None):
        save_path = path or WEIGHTS_DIR
        os.makedirs(save_path, exist_ok=True)

        for i, actor in enumerate(self.actors):
            torch.save(actor.state_dict(), os.path.join(save_path, f"actor_{i}.pth"))
        torch.save(self.critic.state_dict(), os.path.join(save_path, "critic.pth"))

    def take_action(self, state_per_agent):
        actions = []
        action_probs = []

        for i, actor in enumerate(self.actors):
            state_tensor = torch.tensor(np.asarray([state_per_agent[i]]), dtype=torch.float32, device=self.device)
            probs = actor(state_tensor)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()

            actions.append(action.item())
            action_probs.append(probs.detach().cpu().numpy()[0])

        return actions, action_probs

    def update(self, transition_dicts):
        time_steps = len(transition_dicts[0]["states"])
        if time_steps == 0:
            return 0.0, 0.0, 0.0

        states_all = []
        next_states_all = []
        for t in range(time_steps):
            states_all.append(np.concatenate([transition_dicts[i]["states"][t] for i in range(self.team_size)]))
            next_states_all.append(
                np.concatenate([transition_dicts[i]["next_states"][t] for i in range(self.team_size)])
            )

        states_all = torch.tensor(np.asarray(states_all), dtype=torch.float32, device=self.device)
        next_states_all = torch.tensor(np.asarray(next_states_all), dtype=torch.float32, device=self.device)
        rewards_all = torch.tensor(
            [[transition_dicts[i]["rewards"][t] for i in range(self.team_size)] for t in range(time_steps)],
            dtype=torch.float32,
            device=self.device,
        )
        dones_all = torch.tensor(
            [[transition_dicts[i]["dones"][t] for i in range(self.team_size)] for t in range(time_steps)],
            dtype=torch.float32,
            device=self.device,
        )

        values = self.critic(states_all)
        next_values = self.critic(next_states_all)
        td_target = rewards_all + self.gamma * next_values * (1 - dones_all)
        td_delta = td_target - values

        advantages = [compute_advantage(self.gamma, self.lmbda, td_delta[:, i]).to(self.device) for i in range(self.team_size)]

        critic_loss = F.mse_loss(values, td_target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        action_losses = []
        entropies = []

        for i in range(self.team_size):
            states = torch.tensor(np.asarray(transition_dicts[i]["states"]), dtype=torch.float32, device=self.device)
            actions = torch.tensor(transition_dicts[i]["actions"], dtype=torch.long, device=self.device).view(-1, 1)
            old_probs = torch.tensor(
                np.asarray(transition_dicts[i]["action_probs"]), dtype=torch.float32, device=self.device
            )

            current_probs = self.actors[i](states)
            log_probs = torch.log(current_probs.gather(1, actions).clamp_min(1e-8))
            old_log_probs = torch.log(old_probs.gather(1, actions).clamp_min(1e-8)).detach()

            ratio = torch.exp(log_probs - old_log_probs)
            advantage = advantages[i].unsqueeze(-1)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            action_loss = torch.mean(-torch.min(surr1, surr2))
            entropy_val = compute_entropy(current_probs)

            self.actor_optimizers[i].zero_grad()
            action_loss.backward()
            self.actor_optimizers[i].step()

            action_losses.append(action_loss.item())
            entropies.append(entropy_val)

        return float(np.mean(action_losses)), critic_loss.item(), float(np.mean(entropies))


def train(args):
    prepare_output_dirs()

    actor_lr = 3e-4
    critic_lr = 1e-3
    hidden_dim = 64
    gamma = 0.99
    lmbda = 0.97
    eps = 0.3
    team_size = args.team_size
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")

    env = simple_spread_v3.parallel_env(N=team_size)

    reset_result = env.reset()
    current_states = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    agent_order = list(env.agents)

    state_dim = env.observation_space(agent_order[0]).shape[0]
    action_dim = env.action_space(agent_order[0]).n

    mappo = MAPPO(team_size, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, eps, gamma, device)

    total_rewards_per_episode = []
    episode_lengths = []
    policy_losses = []
    value_losses = []
    entropies = []

    avg_total_rewards_per_50 = []
    avg_episode_length_per_50 = []
    avg_policy_loss_per_50 = []
    avg_value_loss_per_50 = []
    avg_entropy_per_50 = []

    with tqdm(total=args.episodes, desc="Training") as pbar:
        for episode in range(1, args.episodes + 1):
            buffers = [
                {
                    "states": [],
                    "actions": [],
                    "next_states": [],
                    "rewards": [],
                    "dones": [],
                    "action_probs": [],
                }
                for _ in range(team_size)
            ]

            reset_result = env.reset()
            current_states = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            agent_order = list(env.agents)

            terminal = False
            episode_reward = 0.0
            steps = 0

            while not terminal:
                steps += 1
                state_list = [current_states[agent] for agent in agent_order]
                actions, prob_dists = mappo.take_action(state_list)
                action_dict = {agent_order[i]: actions[i] for i in range(team_size)}

                next_states, rewards, terminations, truncations, _ = env.step(action_dict)
                dones = {
                    agent: terminations.get(agent, False) or truncations.get(agent, False)
                    for agent in agent_order
                }

                step_reward = sum(rewards.get(agent, 0.0) for agent in agent_order)
                episode_reward += step_reward

                for i, agent in enumerate(agent_order):
                    next_state = next_states.get(agent, np.zeros(state_dim, dtype=np.float32))
                    buffers[i]["states"].append(np.asarray(current_states[agent], dtype=np.float32))
                    buffers[i]["actions"].append(actions[i])
                    buffers[i]["next_states"].append(np.asarray(next_state, dtype=np.float32))
                    buffers[i]["rewards"].append(rewards.get(agent, 0.0))
                    buffers[i]["dones"].append(float(dones[agent]))
                    buffers[i]["action_probs"].append(prob_dists[i])

                current_states = next_states
                terminal = all(dones.values())

            a_loss, c_loss, ent = mappo.update(buffers)

            total_rewards_per_episode.append(episode_reward)
            episode_lengths.append(steps)
            policy_losses.append(a_loss)
            value_losses.append(c_loss)
            entropies.append(ent)

            if episode % args.save_interval == 0:
                mappo.save_model()
                log_message(f"Model saved at episode {episode}")

            if episode % 50 == 0:
                avg_reward_50 = float(np.mean(total_rewards_per_episode[-50:]))
                avg_length_50 = float(np.mean(episode_lengths[-50:]))
                avg_policy_loss_50 = float(np.mean(policy_losses[-50:]))
                avg_value_loss_50 = float(np.mean(value_losses[-50:]))
                avg_entropy_50 = float(np.mean(entropies[-50:]))

                avg_total_rewards_per_50.append(avg_reward_50)
                avg_episode_length_per_50.append(avg_length_50)
                avg_policy_loss_per_50.append(avg_policy_loss_50)
                avg_value_loss_per_50.append(avg_value_loss_50)
                avg_entropy_per_50.append(avg_entropy_50)

                log_message(
                    f"Episode {episode}: "
                    f"AvgTotalReward(last50)={avg_reward_50:.3f}, "
                    f"AvgEpisodeLength(last50)={avg_length_50:.3f}, "
                    f"AvgPolicyLoss(last50)={avg_policy_loss_50:.3f}, "
                    f"AvgValueLoss(last50)={avg_value_loss_50:.3f}, "
                    f"AvgEntropy(last50)={avg_entropy_50:.3f}"
                )

                metrics_dict = {
                    "Average_Total_Reward": avg_total_rewards_per_50,
                    "Average_Episode_Length": avg_episode_length_per_50,
                    "Average_Policy_Loss": avg_policy_loss_per_50,
                    "Average_Value_Loss": avg_value_loss_per_50,
                    "Average_Entropy": avg_entropy_per_50,
                }
                plot_all_metrics(metrics_dict, episode)

            pbar.update(1)

    env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--team-size", type=int, default=2)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
