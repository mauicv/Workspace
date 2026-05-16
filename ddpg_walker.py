import os
import random
from collections import deque

os.environ["MUJOCO_GL"] = "disabled"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dm_control import suite


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        to = lambda x: torch.FloatTensor(np.array(x))
        return to(s), to(a), to(r).unsqueeze(1), to(ns), to(d).unsqueeze(1)

    def __len__(self):
        return len(self.buf)


# ── Networks ──────────────────────────────────────────────────────────────────

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, hidden=256):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, act_dim], output_activation=nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        return self.act_limit * self.net(obs)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, hidden, hidden, 1])

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


# ── Ornstein-Uhlenbeck Noise ──────────────────────────────────────────────────

class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu.copy()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(*self.state.shape)
        self.state += dx
        return self.state


# ── DDPG Agent ────────────────────────────────────────────────────────────────

class DDPG:
    def __init__(self, obs_dim, act_dim, act_limit,
                 gamma=0.99, tau=0.005, actor_lr=1e-3, critic_lr=1e-3):
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(obs_dim, act_dim, act_limit)
        self.actor_target = Actor(obs_dim, act_dim, act_limit)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(obs_dim, act_dim)
        self.critic_target = Critic(obs_dim, act_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, obs):
        with torch.no_grad():
            return self.actor(torch.FloatTensor(obs)).numpy()

    def update(self, buffer, batch_size=256):
        s, a, r, ns, d = buffer.sample(batch_size)

        with torch.no_grad():
            target_q = r + self.gamma * (1 - d) * self.critic_target(ns, self.actor_target(ns))

        critic_loss = F.mse_loss(self.critic(s, a), target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

        return critic_loss.item(), actor_loss.item()


# ── dm_control helpers ────────────────────────────────────────────────────────

def flatten_obs(time_step):
    return np.concatenate([np.atleast_1d(v).ravel() for v in time_step.observation.values()])


def obs_dim_from_spec(spec):
    return sum(max(1, int(np.prod(v.shape))) for v in spec.values())


# ── Main training loop ────────────────────────────────────────────────────────

def train(
    total_steps=200_000,
    start_steps=5_000,
    batch_size=256,
    eval_every=10_000,
    eval_episodes=5,
    seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = suite.load("walker", "walk", task_kwargs={"random": seed})
    eval_env = suite.load("walker", "walk", task_kwargs={"random": seed + 1})

    obs_dim = obs_dim_from_spec(env.observation_spec())
    act_spec = env.action_spec()
    act_dim = act_spec.shape[0]
    act_limit = float(act_spec.maximum[0])

    agent = DDPG(obs_dim, act_dim, act_limit)
    buffer = ReplayBuffer()
    noise = OUNoise(act_dim)

    def evaluate():
        returns = []
        for _ in range(eval_episodes):
            ts = eval_env.reset()
            ep_ret = 0.0
            noise.reset()
            while not ts.last():
                obs = flatten_obs(ts)
                act = agent.select_action(obs)
                ts = eval_env.step(act)
                ep_ret += ts.reward
            returns.append(ep_ret)
        return np.mean(returns)

    ts = env.reset()
    noise.reset()
    ep_ret, ep_len = 0.0, 0
    step = 0

    print(f"{'Step':>8}  {'EpReturn':>10}  {'EvalReturn':>12}  {'CriticLoss':>12}  {'ActorLoss':>12}")
    print("-" * 60)

    c_loss_log, a_loss_log = [], []

    while step < total_steps:
        obs = flatten_obs(ts)

        if step < start_steps:
            act = np.random.uniform(act_spec.minimum, act_spec.maximum)
        else:
            act = np.clip(
                agent.select_action(obs) + noise.sample(),
                act_spec.minimum, act_spec.maximum,
            )

        next_ts = env.step(act)
        next_obs = flatten_obs(next_ts)
        done = float(next_ts.last())

        buffer.push(obs, act, next_ts.reward, next_obs, done)
        ep_ret += next_ts.reward
        ep_len += 1
        step += 1

        if next_ts.last():
            ts = env.reset()
            noise.reset()
            ep_ret, ep_len = 0.0, 0
        else:
            ts = next_ts

        if len(buffer) >= batch_size and step >= start_steps:
            c_loss, a_loss = agent.update(buffer, batch_size)
            c_loss_log.append(c_loss)
            a_loss_log.append(a_loss)

        if step % eval_every == 0:
            eval_ret = evaluate()
            avg_c = np.mean(c_loss_log[-1000:]) if c_loss_log else float("nan")
            avg_a = np.mean(a_loss_log[-1000:]) if a_loss_log else float("nan")
            print(f"{step:>8}  {'':>10}  {eval_ret:>12.2f}  {avg_c:>12.4f}  {avg_a:>12.4f}")

    print("\nTraining complete.")


if __name__ == "__main__":
    train()
