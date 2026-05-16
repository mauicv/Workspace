import csv
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


# ── Reward Gating ─────────────────────────────────────────────────────────────

class RewardGating:
    """
    Tracks a running EMA of episode returns and maps each episode's return
    to (tau_eff, pullback_lambda) for the slow-actor update.

    Dead zone  |adv| < dead_zone  → no slow-actor update, no pull-back.
    Good zone  adv >  dead_zone  → tau_eff scales with advantage (pull slow toward fast).
    Bad  zone  adv < -dead_zone  → pullback_lambda scales with disadvantage (pull fast toward slow).
    """
    def __init__(self, ema_alpha=0.05, dead_zone=0.5, tau_base=0.005, tau_max=0.05, lambda_max=1.0):
        self.ema_alpha = ema_alpha
        self.dead_zone = dead_zone
        self.tau_base = tau_base
        self.tau_max = tau_max
        self.lambda_max = lambda_max
        self.mean = None
        self.var = 1.0

    def update(self, ep_return):
        if self.mean is None:
            self.mean = float(ep_return)
        else:
            delta = float(ep_return) - self.mean
            self.mean += self.ema_alpha * delta
            self.var = (1 - self.ema_alpha) * self.var + self.ema_alpha * delta ** 2

    @property
    def std(self):
        return max(np.sqrt(self.var), 1e-3)

    def normalized_adv(self, ep_return):
        if self.mean is None:
            return 0.0
        return (float(ep_return) - self.mean) / self.std

    def get_update_params(self, ep_return):
        """Returns (tau_eff, pullback_lambda, normalized_adv)."""
        adv = self.normalized_adv(ep_return)
        if abs(adv) < self.dead_zone:
            return 0.0, 0.0, adv
        elif adv > self.dead_zone:
            scale = float(np.tanh(adv - self.dead_zone))
            tau_eff = self.tau_base + scale * (self.tau_max - self.tau_base)
            return tau_eff, 0.0, adv
        else:
            scale = float(np.tanh(-adv - self.dead_zone))
            return 0.0, scale * self.lambda_max, adv


# ── Fast-Slow DDPG Agent ──────────────────────────────────────────────────────

class FastSlowDDPG:
    """
    Three actors:
      fast_actor    — online, gets critic gradient updates, used for data collection.
      actor_target  — standard EMA (fixed tau) of fast_actor, used for critic target Q.
      slow_actor    — reward-gated EMA of fast_actor, used as anchor for pull-back.

    Pull-back: when episode return is below the running mean by > dead_zone std,
    the fast actor loss gains a penalty proportional to its distance from the slow actor.
    Slow-actor pull: when episode return exceeds the running mean by > dead_zone std,
    the slow actor is nudged toward the fast actor with tau_eff > 0.
    """
    def __init__(self, obs_dim, act_dim, act_limit,
                 gamma=0.99, tau_critic=0.005, tau_actor_target=0.005,
                 actor_lr=1e-3, critic_lr=1e-3):
        self.gamma = gamma
        self.tau_critic = tau_critic
        self.tau_actor_target = tau_actor_target
        self.pullback_lambda = 0.0

        self.fast_actor = Actor(obs_dim, act_dim, act_limit)

        self.actor_target = Actor(obs_dim, act_dim, act_limit)
        self.actor_target.load_state_dict(self.fast_actor.state_dict())
        for p in self.actor_target.parameters():
            p.requires_grad = False

        self.slow_actor = Actor(obs_dim, act_dim, act_limit)
        self.slow_actor.load_state_dict(self.fast_actor.state_dict())
        for p in self.slow_actor.parameters():
            p.requires_grad = False

        self.critic = Critic(obs_dim, act_dim)
        self.critic_target = Critic(obs_dim, act_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.fast_actor_opt = optim.Adam(self.fast_actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, obs):
        with torch.no_grad():
            return self.fast_actor(torch.FloatTensor(obs)).numpy()

    def update_slow_actor(self, tau_eff):
        if tau_eff <= 0.0:
            return
        for p_fast, p_slow in zip(self.fast_actor.parameters(), self.slow_actor.parameters()):
            p_slow.data.copy_(tau_eff * p_fast.data + (1 - tau_eff) * p_slow.data)

    def update(self, buffer, batch_size=256):
        s, a, r, ns, d = buffer.sample(batch_size)

        # Critic update using actor_target for stable bootstrapping
        with torch.no_grad():
            target_q = r + self.gamma * (1 - d) * self.critic_target(ns, self.actor_target(ns))
        critic_loss = F.mse_loss(self.critic(s, a), target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Fast actor: maximize Q, with optional pull-back penalty toward slow actor
        fast_actions = self.fast_actor(s)
        q_loss = -self.critic(s, fast_actions).mean()
        if self.pullback_lambda > 0:
            pullback_loss = F.mse_loss(fast_actions, self.slow_actor(s))
        else:
            pullback_loss = torch.tensor(0.0)
        actor_loss = q_loss + self.pullback_lambda * pullback_loss

        self.fast_actor_opt.zero_grad()
        actor_loss.backward()
        self.fast_actor_opt.step()

        # Standard EMA target updates
        for p, pt in zip(self.fast_actor.parameters(), self.actor_target.parameters()):
            pt.data.copy_(self.tau_actor_target * p.data + (1 - self.tau_actor_target) * pt.data)
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.copy_(self.tau_critic * p.data + (1 - self.tau_critic) * pt.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "q_loss": q_loss.item(),
            "pullback_loss": pullback_loss.item(),
        }

    @torch.no_grad()
    def actor_slow_dist(self, buffer, n=512):
        """MSE between fast and slow actor on a random buffer sample."""
        if len(buffer) < n:
            return float("nan")
        s, *_ = buffer.sample(n)
        return F.mse_loss(self.fast_actor(s), self.slow_actor(s)).item()


# ── dm_control helpers ────────────────────────────────────────────────────────

def flatten_obs(time_step):
    return np.concatenate([np.atleast_1d(v).ravel() for v in time_step.observation.values()])


def obs_dim_from_spec(spec):
    return sum(max(1, int(np.prod(v.shape))) for v in spec.values())


# ── CSV Logger ────────────────────────────────────────────────────────────────

class CSVLogger:
    def __init__(self, path, fieldnames):
        self.fieldnames = fieldnames
        self.file = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()

    def write(self, **kwargs):
        row = {k: "" for k in self.fieldnames}
        row.update(kwargs)
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


# ── Main training loop ────────────────────────────────────────────────────────

def train(
    total_steps=200_000,
    start_steps=5_000,
    batch_size=256,
    eval_every=10_000,
    eval_episodes=5,
    seed=42,
    csv_path="fast_slow_walker.csv",
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

    agent = FastSlowDDPG(obs_dim, act_dim, act_limit)
    buffer = ReplayBuffer()
    noise = OUNoise(act_dim)
    gating = RewardGating()

    logger = CSVLogger(csv_path, [
        "step", "event",
        "ep_return", "ep_len",
        "eval_return",
        "critic_loss", "actor_loss", "q_loss", "pullback_loss",
        "tau_eff", "pullback_lambda", "normalized_adv",
        "reward_running_mean", "reward_running_std",
        "actor_slow_dist",
    ])

    def evaluate():
        returns = []
        for _ in range(eval_episodes):
            ts = eval_env.reset()
            ep_ret = 0.0
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
    ep_losses = []
    tau_eff_last, lambda_last, adv_last = 0.0, 0.0, 0.0

    print(f"{'Step':>8}  {'EvalReturn':>12}  {'CriticLoss':>12}  {'ActorLoss':>12}  {'TauEff':>10}  {'PullbackL':>10}")
    print("-" * 75)

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

        if len(buffer) >= batch_size and step >= start_steps:
            losses = agent.update(buffer, batch_size)
            ep_losses.append(losses)

        if next_ts.last():
            gating.update(ep_ret)
            tau_eff_last, lambda_last, adv_last = gating.get_update_params(ep_ret)
            agent.update_slow_actor(tau_eff_last)
            agent.pullback_lambda = lambda_last

            avg = lambda key: np.mean([l[key] for l in ep_losses]) if ep_losses else float("nan")
            dist = agent.actor_slow_dist(buffer)

            logger.write(
                step=step, event="episode",
                ep_return=round(ep_ret, 4), ep_len=ep_len,
                critic_loss=round(float(avg("critic_loss")), 6),
                actor_loss=round(float(avg("actor_loss")), 6),
                q_loss=round(float(avg("q_loss")), 6),
                pullback_loss=round(float(avg("pullback_loss")), 6),
                tau_eff=round(tau_eff_last, 6),
                pullback_lambda=round(lambda_last, 6),
                normalized_adv=round(adv_last, 4),
                reward_running_mean=round(gating.mean, 4),
                reward_running_std=round(gating.std, 4),
                actor_slow_dist="" if np.isnan(dist) else round(dist, 6),
            )

            ts = env.reset()
            noise.reset()
            ep_ret, ep_len = 0.0, 0
            ep_losses = []
        else:
            ts = next_ts

        if step % eval_every == 0:
            eval_ret = evaluate()
            logger.write(step=step, event="eval", eval_return=round(eval_ret, 4))
            avg_c = np.mean([l["critic_loss"] for l in ep_losses]) if ep_losses else float("nan")
            avg_a = np.mean([l["actor_loss"] for l in ep_losses]) if ep_losses else float("nan")
            print(f"{step:>8}  {eval_ret:>12.2f}  {avg_c:>12.4f}  {avg_a:>12.4f}  {tau_eff_last:>10.5f}  {lambda_last:>10.5f}")

    logger.close()
    print("\nTraining complete.")


if __name__ == "__main__":
    train()
