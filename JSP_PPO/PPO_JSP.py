import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class Actor(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128):
        super(Actor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state_features, mask):
        encoded = self.encoder(state_features)
        scores = self.attention(encoded).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        return torch.softmax(scores, dim=-1)

class Critic(nn.Module):
    def __init__(self, n_jobs, n_machines, hidden_dim=128):
        super(Critic, self).__init__()
        self.op_encoder = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU())
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        ready_ops = state['ready_ops']
        if ready_ops.dim() == 2: ready_ops = ready_ops.unsqueeze(0)
        encoded = self.op_encoder(ready_ops)
        global_feat = encoded.mean(dim=1)
        return self.value_head(global_feat)

class PPO_JSP:
    def __init__(self, n_jobs, n_machines, lr=3e-4, gamma=0.99, lam=0.95, clip_epsilon=0.2, device='cuda'):
        self.device = device; self.gamma = gamma; self.lam = lam; self.clip_epsilon = clip_epsilon
        self.actor = Actor().to(device)
        self.critic = Critic(n_jobs, n_machines).to(device)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.buffer = []

    def select_action(self, state):
        ready_ops, mask = state['ready_ops'].unsqueeze(0), state['mask'].unsqueeze(0)
        with torch.no_grad(): probs = self.actor(ready_ops, mask)
        dist = Categorical(probs.squeeze(0))
        job_id = dist.sample()
        return job_id.item(), dist.log_prob(job_id).item()

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.buffer.append({'state': state, 'action': action, 'reward': reward,
                            'next_state': next_state, 'done': done, 'log_prob': log_prob})

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            next_value = 0 if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return torch.FloatTensor(advantages).to(self.device)

    def update(self, n_epochs=4, batch_size=32):
        if not self.buffer: return
        states = [t['state'] for t in self.buffer]
        actions = torch.LongTensor([t['action'] for t in self.buffer]).to(self.device)
        rewards = torch.FloatTensor([t['reward'] for t in self.buffer]).to(self.device)
        old_log_probs = torch.FloatTensor([t['log_prob'] for t in self.buffer]).to(self.device)
        dones = torch.FloatTensor([t['done'] for t in self.buffer]).to(self.device)

        with torch.no_grad(): values = torch.cat([self.critic(s) for s in states]).squeeze()
        advantages = self.compute_gae(rewards.cpu().numpy(), values.cpu().numpy(), dones.cpu().numpy())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(self.buffer)
        for epoch in range(n_epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                batch_states, batch_actions = [states[i] for i in batch_idx], actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = advantages[batch_idx] + values[batch_idx]

                new_log_probs = []
                for s, a in zip(batch_states, batch_actions):
                    probs = self.actor(s['ready_ops'].unsqueeze(0), s['mask'].unsqueeze(0)).squeeze(0)
                    new_log_probs.append(Categorical(probs).log_prob(a))
                new_log_probs = torch.stack(new_log_probs)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                current_values = torch.cat([self.critic(s) for s in batch_states]).squeeze()
                critic_loss = nn.MSELoss()(current_values, batch_returns)
                entropy = -(new_log_probs * torch.exp(new_log_probs)).mean()

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()
        self.buffer = []

    def save(self, path):
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])