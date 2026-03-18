"""
单动作PPO算法用于JSP问题
移除了第二个Actor网络，只保留工序选择策略
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class Actor(nn.Module):
    """工序选择策略网络（简化版，不使用GNN）"""

    def __init__(self, input_dim=4, hidden_dim=128):
        super(Actor, self).__init__()

        # 简单MLP编码器（替代GNN）
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 注意力机制计算每个就绪工序的分数
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state_features, mask):
        """
        Args:
            state_features: [batch, n_jobs, 4] 就绪工序特征
            mask: [batch, n_jobs] bool mask，True表示可选
        Returns:
            action_probs: [batch, n_jobs] 选择每个工件的概率
        """
        batch_size, n_jobs, _ = state_features.shape

        # 编码每个工序 [batch, n_jobs, hidden]
        encoded = self.encoder(state_features)

        # 计算注意力分数 [batch, n_jobs, 1]
        scores = self.attention(encoded).squeeze(-1)

        # 应用mask：不可选的工序设为极小值
        scores = scores.masked_fill(~mask, -1e9)

        # Softmax得到概率分布
        action_probs = torch.softmax(scores, dim=-1)

        return action_probs


class Critic(nn.Module):
    """价值网络：评估当前状态"""

    def __init__(self, n_jobs, n_machines, hidden_dim=128):
        super(Critic, self).__init__()

        # 全局状态编码
        self.op_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU()
        )

        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        """
        Args:
            state: dict with 'ready_ops' [batch, n_jobs, 4]
        Returns:
            value: [batch, 1] 状态价值估计
        """
        ready_ops = state['ready_ops']

        # 处理非批处理输入 [n_jobs, 4] -> [1, n_jobs, 4]
        if ready_ops.dim() == 2:
            ready_ops = ready_ops.unsqueeze(0)

        batch_size = ready_ops.shape[0]

        # 编码每个工序
        encoded = self.op_encoder(ready_ops)  # [batch, n_jobs, hidden]

        # 全局池化（取平均）
        global_feat = encoded.mean(dim=1)  # [batch, hidden]

        # 计算价值
        value = self.value_head(global_feat)

        return value


class PPO_JSP:
    def __init__(self, n_jobs, n_machines, lr=3e-4, gamma=0.99,
                 lam=0.95, clip_epsilon=0.2, device='cuda'):
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon

        # 单Actor + Critic（移除了第二个Actor）
        self.actor = Actor().to(device)
        self.critic = Critic(n_jobs, n_machines).to(device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )

        # 经验缓冲区
        self.buffer = []

    def select_action(self, state):
        """
        选择动作（仅选择工件，机器是固定的）

        Returns:
            job_id: 选择的工件ID
            log_prob: 动作对数概率（用于训练）
        """
        ready_ops = state['ready_ops'].unsqueeze(0)  # [1, n_jobs, 4]
        mask = state['mask'].unsqueeze(0)  # [1, n_jobs]

        with torch.no_grad():
            probs = self.actor(ready_ops, mask)  # [1, n_jobs]

        dist = Categorical(probs.squeeze(0))
        job_id = dist.sample()
        log_prob = dist.log_prob(job_id)

        return job_id.item(), log_prob.item()

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        """存储转移"""
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob
        })

    def compute_gae(self, rewards, values, dones):
        """计算广义优势估计"""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return torch.FloatTensor(advantages).to(self.device)

    def update(self, n_epochs=4, batch_size=32):
        """PPO更新（单动作版本）"""
        if len(self.buffer) == 0:
            return

        # 准备数据
        states = [t['state'] for t in self.buffer]
        actions = torch.LongTensor([t['action'] for t in self.buffer]).to(self.device)
        rewards = torch.FloatTensor([t['reward'] for t in self.buffer]).to(self.device)
        old_log_probs = torch.FloatTensor([t['log_prob'] for t in self.buffer]).to(self.device)
        dones = torch.FloatTensor([t['done'] for t in self.buffer]).to(self.device)

        # 计算价值估计
        with torch.no_grad():
            values = torch.cat([self.critic(s) for s in states]).squeeze()

        # 计算优势函数
        advantages = self.compute_gae(
            rewards.cpu().numpy(),
            values.cpu().numpy(),
            dones.cpu().numpy()
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 训练
        dataset_size = len(self.buffer)
        for epoch in range(n_epochs):
            indices = np.random.permutation(dataset_size)

            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                batch_states = [states[i] for i in batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = advantages[batch_idx] + values[batch_idx]

                # 计算新策略的概率
                new_log_probs = []
                for s, a in zip(batch_states, batch_actions):
                    ready_ops = s['ready_ops'].unsqueeze(0)
                    mask = s['mask'].unsqueeze(0)
                    probs = self.actor(ready_ops, mask).squeeze(0)
                    dist = Categorical(probs)
                    new_log_probs.append(dist.log_prob(a))

                new_log_probs = torch.stack(new_log_probs)

                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # 裁剪目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                current_values = torch.cat([self.critic(s) for s in batch_states]).squeeze()
                critic_loss = nn.MSELoss()(current_values, batch_returns)

                # 熵正则（鼓励探索）
                entropy = -(new_log_probs * torch.exp(new_log_probs)).mean()

                # 总损失
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()

        # 清空缓冲区
        self.buffer = []

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])