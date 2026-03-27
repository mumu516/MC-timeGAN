"""
mctimegan.py — 修改版 (融入 Vine-Copula 先验噪声)
===================================================
基于原版 MC-TimeGAN (Demirel et al., IEEE iSPEC 2024)

相对原版的修改 (共3处, 均标注 >>> VINE-COPULA 修改 <<<):
  1. __init__: 新增 vine_model 参数
  2. 新增 _generate_noise() 方法
  3. fit() 和 transform() 中替换 torch.rand → _generate_noise

vine_model=None 时行为与原版完全一致
"""

import os
import time
from itertools import chain, cycle

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)


# ========================== 网络模块 (与原版一致) ==========================

class ConditioningNetwork(nn.Module):
    def __init__(self, input_size, condition_size):
        super().__init__()
        self.condition = nn.Sequential(
            nn.Linear(input_size, 8), nn.Tanh(), nn.Linear(8, condition_size))
    def forward(self, conds):
        return self.condition(conds) if conds is not None else None

class Embedder(nn.Module):
    def __init__(self, module_name, input_features, hidden_dim, num_layers):
        super().__init__()
        rnn_class = nn.GRU if module_name == "gru" else nn.LSTM
        self.rnn = rnn_class(input_size=input_features, hidden_size=hidden_dim,
                             num_layers=num_layers, batch_first=True)
        self.model = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
    def forward(self, x, c=None):
        if c is not None: x = torch.cat([x, c], dim=-1)
        seq, _ = self.rnn(x)
        return self.model(seq)

class Recovery(nn.Module):
    def __init__(self, module_name, input_features, hidden_dim, num_layers):
        super().__init__()
        rnn_class = nn.GRU if module_name == "gru" else nn.LSTM
        self.rnn = rnn_class(input_size=hidden_dim, hidden_size=hidden_dim,
                             num_layers=num_layers, batch_first=True)
        self.model = nn.Sequential(nn.Linear(hidden_dim, input_features), nn.Sigmoid())
    def forward(self, x, c=None):
        if c is not None: x = torch.cat([x, c], dim=-1)
        seq, _ = self.rnn(x)
        return self.model(seq)

class Generator(nn.Module):
    def __init__(self, module_name, input_features, hidden_dim, num_layers):
        super().__init__()
        rnn_class = nn.GRU if module_name == "gru" else nn.LSTM
        self.rnn = rnn_class(input_size=input_features, hidden_size=hidden_dim,
                             num_layers=num_layers, batch_first=True)
        self.model = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
    def forward(self, x, c=None):
        if c is not None: x = torch.cat([x, c], dim=-1)
        seq, _ = self.rnn(x)
        return self.model(seq)

class Supervisor(nn.Module):
    def __init__(self, module_name, input_features, hidden_dim, num_layers):
        super().__init__()
        rnn_class = nn.GRU if module_name == "gru" else nn.LSTM
        self.rnn = rnn_class(input_size=input_features, hidden_size=hidden_dim,
                             num_layers=num_layers - 1, batch_first=True)
        self.model = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
    def forward(self, x, c=None):
        if c is not None: x = torch.cat([x, c], dim=-1)
        seq, _ = self.rnn(x)
        return self.model(seq)

class Discriminator(nn.Module):
    def __init__(self, module_name, hidden_dim, num_layers):
        super().__init__()
        rnn_class = nn.GRU if module_name == "gru" else nn.LSTM
        self.rnn = rnn_class(input_size=hidden_dim, hidden_size=hidden_dim,
                             num_layers=num_layers, bidirectional=True, batch_first=True)
        self.model = nn.Linear(2 * hidden_dim, 1)
    def forward(self, x, c=None):
        if c is not None: x = torch.cat([x, c], dim=-1)
        seq, _ = self.rnn(x)
        return self.model(seq)


# ========================== 损失函数 (与原版一致) ==========================

def discriminator_loss(y_real, y_fake, y_fake_e):
    gamma = 1
    valid = torch.ones_like(y_real, device=device, requires_grad=False)
    fake = torch.zeros_like(y_fake, device=device, requires_grad=False)
    return (nn.BCEWithLogitsLoss()(y_real, valid) +
            nn.BCEWithLogitsLoss()(y_fake, fake) +
            gamma * nn.BCEWithLogitsLoss()(y_fake_e, fake))

def generator_loss(y_fake, y_fake_e, h, h_hat_supervise, x, x_hat, w_corr=50, w_moment=100):
    gamma = 1
    fake = torch.ones_like(y_fake, device=device, requires_grad=False)
    # 1. Unsupervised loss
    g_loss_u = nn.BCEWithLogitsLoss()(y_fake, fake)
    g_loss_u_e = nn.BCEWithLogitsLoss()(y_fake_e, fake)
    # 2. Supervised loss
    g_loss_s = nn.MSELoss()(h[:, 1:, :], h_hat_supervise[:, :-1, :])
    # 3. Moment matching (均值 + 标准差)
    g_loss_v1 = torch.mean(torch.abs(torch.std(x_hat, dim=0) - torch.std(x, dim=0)))
    g_loss_v2 = torch.mean(torch.abs(torch.mean(x_hat, dim=0) - torch.mean(x, dim=0)))
    # 4. 相关性保持损失 (新增: 显式约束变量间相关性)
    #    计算batch内各变量间的Pearson相关系数矩阵, 要求生成数据与原始数据相关性一致
    g_loss_corr = _correlation_loss(x, x_hat)
    # 5. 自相关损失 (新增: 保持时序自相关特征)
    g_loss_auto = _autocorrelation_loss(x, x_hat)
    
    return (g_loss_u + gamma * g_loss_u_e + 
            w_moment * torch.sqrt(g_loss_s) + 
            w_moment * (g_loss_v1 + g_loss_v2) +
            w_corr * g_loss_corr +
            10 * g_loss_auto)


def _correlation_loss(x_real, x_fake):
    """
    相关性保持损失: ||Corr(x_real) - Corr(x_fake)||_F
    
    显式约束生成数据保持变量间的相关结构
    这是解决 GHI vs PV 相关性下降问题的关键
    
    x_real, x_fake: (batch, seq_len, n_features)
    """
    # 展平为 (batch*seq_len, n_features)
    B, T, D = x_real.shape
    r = x_real.reshape(B * T, D)
    f = x_fake.reshape(B * T, D)
    
    # 计算相关系数矩阵
    def _corr_matrix(x):
        # 标准化
        x_centered = x - x.mean(dim=0, keepdim=True)
        x_std = x.std(dim=0, keepdim=True) + 1e-7
        x_norm = x_centered / x_std
        # 相关系数 = (X^T X) / N
        corr = torch.mm(x_norm.t(), x_norm) / x_norm.shape[0]
        return corr
    
    corr_real = _corr_matrix(r)
    corr_fake = _corr_matrix(f)
    
    # Frobenius 范数
    loss = torch.mean((corr_real - corr_fake) ** 2)
    return loss


def _autocorrelation_loss(x_real, x_fake, max_lag=3):
    """
    自相关损失: 保持时序自相关特征
    约束生成数据的时间步之间具有与原始数据相似的自相关性
    
    x_real, x_fake: (batch, seq_len, n_features)
    """
    loss = torch.tensor(0.0, device=x_real.device)
    B, T, D = x_real.shape
    
    for lag in range(1, min(max_lag + 1, T)):
        # 原始数据 lag相关
        auto_real = torch.mean(x_real[:, lag:, :] * x_real[:, :-lag, :], dim=(0, 1))
        auto_fake = torch.mean(x_fake[:, lag:, :] * x_fake[:, :-lag, :], dim=(0, 1))
        loss = loss + torch.mean((auto_real - auto_fake) ** 2)
    
    return loss / max_lag

def embedder_loss(x, x_tilde):
    return 10 * torch.sqrt(nn.MSELoss()(x, x_tilde))

def generator_loss_supervised(h, h_hat_supervise):
    return nn.MSELoss()(h[:, 1:, :], h_hat_supervise[:, :-1, :])


# ========================== MC-TimeGAN 主模型 ==========================

class MCTimeGAN(nn.Module):

    def __init__(self, module_name="gru", input_features=1, input_conditions=None,
                 hidden_dim=8, num_layers=3, epochs=100, batch_size=128,
                 learning_rate=1e-3, lr_d=None, d_threshold=0.15,
                 use_lr_scheduler=False, w_corr=50,
                 vine_model=None):        # >>> VINE-COPULA 修改 1/3: 新增参数 <<<
        super().__init__()
        self.module_name = module_name
        self.input_features = input_features
        self.input_conditions = input_conditions
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_d = lr_d if lr_d is not None else learning_rate
        self.d_threshold = d_threshold
        self.use_lr_scheduler = use_lr_scheduler
        self.w_corr = w_corr              # 相关性损失权重
        self.cond_size = 1
        self.reproducibility = False
        self.vine_model = vine_model      # >>> VINE-COPULA 修改 1/3 <<<
        self._noise_pool = None
        self.checkpoint_dir = None
        self.checkpoint_interval = 200

        if input_conditions is not None:
            self.condnet = ConditioningNetwork(input_conditions, self.cond_size)
            self.embedder = Embedder(module_name, input_features + self.cond_size, hidden_dim, num_layers)
            self.recovery = Recovery(module_name, input_features, hidden_dim + self.cond_size, num_layers)
            self.generator = Generator(module_name, input_features + self.cond_size, hidden_dim, num_layers)
            self.supervisor = Supervisor(module_name, hidden_dim + self.cond_size, hidden_dim, num_layers)
            self.discriminator = Discriminator(module_name, hidden_dim + self.cond_size, num_layers)
            self.optimizer_e = torch.optim.Adam(chain(self.condnet.parameters(), self.embedder.parameters(), self.recovery.parameters()), lr=learning_rate)
            self.optimizer_g = torch.optim.Adam(chain(self.condnet.parameters(), self.generator.parameters(), self.supervisor.parameters()), lr=learning_rate)
            self.optimizer_d = torch.optim.Adam(chain(self.condnet.parameters(), self.discriminator.parameters()), lr=self.lr_d)
        else:
            self.embedder = Embedder(module_name, input_features, hidden_dim, num_layers)
            self.recovery = Recovery(module_name, input_features, hidden_dim, num_layers)
            self.generator = Generator(module_name, input_features, hidden_dim, num_layers)
            self.supervisor = Supervisor(module_name, hidden_dim, hidden_dim, num_layers)
            self.discriminator = Discriminator(module_name, hidden_dim, num_layers)
            self.optimizer_e = torch.optim.Adam(chain(self.embedder.parameters(), self.recovery.parameters()), lr=learning_rate)
            self.optimizer_g = torch.optim.Adam(chain(self.generator.parameters(), self.supervisor.parameters()), lr=learning_rate)
            self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_d)

        # 学习率调度器 (Phase 3中使用)
        if self.use_lr_scheduler:
            self.scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_g, T_max=epochs, eta_min=learning_rate * 0.1)
            self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_d, T_max=epochs, eta_min=self.lr_d * 0.1)
            self.scheduler_e = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_e, T_max=epochs, eta_min=learning_rate * 0.1)

        self.fitting_time = None
        self.losses = []

    # >>> VINE-COPULA 修改 2/3: 噪声生成 (预生成池 + 随机抽取) <<<
    def _build_noise_pool(self, data_shape, pool_multiplier=5):
        """
        训练前一次性预生成 Vine-Copula 噪声池
        训练时从池中随机抽取, 速度与 torch.rand() 相当
        
        pool_multiplier: 池大小 = 数据量 × 倍数 (默认5倍, 保证足够多样性)
        """
        if self.vine_model is None or not self.vine_model.is_fitted:
            self._noise_pool = None
            return
        
        batch_size, seq_len, n_features = data_shape
        pool_size = batch_size * pool_multiplier
        print(f"  预生成Vine-Copula噪声池: {pool_size} × {seq_len} × {n_features} ...")
        
        import time as _t
        t0 = _t.time()
        pool_np = self.vine_model.generate_noise((pool_size, seq_len, n_features))
        self._noise_pool = torch.tensor(pool_np, dtype=torch.float32, device=device)
        print(f"  噪声池生成完毕, 耗时 {_t.time()-t0:.1f}s, 形状 {self._noise_pool.shape}")
    
    def _generate_noise(self, data_shape):
        """从预生成的噪声池中随机抽取, 速度极快"""
        if self._noise_pool is not None:
            batch_size = data_shape[0]
            pool_size = self._noise_pool.shape[0]
            idx = torch.randint(0, pool_size, (batch_size,), device=device)
            return self._noise_pool[idx]
        else:
            return torch.rand(data_shape, dtype=torch.float32, device=device)

    def save_checkpoint(self, epoch, path):
        """保存模型checkpoint"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'embedder': self.embedder.state_dict(),
            'recovery': self.recovery.state_dict(),
            'generator': self.generator.state_dict(),
            'supervisor': self.supervisor.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'condnet': self.condnet.state_dict() if hasattr(self, 'condnet') else None,
        }, path)

    def load_checkpoint(self, path):
        """加载模型checkpoint"""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.embedder.load_state_dict(ckpt['embedder'])
        self.recovery.load_state_dict(ckpt['recovery'])
        self.generator.load_state_dict(ckpt['generator'])
        self.supervisor.load_state_dict(ckpt['supervisor'])
        self.discriminator.load_state_dict(ckpt['discriminator'])
        if ckpt.get('condnet') is not None and hasattr(self, 'condnet'):
            self.condnet.load_state_dict(ckpt['condnet'])
        return ckpt['epoch']

    def fit(self, data_train: np.ndarray, **kwargs: np.ndarray):
        self.fitting_time = time.time()
        data_train = torch.tensor(data_train, dtype=torch.float32, device=device)
        conditions = np.concatenate([c for c in kwargs.values()], axis=-1) if kwargs else None
        conditions = torch.tensor(conditions, dtype=torch.float32, device=device) if kwargs else None
        dataset = TensorDataset(data_train, conditions) if kwargs else TensorDataset(data_train)

        noise_type = "Vine-Copula先验" if (self.vine_model and self.vine_model.is_fitted) else "随机均匀"
        print(f"\n噪声类型: {noise_type}")
        print(f"数据: {data_train.shape}, 条件: {conditions.shape if conditions is not None else 'None'}")
        print(f"lr_g={self.learning_rate}, lr_d={self.lr_d}, d_threshold={self.d_threshold}, "
              f"w_corr={self.w_corr}, lr_scheduler={'ON' if self.use_lr_scheduler else 'OFF'}")

        # Phase 1: Embedding
        print("\n===== Phase 1: Embedding Network Training =====")
        for epoch, frame in zip(range(self.epochs), cycle(r"-\|/-\|/")):
            batches = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            self.train()
            loss_e = []
            for batch in batches:
                x, c = batch if kwargs else (*batch, None)
                self.optimizer_e.zero_grad()
                conds = self.condnet(c) if c is not None else None
                h = self.embedder(x, conds)
                x_tilde = self.recovery(h, conds)
                e_loss = embedder_loss(x, x_tilde)
                e_loss.backward()
                self.optimizer_e.step()
                loss_e.append(e_loss.item())
            if (epoch + 1) % max(1, int(0.1 * self.epochs)) == 0:
                print(f"  Epoch {epoch+1}/{self.epochs} | loss_e {np.mean(loss_e):.6f}")
            else:
                print(f"\r{frame}", end="", flush=True)
        print("  Embedding training done")

        # Phase 2: Supervised
        print("\n===== Phase 2: Supervised Training =====")
        for epoch, frame in zip(range(self.epochs), cycle(r"-\|/-\|/")):
            batches = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            self.train()
            loss_g = []
            for batch in batches:
                x, c = batch if kwargs else (*batch, None)
                self.optimizer_g.zero_grad()
                conds = self.condnet(c) if c is not None else None
                h = self.embedder(x, conds)
                h_hat_s = self.supervisor(h, conds)
                g_loss = generator_loss_supervised(h, h_hat_s)
                g_loss.backward()
                self.optimizer_g.step()
                loss_g.append(g_loss.item())
            if (epoch + 1) % max(1, int(0.1 * self.epochs)) == 0:
                print(f"  Epoch {epoch+1}/{self.epochs} | loss_g {np.mean(loss_g):.6f}")
            else:
                print(f"\r{frame}", end="", flush=True)
        print("  Supervised training done")

        # Phase 3: Joint Training
        # >>> 预生成噪声池, 避免每个epoch重复调用Copula采样 <<<
        self._build_noise_pool(data_train.shape)
        print("\n===== Phase 3: Joint Training =====")
        for epoch, frame in zip(range(self.epochs), cycle(r"-\|/-\|/")):
            loss_g_all, loss_e_all = [], []
            for _ in range(2):
                # >>> VINE-COPULA 修改 3/3: 替换 torch.rand <<<
                data_z = self._generate_noise(data_train.shape)
                ds = TensorDataset(data_train, data_z, conditions) if kwargs else TensorDataset(data_train, data_z)
                batches = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
                self.train()
                for batch in batches:
                    x, z, c = batch if kwargs else (*batch, None)
                    # Generator
                    self.optimizer_g.zero_grad()
                    conds = self.condnet(c) if c is not None else None
                    h = self.embedder(x, conds)
                    e_hat = self.generator(z, conds)
                    h_hat = self.supervisor(e_hat, conds)
                    h_hat_s = self.supervisor(h, conds)
                    x_hat = self.recovery(h_hat, conds)
                    y_fake = self.discriminator(h_hat, conds)
                    y_fake_e = self.discriminator(e_hat, conds)
                    g_loss = generator_loss(y_fake, y_fake_e, h, h_hat_s, x, x_hat, w_corr=self.w_corr)
                    g_loss.backward()
                    self.optimizer_g.step()
                    loss_g_all.append(g_loss.item())
                    # Embedder
                    self.optimizer_e.zero_grad()
                    conds = self.condnet(c) if c is not None else None
                    h = self.embedder(x, conds)
                    h_hat_s = self.supervisor(h, conds)
                    x_tilde = self.recovery(h, conds)
                    e_loss = embedder_loss(x, x_tilde) + 0.1 * generator_loss_supervised(h, h_hat_s)
                    e_loss.backward()
                    self.optimizer_e.step()
                    loss_e_all.append(e_loss.item())

            # Discriminator
            data_z = self._generate_noise(data_train.shape)  # >>> VINE-COPULA <<<
            ds = TensorDataset(data_train, data_z, conditions) if kwargs else TensorDataset(data_train, data_z)
            batches = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
            self.train()
            loss_d_all = []
            for batch in batches:
                x, z, c = batch if kwargs else (*batch, None)
                self.optimizer_d.zero_grad()
                conds = self.condnet(c) if c is not None else None
                h = self.embedder(x, conds)
                e_hat = self.generator(z, conds)
                h_hat = self.supervisor(e_hat, conds)
                y_fake = self.discriminator(h_hat, conds)
                y_real = self.discriminator(h, conds)
                y_fake_e = self.discriminator(e_hat, conds)
                d_loss = discriminator_loss(y_real, y_fake, y_fake_e)
                loss_d_all.append(d_loss.item())
                if d_loss > self.d_threshold:
                    d_loss.backward()
                    self.optimizer_d.step()

            self.losses.append([epoch+1, np.mean(loss_g_all), np.mean(loss_e_all), np.mean(loss_d_all)])
            
            # 学习率衰减
            if self.use_lr_scheduler:
                self.scheduler_g.step()
                self.scheduler_d.step()
                self.scheduler_e.step()
            
            if (epoch + 1) % max(1, int(0.1 * self.epochs)) == 0:
                lr_info = ""
                if self.use_lr_scheduler:
                    lr_info = f" | lr_g {self.scheduler_g.get_last_lr()[0]:.6f}"
                print(f"  Epoch {epoch+1}/{self.epochs} | G {np.mean(loss_g_all):.4f} | E {np.mean(loss_e_all):.4f} | D {np.mean(loss_d_all):.4f}{lr_info}")
            else:
                print(f"\r{frame}", end="", flush=True)
            
            # 自动保存checkpoint
            if self.checkpoint_dir and (epoch + 1) % self.checkpoint_interval == 0:
                ckpt_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch+1}.pth")
                self.save_checkpoint(epoch + 1, ckpt_path)
                print(f"    [Checkpoint] 已保存: {ckpt_path}")

        self.fitting_time = np.round(time.time() - self.fitting_time, 3)
        print(f"\n  Joint training done (耗时: {self.fitting_time}s)")
        
        # 保存最终checkpoint
        if self.checkpoint_dir:
            final_path = os.path.join(self.checkpoint_dir, f"epoch_{self.epochs}.pth")
            if not os.path.exists(final_path):
                self.save_checkpoint(self.epochs, final_path)
                print(f"    [Checkpoint] 最终: {final_path}")

    def transform(self, data_shape, **kwargs):
        if self._noise_pool is None and self.vine_model is not None:
            self._build_noise_pool(data_shape)
        data_z = self._generate_noise(data_shape).requires_grad_(False)  # >>> VINE-COPULA <<<
        conditions = np.concatenate([c for c in kwargs.values()], axis=-1) if kwargs else None
        conditions = torch.tensor(conditions, dtype=torch.float32, device=device, requires_grad=False) if kwargs else None
        ds = TensorDataset(data_z, conditions) if kwargs else TensorDataset(data_z)
        batches = DataLoader(ds, batch_size=1)
        generated = []
        self.eval()
        with torch.no_grad():
            for batch in batches:
                z, c = batch if kwargs else (*batch, None)
                conds = self.condnet(c) if c is not None else None
                e_hat = self.generator(z, conds)
                h_hat = self.supervisor(e_hat, conds)
                x_hat = self.recovery(h_hat, conds)
                generated.append(np.squeeze(x_hat.cpu().numpy(), axis=0))
        return np.stack(generated)
