"""
vine_copula.py — Vine-Copula 源端相关性建模
=============================================
建模 [风速, GHI, 温度, PV出力, 风电出力] 五变量联合分布,
生成保持非线性相关性的先验噪声, 替代 MC-TimeGAN 的 torch.rand()

对应开题报告公式(1): F(X1,...,Xd) = C[F(X1),...,F(Xd)]
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import rankdata, kendalltau


class VineCopulaModel:
    
    def __init__(self):
        self.vine = None
        self.marginal_ppfs = []
        self.n_vars = 0
        self.is_fitted = False
        self._use_gaussian = False
        self.corr_matrix = None
    
    def fit(self, data):
        """
        拟合 Vine-Copula 模型
        Args: data: (N, D) 原始数据, D=5
        """
        print(f"[VineCopula] 拟合模型, 数据维度: {data.shape}")
        self.n_vars = data.shape[1]
        n = data.shape[0]
        
        # Step 1: 边缘分布 → 伪观测值
        pseudo_obs = np.zeros_like(data, dtype=float)
        self.marginal_ppfs = []
        for j in range(self.n_vars):
            col = data[:, j].astype(float)
            ranks = rankdata(col, method='ordinal')
            pseudo_obs[:, j] = ranks / (n + 1)
            ecdf_x = np.sort(col)
            ecdf_y = np.arange(1, n + 1) / (n + 1)
            _, uid = np.unique(ecdf_y, return_index=True)
            ppf = interp1d(ecdf_y[uid], ecdf_x[uid], bounds_error=False,
                          fill_value=(ecdf_x[0], ecdf_x[-1]))
            self.marginal_ppfs.append(ppf)
        pseudo_obs = np.clip(pseudo_obs, 0.001, 0.999)
        
        # Step 2: 拟合 Vine-Copula
        try:
            import pyvinecopulib as pv
            controls = pv.FitControlsVinecop(
                family_set=[pv.BicopFamily.gaussian, pv.BicopFamily.student,
                           pv.BicopFamily.clayton, pv.BicopFamily.gumbel,
                           pv.BicopFamily.frank, pv.BicopFamily.joe],
                selection_criterion='aic')
            self.vine = pv.Vinecop.from_data(pseudo_obs, controls=controls)
            self._use_gaussian = False
            print(f"  Vine结构: {self.vine.structure}")
            print(f"  Copula族: {self.vine.families}")
        except ImportError:
            print("  [警告] pyvinecopulib未安装, 退化为Gaussian Copula")
            from scipy.stats import norm
            z = norm.ppf(pseudo_obs)
            z = np.nan_to_num(z, nan=0.0, posinf=3.0, neginf=-3.0)
            self.corr_matrix = np.corrcoef(z.T)
            self._use_gaussian = True
        
        self.is_fitted = True
        return self
    
    def sample_uniform(self, n_samples):
        if not self._use_gaussian:
            u = self.vine.simulate(n_samples)
        else:
            from scipy.stats import norm
            z = np.random.multivariate_normal(
                np.zeros(self.n_vars), self.corr_matrix, size=n_samples)
            u = norm.cdf(z)
        return np.clip(u, 0.001, 0.999)
    
    def generate_noise(self, shape):
        """
        生成 Vine-Copula 先验噪声, 替代 torch.rand()
        Args: shape: (batch_size, seq_len, n_features)
        Returns: np.float32 array, 值在[0,1], 变量间保持相关性
        """
        if not self.is_fitted:
            raise ValueError("请先调用 fit()")
        batch_size, seq_len, n_features = shape
        total = batch_size * seq_len
        u = self.sample_uniform(total)
        if n_features <= self.n_vars:
            noise_flat = u[:, :n_features]
        else:
            extra = np.random.uniform(0, 1, (total, n_features - self.n_vars))
            noise_flat = np.concatenate([u, extra], axis=1)
        return noise_flat.reshape(batch_size, seq_len, n_features).astype(np.float32)
    
    def validate(self, original_data, var_names=None):
        if var_names is None:
            var_names = [f'Var_{j}' for j in range(self.n_vars)]
        n_gen = min(len(original_data), 50000)
        u = self.sample_uniform(n_gen)
        gen = np.zeros_like(u)
        for j in range(self.n_vars):
            gen[:, j] = self.marginal_ppfs[j](u[:, j])
        orig = original_data[:n_gen]
        print("\n[VineCopula] Kendall τ 相关性验证:")
        errors = []
        for i in range(self.n_vars):
            for j in range(i + 1, self.n_vars):
                tau_o, _ = kendalltau(orig[:, i], orig[:, j])
                tau_g, _ = kendalltau(gen[:, i], gen[:, j])
                err = abs(tau_g - tau_o) / (abs(tau_o) + 1e-7) * 100
                errors.append(err)
                print(f"  {var_names[i]:20s} vs {var_names[j]:20s}: "
                      f"τ_orig={tau_o:.4f}, τ_gen={tau_g:.4f}, err={err:.1f}%")
        print(f"  平均误差: {np.mean(errors):.1f}%")
