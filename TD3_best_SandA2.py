import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
import warnings
from collections import deque, defaultdict
from datetime import datetime
from dataclasses import dataclass, asdict
import random
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# ====================================
# 🎯 EARLY STOPPING 超參數 (可調區)
# ====================================
EARLY_STOPPING_CONFIG = {
    'eval_interval': 10,      # 每 10 個 episode 評估一次
    'patience': 20,           # 連續 20 次沒改善就停止
    'min_episodes': 30,       # 至少訓練 30 個 episode
    'use_sharpe_ratio': True, # True: 用 Sharpe ratio, False: 用平均 reward
    'save_best_models': True  # 是否保存最佳模型
}

def setup_gpu():
    """GPU設置"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            gpu_to_use = gpus[3]
            tf.config.experimental.set_visible_devices(gpu_to_use, 'GPU')
            tf.config.experimental.set_memory_growth(gpu_to_use, True)
            print(f"✓ Using single GPU: {gpu_to_use.name}")
            return True
        except Exception as e:
            print(f"✗ GPU setup failed: {e}, falling back to CPU")
            return False
    else:
        print("✗ No GPU detected, using CPU")
        return False

def convert_numpy_types(obj):
    """遞歸轉換numpy類型為Python原生類型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

@dataclass
class EnhancedFinancialConfig:
    # —— 目標占比：Sharpe≈45%，Active≈45%，MDD≈8%，Cash≈2% ——
    sharpe_weight: float = 0.45         # ↑
    return_weight: float = 0.45         # ↑
    mdd_weight: float = 0.08            # ↓
    cash_timing_weight: float = 0.02    # ↓

    # 風險控制參數
    target_max_drawdown: float = 0.15
    target_sharpe_threshold: float = 0.5

    # 現金管理參數
    cash_return_rate: float = 0.02
    optimal_cash_range: tuple = (0.05, 0.25)  # 收窄到 5%~25%

    # EMA 參數
    alpha_fast: float = 0.08
    alpha_slow: float = 0.05

    # 縮放係數（scale）
    sharpe_scale: float = 6.0           # ↑
    active_selection_scale: float = 20.0# ↑
    mdd_penalty_scale: float = 2.0
    cash_timing_scale: float = 0.10     # ↓ 大幅降

    
    def output_objectives(self):
        """🎯 Quant Model Layer 輸出：量化模型定義的目標"""
        return {
            "objectives": {
                "sharpe_weight": self.sharpe_weight,
                "return_weight": self.return_weight,
                "mdd_weight": self.mdd_weight,
                "cash_timing_weight": self.cash_timing_weight,
                "approach": "active_selection_fusion"
            },
            "risk_controls": {
                "target_max_drawdown": self.target_max_drawdown,
                "target_sharpe_threshold": self.target_sharpe_threshold
            },
            "cash_params": {
                "cash_return_rate": self.cash_return_rate,
                "optimal_cash_range": list(self.optimal_cash_range)
            },
            "scaling_factors": {
                "sharpe_scale": self.sharpe_scale,
                "active_selection_scale": self.active_selection_scale,
                "mdd_penalty_scale": self.mdd_penalty_scale,
                "cash_timing_scale": self.cash_timing_scale
            }
        }

class EnhancedFinancialRewardCalculator:
    """融合版獎勵計算器：Sharpe + ActiveSelection + MDD + CashTiming"""
    
    def __init__(self, config: EnhancedFinancialConfig, n_stocks, lookback_window=20):
        self.config = config
        self.n_stocks = n_stocks
        self.n_assets = n_stocks + 1  # 包括現金
        self.lookback_window = lookback_window
        
        # 核心追蹤數據
        self.portfolio_returns_history = deque(maxlen=lookback_window)
        self.benchmark_returns_history = deque(maxlen=lookback_window)
        self.weights_history = deque(maxlen=lookback_window)
        
        # EMA追蹤
        self.portfolio_return_ema = 0.0
        self.portfolio_var_ema = 0.01
        self.benchmark_return_ema = 0.0
        self.excess_return_ema = 0.0
        self.excess_var_ema = 0.01
        
        # 🆕 MDD 追蹤變數
        self.mdd_portfolio_values = []
        self.mdd_running_peak = 1.0  # 從1.0開始，代表初始淨值
        
        # 現金相關
        self.daily_cash_return = config.cash_return_rate / 252
        self.market_trend_ema = 0.0
        
        print("✓ Enhanced Financial Reward Calculator initialized (Fusion Version)")
        print(f"  Focus: Sharpe({config.sharpe_weight:.0%}) + ActiveSelection({config.return_weight:.0%}) + MDD({config.mdd_weight:.0%}) + Cash({config.cash_timing_weight:.0%})")
    
    def reset(self):
        """重置計算器狀態"""
        self.portfolio_returns_history.clear()
        self.benchmark_returns_history.clear()
        self.weights_history.clear()
        self.portfolio_return_ema = 0.0
        self.portfolio_var_ema = 0.01
        self.benchmark_return_ema = 0.0
        self.excess_return_ema = 0.0
        self.excess_var_ema = 0.01
        self.market_trend_ema = 0.0
        
        # 重置MDD變數
        self.mdd_portfolio_values = []
        self.mdd_running_peak = 1.0
    
    def calculate_enhanced_reward(self, portfolio_value, portfolio_return, weights, benchmark_return, stock_returns, step):
        """計算融合版獎勵"""
        
        # 更新MDD的歷史數據
        current_net_value = portfolio_value / 10000.0  # 標準化淨值
        self.mdd_portfolio_values.append(current_net_value)
        self.mdd_running_peak = max(self.mdd_running_peak, current_net_value)
        
        # 更新其他歷史數據
        self.portfolio_returns_history.append(portfolio_return)
        self.benchmark_returns_history.append(benchmark_return)
        self.weights_history.append(weights.copy())
        
        # 更新EMA
        self._update_ema(portfolio_return, benchmark_return)
        
        # 🎯 核心獎勵成分（融合版）
        reward_components = {}
        
        # 1. Sharpe Ratio最大化 (40%)
        reward_components['sharpe'] = self._calculate_sharpe_reward()
        
        # 2. 個股超額回報獎勵 (30%) - 融合版核心
        reward_components['active_selection'] = self._calculate_active_selection_reward(weights, stock_returns)
        
        # 3. 最大回撤懲罰 (15%)
        reward_components['mdd_penalty'] = self._calculate_mdd_penalty()
        
        # 4. 現金時機選擇 (15%)
        reward_components['cash_timing'] = self._calculate_cash_timing_reward(weights[-1], stock_returns)
        
        # 組合最終獎勵
        final_reward = self._combine_enhanced_rewards(reward_components)
        
        return final_reward, reward_components
    
    def _calculate_sharpe_reward(self):
        if len(self.portfolio_returns_history) < 10:
            return 0.0
        std = np.sqrt(max(self.portfolio_var_ema - self.portfolio_return_ema**2, 1e-6))
        excess = self.portfolio_return_ema - self.daily_cash_return
        sharpe = excess / std
        sharpe = np.clip(sharpe, -3.0, 3.0)  # 抑制極端
        bonus = max(0.0, (sharpe - self.config.target_sharpe_threshold) * 2.0)
        return (sharpe + bonus) * self.config.sharpe_scale

    
    def _calculate_active_selection_reward(self, weights, stock_returns):
        market_avg = np.mean(stock_returns)
        excess = stock_returns - market_avg
        cs_std = np.std(excess)
        cs_std = max(cs_std, 1e-5)

        stock_weights = weights[:-1]  # exclude cash
        active_signal = np.dot(stock_weights, excess) / cs_std   # 標準化後
        return active_signal * self.config.active_selection_scale

    
    def _calculate_mdd_penalty(self):
        """計算最大回撤懲罰"""
        if not self.mdd_portfolio_values:
            return 0.0

        # 計算當前回撤百分比（會是0或負數）
        current_drawdown = (self.mdd_portfolio_values[-1] - self.mdd_running_peak) / self.mdd_running_peak
        
        # 回撤越大，懲罰越大
        penalty = current_drawdown * self.config.mdd_penalty_scale
        
        return penalty
    
    def _calculate_cash_timing_reward(self, cash_ratio, stock_returns):
        mkt = float(np.mean(stock_returns))
        self.market_trend_ema = (1-self.config.alpha_fast) * self.market_trend_ema + self.config.alpha_fast * mkt

        # 死區：小波動不給分
        dead_zone = 0.01
        extreme = 0.03

        if mkt <= -extreme:         # 急跌→多現金好
            timing_score = cash_ratio
        elif mkt >=  extreme:       # 急漲→少現金好
            timing_score = 1.0 - cash_ratio
        elif abs(mkt) <= dead_zone: # 小波動→不獎不罰
            timing_score = 0.0
        else:
            # 緩和區：線性插值，避免大幅度
            frac = (abs(mkt)-dead_zone)/(extreme-dead_zone)
            if mkt > 0:
                timing_score = (1.0 - cash_ratio) * frac
            else:
                timing_score = cash_ratio * frac

        # 限幅，確保現金項不主宰
        timing_score = np.clip(timing_score, -0.5, 0.5)
        return timing_score * self.config.cash_timing_scale

    
    def _update_ema(self, portfolio_return, benchmark_return):
        """更新EMA追蹤指標"""
        alpha_fast = self.config.alpha_fast
        alpha_slow = self.config.alpha_slow
        
        # 投資組合EMA
        self.portfolio_return_ema = (1-alpha_slow) * self.portfolio_return_ema + alpha_slow * portfolio_return
        self.portfolio_var_ema = (1-alpha_slow) * self.portfolio_var_ema + alpha_slow * portfolio_return**2
        
        # 基準EMA
        self.benchmark_return_ema = (1-alpha_slow) * self.benchmark_return_ema + alpha_slow * benchmark_return
        
        # 超額收益EMA
        excess_return = portfolio_return - benchmark_return
        self.excess_return_ema = (1-alpha_slow) * self.excess_return_ema + alpha_slow * excess_return
        self.excess_var_ema = (1-alpha_slow) * self.excess_var_ema + alpha_slow * excess_return**2
    
    def _combine_enhanced_rewards(self, reward_components):
        """組合融合版獎勵"""
        weights = {
            'sharpe': self.config.sharpe_weight,
            'active_selection': self.config.return_weight,
            'mdd_penalty': self.config.mdd_weight,
            'cash_timing': self.config.cash_timing_weight,
        }
        
        total_reward = sum(reward_components[key] * weights[key] for key in reward_components.keys())
        return total_reward

class ReplayBuffer:
    """經驗回放緩衝區 for TD3"""
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class TD3Agent:
    """🆕 Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent with Early Stopping"""
    def __init__(self, state_dim, action_dim, learning_rate=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # TD3 超參數
        self.gamma = 0.99
        self.tau = 0.005                    # 軟更新係數
        self.batch_size = 256
        self.policy_delay = 2               # 延遲策略更新
        self.policy_noise = 0.05             # 目標策略平滑噪聲
        self.noise_clip = 0.10               # 噪聲裁剪
        self.exploration_noise = 0.03        # 探索噪聲
        self.max_grad_norm = 1.0            # 梯度裁剪
        
        # 訓練計數器
        self.training_step = 0
        
        # 🆕 構建TD3網絡架構
        self.actor = self._build_actor()                    # 確定性策略
        self.actor_target = self._build_actor()             # 目標策略
        self.critic1 = self._build_critic()                 # Twin Critic 1
        self.critic2 = self._build_critic()                 # Twin Critic 2
        self.critic1_target = self._build_critic()          # 目標 Critic 1
        self.critic2_target = self._build_critic()          # 目標 Critic 2
        
        # 初始化目標網絡權重
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic1_target.set_weights(self.critic1.get_weights())
        self.critic2_target.set_weights(self.critic2.get_weights())
        
        # 優化器
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.critic1_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.critic2_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # 經驗回放
        self.replay_buffer = ReplayBuffer()
        
        print("✓ TD3 Agent initialized (Twin Delayed DDPG with Early Stopping)")
        print(f"  State dim: {state_dim}, Action dim: {action_dim}")
        print(f"  🆕 Networks: Actor + Twin Critics + 3×Target Networks")
        print(f"  🎯 TD3 Features: Policy Delay + Target Smoothing + Twin Critics")
        print(f"  ❌ No entropy term (deterministic policy)")
        print(f"  🔄 Early Stopping Support: Model save/load enabled")
    
    def _build_actor(self):
        """構建確定性Actor網絡"""
        inputs = keras.Input(shape=(self.state_dim,))
        
        x = keras.layers.Dense(512, activation='relu')(inputs)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        
        # TD3: 確定性輸出，使用tanh激活
        actions = keras.layers.Dense(self.action_dim, activation='tanh')(x)
        
        model = keras.Model(inputs=inputs, outputs=actions)
        return model
    
    def _build_critic(self):
        """構建Critic網絡 - Q(s,a)"""
        state_input = keras.Input(shape=(self.state_dim,))
        action_input = keras.Input(shape=(self.action_dim,))
        
        concat = keras.layers.Concatenate()([state_input, action_input])
        
        x = keras.layers.Dense(512, activation='relu')(concat)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        q_value = keras.layers.Dense(1, activation='linear')(x)
        
        model = keras.Model(inputs=[state_input, action_input], outputs=q_value)
        return model
    
    def get_action(self, state, add_noise=True):
        """獲取確定性動作（可選探索噪聲）"""
        if len(state.shape) == 1:
            state = tf.expand_dims(state, 0)
        
        action = self.actor(state)[0]
        
        # 探索噪聲（訓練時）
        if add_noise:
            noise = tf.random.normal(shape=tf.shape(action), stddev=self.exploration_noise)
            action = action + noise
            action = tf.clip_by_value(action, -1.0, 1.0)
        
        action_info = {
            "state": state[0].numpy().tolist(),
            "action": action.numpy().tolist(),
            "add_noise": add_noise,
            "exploration_noise": self.exploration_noise
        }
        
        return action.numpy(), action_info
    
    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def soft_update(self, target_model, source_model):
        """軟更新目標網絡"""
        target_weights = target_model.get_weights()
        source_weights = source_model.get_weights()
        
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * source_weights[i] + (1 - self.tau) * target_weights[i]
        
        target_model.set_weights(target_weights)
    
    def save_best_models(self, save_dir):
        """🆕 保存最佳模型權重"""
        try:
            os.makedirs(save_dir, exist_ok=True)
            self.actor.save(f'{save_dir}/best_actor.keras')
            self.critic1.save(f'{save_dir}/best_critic1.keras')
            self.critic2.save(f'{save_dir}/best_critic2.keras')
            print(f"✅ Best models saved to {save_dir}")
        except Exception as e:
            print(f"⚠️ Failed to save best models: {e}")
    
    def load_best_models(self, save_dir):
        """🆕 載入最佳模型權重"""
        try:
            if os.path.exists(f'{save_dir}/best_actor.keras'):
                self.actor = keras.models.load_model(f'{save_dir}/best_actor.keras')
                print(f"✅ Best actor loaded from {save_dir}")
            if os.path.exists(f'{save_dir}/best_critic1.keras'):
                self.critic1 = keras.models.load_model(f'{save_dir}/best_critic1.keras')
                print(f"✅ Best critic1 loaded from {save_dir}")
            if os.path.exists(f'{save_dir}/best_critic2.keras'):
                self.critic2 = keras.models.load_model(f'{save_dir}/best_critic2.keras')
                print(f"✅ Best critic2 loaded from {save_dir}")
        except Exception as e:
            print(f"⚠️ Failed to load best models: {e}")
    
    def train(self):
        """🆕 TD3 訓練流程"""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        try:
            # 從回放緩衝區採樣
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)
            
            # 檢查數據有效性
            if (tf.reduce_any(tf.math.is_nan(states)) or tf.reduce_any(tf.math.is_nan(actions)) or 
                tf.reduce_any(tf.math.is_nan(rewards)) or tf.reduce_any(tf.math.is_nan(next_states))):
                return {'actor_loss': 0.0, 'critic1_loss': 0.0, 'critic2_loss': 0.0}
            
            rewards = tf.clip_by_value(rewards, -10.0, 10.0)
            
            # 🎯 第一步：訓練Critics（每步都訓練）
            critic1_loss, critic2_loss = self._train_critics(states, actions, rewards, next_states, dones)
            
            actor_loss = None
            # 🎯 第二步：延遲訓練Actor（每policy_delay步訓練一次）
            if self.training_step % self.policy_delay == 0:
                actor_loss = self._train_actor(states)
                # 軟更新所有目標網絡
                self.soft_update(self.actor_target, self.actor)
                self.soft_update(self.critic1_target, self.critic1)
                self.soft_update(self.critic2_target, self.critic2)
            
            self.training_step += 1
            
            return {
                'actor_loss': float(actor_loss) if actor_loss is not None else 0.0,
                'critic1_loss': float(critic1_loss),
                'critic2_loss': float(critic2_loss),
                'avg_q1_value': float(tf.reduce_mean(self.critic1([states, actions]))),
                'avg_q2_value': float(tf.reduce_mean(self.critic2([states, actions]))),
                'training_step': self.training_step
            }
            
        except Exception as e:
            print(f"⚠️ TD3 Training error: {e}")
            return {'actor_loss': 0.0, 'critic1_loss': 0.0, 'critic2_loss': 0.0}
    
    def _train_critics(self, states, actions, rewards, next_states, dones):
        """訓練Twin Critics"""
        with tf.GradientTape(persistent=True) as tape:
            # 🎯 TD3 目標策略平滑 (Target Policy Smoothing)
            target_actions = self.actor_target(next_states)
            
            # 添加裁剪噪聲到目標動作
            noise = tf.random.normal(shape=tf.shape(target_actions), stddev=self.policy_noise)
            noise = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip)
            target_actions = tf.clip_by_value(target_actions + noise, -1.0, 1.0)
            
            # 🎯 TD3 雙延遲Q學習 (Clipped Double Q-Learning)
            target_q1 = self.critic1_target([next_states, target_actions])
            target_q2 = self.critic2_target([next_states, target_actions])
            target_q = tf.minimum(target_q1, target_q2)  # 取最小值減少過估計
            target_q = rewards + self.gamma * (1 - dones) * tf.squeeze(target_q)
            target_q = tf.clip_by_value(target_q, -50.0, 50.0)
            
            # 當前Q值
            current_q1 = tf.squeeze(self.critic1([states, actions]))
            current_q2 = tf.squeeze(self.critic2([states, actions]))
            
            # Critics損失
            critic1_loss = tf.reduce_mean((current_q1 - target_q) ** 2)
            critic2_loss = tf.reduce_mean((current_q2 - target_q) ** 2)
        
        # 更新Critic1
        critic1_grads = tape.gradient(critic1_loss, self.critic1.trainable_variables)
        if critic1_grads:
            critic1_grads = [tf.clip_by_norm(g, self.max_grad_norm) for g in critic1_grads if g is not None]
            self.critic1_optimizer.apply_gradients(zip(critic1_grads, self.critic1.trainable_variables))
        
        # 更新Critic2
        critic2_grads = tape.gradient(critic2_loss, self.critic2.trainable_variables)
        if critic2_grads:
            critic2_grads = [tf.clip_by_norm(g, self.max_grad_norm) for g in critic2_grads if g is not None]
            self.critic2_optimizer.apply_gradients(zip(critic2_grads, self.critic2.trainable_variables))
        
        del tape
        return critic1_loss, critic2_loss
    
    def _train_actor(self, states):
        """訓練Actor（延遲更新）"""
        with tf.GradientTape() as tape:
            # 使用Critic1計算策略梯度（TD3只用其中一個Critic）
            actions = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic1([states, actions]))
        
        # 更新Actor
        if not tf.math.is_nan(actor_loss):
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            if actor_grads:
                actor_grads = [tf.clip_by_norm(g, self.max_grad_norm) for g in actor_grads if g is not None]
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        return actor_loss

def evaluate_on_validation(agent, val_env):
    """
    🆕 在驗證集上評估模型性能
    返回 Sharpe ratio 或平均 reward (根據 EARLY_STOPPING_CONFIG 決定)
    """
    state = val_env.reset()
    rewards = []
    done = False
    
    while not done:
        # 驗證時不加噪聲
        action, _ = agent.get_action(state, add_noise=False)
        state, reward, done, _ = val_env.step(action)
        rewards.append(reward)
    
    rewards = np.array(rewards)
    
    if EARLY_STOPPING_CONFIG['use_sharpe_ratio']:
        # 計算 Sharpe ratio
        if len(rewards) > 1:
            mean_return = np.mean(rewards)
            std_return = np.std(rewards) + 1e-8  # 避免除零
            sharpe_ratio = mean_return / std_return * np.sqrt(252)  # 年化 Sharpe
            return sharpe_ratio
        else:
            return 0.0
    else:
        # 返回平均 reward
        return np.mean(rewards)

class EnhancedPortfolioEnvironment:
    """增強版投資組合環境 - 豐富狀態特徵 + 融合版獎勵函數"""
    
    def __init__(self, csv_file_path, mode='train', config=None):
        self.csv_file_path = csv_file_path
        self.mode = mode
        
        # 載入並預處理數據
        print(f"📊 Loading and preprocessing data for enhanced features...")
        self._load_and_preprocess_enhanced()
        
        # 設置配置
        if config is None:
            config = EnhancedFinancialConfig()
        self.config = config
        
        # 設置現金收益率
        self.daily_cash_return = config.cash_return_rate / 252
        
        # 環境狀態
        self.current_step = 0
        self.portfolio_value = 10000.0
        
        # 動態初始權重：股票等權重80%，現金20%
        stock_weight = 0.8 / self.n_stocks
        self.initial_weights = np.array([stock_weight] * self.n_stocks + [0.2], dtype=np.float32)
        
        self.weights = self.initial_weights.copy()
        self.weights_history = []
        self.portfolio_values_history = []
        
        # 融合版獎勵計算器
        self.reward_calculator = EnhancedFinancialRewardCalculator(
            config=config,
            n_stocks=self.n_stocks,
            lookback_window=20
        )
        
        # 追蹤詳細獎勵信息
        self.reward_history = []
        
        print(f"✓ Enhanced Portfolio Environment initialized (Fusion Version)")
        print(f"✓ State dim: {self.state_dim}, Action dim: {self.action_dim}")
        print(f"✓ Features per stock: 5 (mom5d, mom20d, vol20d, rsi, bollinger_b)")
        print(f"✓ Total trading days: {self.n_days}")
    
    def _load_and_preprocess_enhanced(self):
        """載入並預處理數據 - 增加豐富的技術特徵"""
        
        # 1. 載入原始數據
        df = pd.read_csv(self.csv_file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.ffill().dropna()
        
        print(f"✓ Loaded price data: {df.shape}")
        
        # 2. 計算回報率
        returns = df.pct_change().fillna(0)
        
        # 3. 計算豐富的技術特徵
        print(f"📈 Computing enhanced technical features...")
        
        # 核心擴展：動能和波動性
        momentum_5d = returns.rolling(window=5).mean().fillna(0)
        momentum_20d = returns.rolling(window=20).mean().fillna(0)
        volatility_20d = returns.rolling(window=20).std().fillna(0)
        
        # 技術指標：RSI
        delta = returns.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50) / 100.0  # 標準化到 [0,1]
        
        # 技術指標：布林帶 %B
        ma_20 = df.rolling(window=20).mean()
        std_20 = df.rolling(window=20).std()
        upper_band = ma_20 + (2 * std_20)
        lower_band = ma_20 - (2 * std_20)
        bollinger_b = (df - lower_band) / (upper_band - lower_band + 1e-8)
        bollinger_b = bollinger_b.fillna(0.5)  # 用0.5填充（中位值）
        
        # 4. 合併所有特徵
        self.feature_df = pd.concat([
            momentum_5d.add_suffix('_mom5d'),
            momentum_20d.add_suffix('_mom20d'),
            volatility_20d.add_suffix('_vol20d'),
            rsi.add_suffix('_rsi'),
            bollinger_b.add_suffix('_bollinger_b'),
        ], axis=1)
            
        # --- 關鍵修正：將所有特徵數據向下平移一天 ---
        # 這樣在 t 日刻做決策時，我們用的是 t-1 日的特徵
        self.feature_df = self.feature_df#.shift(1)
        
        # 平移後第一行會是 NaN，需要處理
        # 我們同時也平移回報率數據，以確保對齊
        self.price_data = df.copy()
        self.returns_data = returns.copy().shift(-1)

        # 重新對齊並去除因平移產生的NaN值
        combined = pd.concat([self.feature_df, self.returns_data], axis=1).dropna()

        self.feature_df = combined[self.feature_df.columns]
        self.returns_data = combined[self.returns_data.columns]

        print(f"✅ Features shifted to prevent lookahead bias.")
     
        # 6. 獲取股票數量和名稱
        self.n_stocks = df.shape[1]
        self.stock_names = df.columns.tolist()
        self.n_assets = self.n_stocks + 1
        
        print(f"✓ Enhanced features computed: {self.feature_df.shape}")
        print(f"✓ Stocks detected: {self.n_stocks}")
        
        # 7. 數據集分割
        total = len(self.feature_df)
        train_end = int(0.7 * total)
        val_end = int(0.8 * total)
        
        if self.mode == 'train':
            self.features = self.feature_df.iloc[:train_end].values
            self.returns = self.returns_data.iloc[:train_end].values
            self.price_subset = df.iloc[:train_end]
        elif self.mode == 'val':
            self.features = self.feature_df.iloc[train_end:val_end].values
            self.returns = self.returns_data.iloc[train_end:val_end].values
            self.price_subset = df.iloc[train_end:val_end]
        else:  # test
            self.features = self.feature_df.iloc[val_end:].values
            self.returns = self.returns_data.iloc[val_end:].values
            self.price_subset = df.iloc[val_end:]
        
        self.n_days = len(self.features)
        
        # 8. 重新計算state_dim
        # 每個股票有5個特徵 + n_assets個當前權重 + 1個現金回報率
        self.state_dim = (self.n_stocks * 5) + self.n_assets + 1
        self.action_dim = self.n_assets
        
        print(f"✓ Data split completed: {self.n_days} days for {self.mode}")
        print(f"✓ State dimension: {self.state_dim}")
    
    def reset(self):
        """重置環境"""
        self.current_step = 0
        self.portfolio_value = 10000.0
        self.weights = self.initial_weights.copy()
        self.weights_history = []
        self.portfolio_values_history = [self.portfolio_value]
        self.reward_history = []
        
        # 重置獎勵計算器
        self.reward_calculator.reset()
        
        return self._get_current_state()
    
    def _get_current_state(self):
        """獲取當前的增強版狀態"""
        if self.current_step >= len(self.features):
            return np.zeros(self.state_dim, dtype=np.float32)
        
        # 1. 股票技術特徵 (n_stocks * 5)
        stock_features = self.features[self.current_step]
        
        # 2. 現金特徵 (1)
        cash_feature = np.array([self.daily_cash_return])
        
        # 3. 當前持倉權重 (n_assets)
        current_weights = self.weights
        
        # 4. 合併成完整狀態
        state = np.concatenate([stock_features, cash_feature, current_weights]).astype(np.float32)
        
        return state
    
    def step(self, action):
        """執行一步 - 使用融合版獎勵函數"""
        
        # 添加NaN檢查
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            action = np.ones(self.action_dim) / self.action_dim
            
        # 將連續動作轉換為有效的投資組合權重
        try:
            action_clipped = np.clip(action, -10, 10)
            exp_action = np.exp(action_clipped - np.max(action_clipped))
            self.weights = exp_action / (np.sum(exp_action) + 1e-8)
            
            if np.any(np.isnan(self.weights)) or np.any(np.isinf(self.weights)) or np.sum(self.weights) < 0.99:
                self.weights = self.initial_weights.copy()
                
        except Exception as e:
            self.weights = self.initial_weights.copy()
        
        self.weights_history.append(self.weights.copy())
        
        # 計算投資組合收益
        stock_returns = self.returns[self.current_step]
        cash_return = self.daily_cash_return
        
        if np.any(np.isnan(stock_returns)) or np.any(np.isinf(stock_returns)):
            stock_returns = np.zeros_like(stock_returns)
        
        # 股票部分收益
        stock_portfolio_return = np.dot(self.weights[:-1], stock_returns)
        
        # 現金部分收益
        cash_portfolio_return = self.weights[-1] * cash_return
        
        # 總投資組合收益
        portfolio_return = stock_portfolio_return + cash_portfolio_return
        portfolio_return = np.clip(portfolio_return, -0.5, 0.5)
        
        # 計算基準收益
        equal_stock_weight = 0.8 / self.n_stocks
        benchmark_weights = np.array([equal_stock_weight] * self.n_stocks + [0.2])
        benchmark_stock_return = np.dot(benchmark_weights[:-1], stock_returns)
        benchmark_cash_return = benchmark_weights[-1] * cash_return
        benchmark_return = benchmark_stock_return + benchmark_cash_return
        benchmark_return = np.clip(benchmark_return, -0.5, 0.5)
        
        # 更新投資組合價值
        self.portfolio_value *= (1 + portfolio_return)
        
        if np.isnan(self.portfolio_value) or np.isinf(self.portfolio_value) or self.portfolio_value <= 0:
            self.portfolio_value = self.portfolio_values_history[-1] if self.portfolio_values_history else 10000.0
            
        self.portfolio_values_history.append(self.portfolio_value)
        
        # 使用融合版獎勵函數
        try:
            reward, reward_components = self.reward_calculator.calculate_enhanced_reward(
                portfolio_value=self.portfolio_value,
                portfolio_return=portfolio_return,
                weights=self.weights,
                benchmark_return=benchmark_return,
                stock_returns=stock_returns,
                step=self.current_step
            )
            
            if np.isnan(reward) or np.isinf(reward):
                reward = 0.0
                reward_components = {k: 0.0 for k in ['sharpe', 'active_selection', 'mdd_penalty', 'cash_timing']}
                
        except Exception as e:
            reward = 0.0
            reward_components = {k: 0.0 for k in ['sharpe', 'active_selection', 'mdd_penalty', 'cash_timing']}
        
        # 記錄詳細獎勵信息
        self.reward_history.append({
            'step': self.current_step,
            'total_reward': reward,
            'components': reward_components.copy()
        })
        
        self.current_step += 1
        done = self.current_step >= self.n_days - 1
        
        # 構造下一個狀態
        next_state = self._get_current_state() if not done else np.zeros(self.state_dim, dtype=np.float32)
        
        # 詳細的環境信息輸出
        info = {
            'portfolio_value': self.portfolio_value,
            'weights': self.weights.copy(),
            'portfolio_return': portfolio_return,
            'benchmark_return': benchmark_return,
            'excess_return': portfolio_return - benchmark_return,
            'cash_ratio': self.weights[-1],
            'reward_components': reward_components,
            'total_reward': reward,
            'trading_day': self.current_step,  # 🆕 添加交易日信息
            
            "market_conditions": {
                "market_return": float(np.mean(stock_returns)),
                "market_volatility": float(np.std(stock_returns)),
                "individual_stock_returns": stock_returns.tolist(),
                "cash_return": float(cash_return),
                "market_trend": "bullish" if np.mean(stock_returns) > 0.01 else "bearish" if np.mean(stock_returns) < -0.01 else "neutral"
            },
            
            "reward_breakdown": {
                "sharpe_component": float(reward_components.get('sharpe', 0)),
                "active_selection_component": float(reward_components.get('active_selection', 0)),
                "mdd_penalty_component": float(reward_components.get('mdd_penalty', 0)),
                "cash_timing_component": float(reward_components.get('cash_timing', 0)),
                "reward_approach": "active_selection_fusion"
            },
            
            "action_analysis": {
                "action_received": action.tolist() if isinstance(action, np.ndarray) else action,
                "final_weights": self.weights.tolist(),
                "weight_changes": (self.weights - self.weights_history[-2]).tolist() if len(self.weights_history) > 1 else [0.0] * len(self.weights),
                "portfolio_concentration": float(np.max(self.weights)),
                "weight_entropy": float(-np.sum(self.weights * np.log(self.weights + 1e-8))),
                "stock_concentration": float(np.sum(self.weights[:-1])),
                "individual_stock_weights": {
                    self.stock_names[i]: float(self.weights[i]) 
                    for i in range(len(self.stock_names))
                }
            },
            
            "performance_metrics": {
                "current_day": self.current_step,  # 🆕 當前第幾天
                "total_days": self.n_days,        # 🆕 總天數
                "portfolio_growth": float((self.portfolio_value / 10000.0 - 1) * 100),
                "is_outperforming": bool(portfolio_return > benchmark_return),
                "days_remaining": self.n_days - 1 - self.current_step,
                "max_drawdown_current": float(self.reward_calculator.mdd_portfolio_values[-1] - self.reward_calculator.mdd_running_peak) / self.reward_calculator.mdd_running_peak if self.reward_calculator.mdd_portfolio_values else 0.0
            }
        }
        
        return next_state, reward, done, info
    
    def calculate_equal_weight_benchmark(self):
        """計算等權重基準收益"""
        equal_stock_weight = 0.8 / self.n_stocks
        equal_weights = np.array([equal_stock_weight] * self.n_stocks + [0.2])
        benchmark_values = [10000.0]
        
        for i in range(len(self.returns)):
            stock_returns = self.returns[i]
            cash_return = self.daily_cash_return
            
            stock_return = np.dot(equal_weights[:-1], stock_returns)
            cash_return_portion = equal_weights[-1] * cash_return
            total_return = stock_return + cash_return_portion
            
            new_value = benchmark_values[-1] * (1 + total_return)
            benchmark_values.append(new_value)
            
        return benchmark_values

def create_td3_learning_monitor_with_early_stopping(exp_dir, episode_data, training_losses, reward_history, test_weights_data, validation_scores, early_stopping_info):
    """🧠 TD3學習監控圖表 - 包含Early Stopping信息"""
    plt.style.use('default')
    
    # 提取關鍵數據
    episodes = [data['episode'] for data in episode_data]
    returns = [data['return'] for data in episode_data]
    portfolio_values = [data['portfolio_value'] for data in episode_data]
    cash_ratios = [data['final_cash_ratio'] for data in episode_data]
    
    # 創建3x3子圖
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))
    fig.suptitle('🧠 TD3 Learning Monitor with Early Stopping (Twin Delayed DDPG)', fontsize=16, fontweight='bold', y=0.98)
    
    # ① Episode Reward 學習趨勢 + Early Stopping 標記
    ax1 = axes[0, 0]
    ax1.plot(episodes, returns, 'lightblue', alpha=0.6, linewidth=1, label='Raw Rewards')
    
    # 移動平均線
    if len(returns) >= 10:
        window = max(5, len(returns) // 10)
        moving_avg = pd.Series(returns).rolling(window=window, center=True).mean()
        ax1.plot(episodes, moving_avg, 'red', linewidth=3, label=f'Moving Avg ({window})')
        
        # 線性趨勢線
        z = np.polyfit(episodes, returns, 1)
        trend_line = np.poly1d(z)
        ax1.plot(episodes, trend_line(episodes), 'green', linestyle='--', linewidth=2, 
                label=f'Trend (slope: {z[0]:.3f})')
        
        # 學習判斷
        learning_status = "✅ Learning!" if z[0] > 0 else "⚠️ Not Learning"
        ax1.text(0.02, 0.98, learning_status, transform=ax1.transAxes, 
                bbox=dict(boxstyle='round', facecolor='lightgreen' if z[0] > 0 else 'yellow'),
                fontsize=12, fontweight='bold', verticalalignment='top')
    
    # 🆕 標記Early Stopping點
    if early_stopping_info['early_stopped']:
        ax1.axvline(x=early_stopping_info['best_episode'], color='red', linestyle=':', 
                   linewidth=2, alpha=0.7, label=f'Best Episode ({early_stopping_info["best_episode"]})')
        ax1.axvline(x=len(episodes), color='orange', linestyle='--', 
                   linewidth=2, alpha=0.7, label='Early Stop')
    
    ax1.set_title('① Episode Reward Learning Trend + Early Stopping', fontweight='bold')
    ax1.set_xlabel('Training Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ② 🆕 驗證分數趋势
    ax2 = axes[0, 1]
    if validation_scores:
        val_episodes = [score['episode'] for score in validation_scores]
        val_scores = [score['score'] for score in validation_scores]
        
        ax2.plot(val_episodes, val_scores, 'purple', linewidth=2.5, marker='o', markersize=4, label='Validation Score')
        
        # 標記最佳分數
        best_idx = np.argmax(val_scores)
        ax2.plot(val_episodes[best_idx], val_scores[best_idx], 'red', marker='*', markersize=12, label='Best Score')
        
        # 顯示Early Stopping信息
        metric_name = 'Sharpe Ratio' if EARLY_STOPPING_CONFIG['use_sharpe_ratio'] else 'Avg Reward'
        ax2.text(0.02, 0.98, f'Best {metric_name}: {early_stopping_info["best_validation_score"]:.4f}\nBest Episode: {early_stopping_info["best_episode"]}\nEarly Stopped: {early_stopping_info["early_stopped"]}', 
                transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgreen' if early_stopping_info["early_stopped"] else 'lightblue'),
                fontsize=10, verticalalignment='top')
        
        ax2.set_title('② Validation Score Trend', fontweight='bold')
        ax2.set_xlabel('Training Episode')
        ax2.set_ylabel(metric_name)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # ③ Cash Allocation 智能程度（測試期間的天數）
    ax3 = axes[1, 0]
    if len(test_weights_data) > 0:
        test_days = list(range(1, len(test_weights_data) + 1))
        cash_pct_test = [w[-1] * 100 for w in test_weights_data]
        ax3.plot(test_days, cash_pct_test, 'orange', linewidth=2, label='Cash % (Testing)')
        ax3.axhline(y=20, color='gray', linestyle=':', alpha=0.7, label='Benchmark (20%)')
        
        # 計算現金配置的智能度
        if len(cash_pct_test) > 10:
            cash_std = np.std(cash_pct_test)
            cash_range = max(cash_pct_test) - min(cash_pct_test)
            adaptability = "🤖 Adaptive" if cash_std > 5 else "😴 Static"
            
            ax3.text(0.02, 0.98, f'{adaptability}\nStd: {cash_std:.1f}%\nRange: {cash_range:.1f}%', 
                    transform=ax3.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightblue'),
                    fontsize=10, verticalalignment='top')
        
        ax3.set_title('③ Cash Allocation Intelligence (Testing)', fontweight='bold')
        ax3.set_xlabel('Trading Day')
        ax3.set_ylabel('Cash Allocation (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # ④ 訓練收斂狀況 (TD3 specific)
    ax4 = axes[1, 1]
    if training_losses:
        update_episodes = list(range(1, len(training_losses) + 1))
        actor_losses = [loss['actor_loss'] for loss in training_losses]
        critic1_losses = [loss.get('critic1_loss', 0) for loss in training_losses]
        critic2_losses = [loss.get('critic2_loss', 0) for loss in training_losses]
        
        ax4.plot(update_episodes, actor_losses, 'red', linewidth=2, label='Actor Loss', marker='o', markersize=2)
        ax4.plot(update_episodes, critic1_losses, 'blue', linewidth=2, label='Critic1 Loss', marker='s', markersize=2)
        ax4.plot(update_episodes, critic2_losses, 'purple', linewidth=2, label='Critic2 Loss', marker='^', markersize=2)
        
        # 收斂判斷
        if len(actor_losses) >= 5:
            recent_actor_std = np.std(actor_losses[-5:])
            recent_c1_std = np.std(critic1_losses[-5:])
            convergence_status = "✅ Converging" if (recent_actor_std < 0.1 and recent_c1_std < 0.1) else "⚠️ Unstable"
            ax4.text(0.02, 0.98, f'{convergence_status}\nActor Std: {recent_actor_std:.3f}\nCritic1 Std: {recent_c1_std:.3f}',
                    transform=ax4.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgreen' if recent_actor_std < 0.1 and recent_c1_std < 0.1 else 'yellow'),
                    fontsize=10, verticalalignment='top')
    
    ax4.set_title('④ TD3 Training Convergence', fontweight='bold')
    ax4.set_xlabel('Training Update Step')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # ⑤ Q值健康度 (TD3 specific)
    ax5 = axes[2, 0]
    if training_losses:
        q1_values = [loss.get('avg_q1_value', 0) for loss in training_losses]
        q2_values = [loss.get('avg_q2_value', 0) for loss in training_losses]
        
        ax5.plot(update_episodes, q1_values, 'blue', linewidth=2, marker='o', markersize=2, label='Avg Q1-Value')
        ax5.plot(update_episodes, q2_values, 'purple', linewidth=2, marker='s', markersize=2, label='Avg Q2-Value')
        
        # Q值健康度評估
        avg_q1 = np.mean(q1_values) if q1_values else 0
        avg_q2 = np.mean(q2_values) if q2_values else 0
        q1_std = np.std(q1_values) if q1_values else 0
        q2_std = np.std(q2_values) if q2_values else 0
        
        # 整體健康度判斷
        q1_health = abs(avg_q1) < 10 and q1_std < 5
        q2_health = abs(avg_q2) < 10 and q2_std < 5
        overall_health = "💚 Healthy" if (q1_health and q2_health) else "💛 Caution" if (q1_health or q2_health) else "❤️ Dangerous"
        
        ax5.text(0.02, 0.98, f'{overall_health}\nAvg Q1: {avg_q1:.2f} (±{q1_std:.2f})\nAvg Q2: {avg_q2:.2f} (±{q2_std:.2f})',
                transform=ax5.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgreen' if (q1_health and q2_health) else 'yellow' if (q1_health or q2_health) else 'pink'),
                fontsize=10, verticalalignment='top')
    
    ax5.set_title('⑤ Twin Q-Value Health (TD3)', fontweight='bold')
    ax5.set_xlabel('Training Update Step')
    ax5.set_ylabel('Average Q-Value')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # ⑥ 權重穩定性分析（測試期間，按天數）
    ax6 = axes[2, 1]
    if len(test_weights_data) > 10:
        test_days = list(range(1, len(test_weights_data) + 1))
        
        # 計算每日權重變化
        weight_changes = []
        for i in range(1, len(test_weights_data)):
            change = np.sum(np.abs(np.array(test_weights_data[i]) - np.array(test_weights_data[i-1])))
            weight_changes.append(change)
        
        change_days = list(range(2, len(test_weights_data) + 1))
        ax6.plot(change_days, weight_changes, 'purple', linewidth=2, alpha=0.8)
        
        # 穩定性評估
        avg_change = np.mean(weight_changes)
        stability_status = "📈 Stable" if avg_change < 0.1 else "⚡ Dynamic"
        ax6.text(0.02, 0.98, f'{stability_status}\nAvg Change: {avg_change:.3f}',
                transform=ax6.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgreen' if avg_change < 0.1 else 'yellow'),
                fontsize=10, verticalalignment='top')
        
        ax6.set_title('⑥ Weight Stability (Testing Days)', fontweight='bold')
        ax6.set_xlabel('Trading Day')
        ax6.set_ylabel('Daily Weight Change')
        ax6.grid(True, alpha=0.3)
    
    # ⑦ 融合版獎勵成分分析（測試期間）
    ax7 = axes[0, 2]
    if reward_history and len(reward_history) > 0:
        try:
            # ==================================
            #  BUG FIX: 對龐大的 reward_history 數據進行降採樣/平滑化
            # ==================================
            reward_df = pd.DataFrame([r['components'] for r in reward_history])
            
            # 計算移動平均，窗口大小設為總步數的 1% 或至少為 100
            window_size = max(100, len(reward_df) // 100)
            
            # 使用 rolling().mean() 進行平滑化
            sharpe_rewards_smooth = reward_df['sharpe'].rolling(window=window_size, min_periods=1).mean()
            active_selection_rewards_smooth = reward_df['active_selection'].rolling(window=window_size, min_periods=1).mean()
            mdd_penalties_smooth = reward_df['mdd_penalty'].rolling(window=window_size, min_periods=1).mean()
            cash_timing_rewards_smooth = reward_df['cash_timing'].rolling(window=window_size, min_periods=1).mean()
            
            # 使用原始的 global steps 作為 x 軸
            reward_steps = range(len(reward_df))

            ax7.plot(reward_steps, sharpe_rewards_smooth, 'g-', linewidth=2, alpha=0.8, label=f'Sharpe (MA {window_size})')
            ax7.plot(reward_steps, active_selection_rewards_smooth, 'b-', linewidth=2, alpha=0.8, label=f'Active Selection (MA {window_size})')
            ax7.plot(reward_steps, mdd_penalties_smooth, 'r-', linewidth=2, alpha=0.8, label=f'MDD Penalty (MA {window_size})')
            ax7.plot(reward_steps, cash_timing_rewards_smooth, 'orange', linewidth=2, alpha=0.8, label=f'Cash Timing (MA {window_size})')
            
            # 將標題修正為更能反映數據來源
            ax7.set_title('⑦ Smoothed Reward Components (Training)', fontweight='bold')
            ax7.set_xlabel('Global Training Step') # x 軸現在是全局訓練步數
            ax7.set_ylabel('Smoothed Reward Component')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
                
        except Exception as e:
            ax7.text(0.5, 0.5, f'Reward analysis error:\n{str(e)[:50]}...', 
                    transform=ax7.transAxes, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='pink'))
            ax7.set_title('⑦ Fusion Reward Components (Error)', fontweight='bold')
    
    # ⑧ 股票 vs 現金比例動態（測試期間）
    ax8 = axes[1, 2]
    if len(test_weights_data) > 0:
        test_days = list(range(1, len(test_weights_data) + 1))
        stock_weights = [sum(w[:-1]) for w in test_weights_data]
        cash_weights = [w[-1] for w in test_weights_data]
        
        ax8.plot(test_days, stock_weights, label='Total Stocks', color='#C41E3A', linewidth=2.5, alpha=0.9)
        ax8.plot(test_days, cash_weights, label='Cash', color='#B8860B', linewidth=2.5, linestyle='--', alpha=0.9)
        
        ax8.axhline(y=0.8, color='#191970', linestyle=':', alpha=0.8, linewidth=2, label='Benchmark Stocks (80%)')
        ax8.axhline(y=0.2, color='#FF8C00', linestyle=':', alpha=0.8, linewidth=2, label='Benchmark Cash (20%)')
        
        ax8.set_title('⑧ Stocks vs Cash Dynamics (Testing)', fontweight='bold')
        ax8.set_xlabel('Trading Day')
        ax8.set_ylabel('Weight')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    
    # ⑨ 🆕 TD3 + Early Stopping架構說明
    ax9 = axes[2, 2]
    ax9.text(0.1, 0.95, '🤖 TD3 + Early Stopping Architecture', fontsize=14, fontweight='bold', transform=ax9.transAxes, color='red')
    ax9.text(0.1, 0.85, '✅ Actor Network: π(a|s) - Deterministic', fontsize=11, transform=ax9.transAxes)
    ax9.text(0.1, 0.78, '✅ Target Actor: π̄(a|s)', fontsize=11, transform=ax9.transAxes, color='blue')
    ax9.text(0.1, 0.71, '✅ Twin Critics: Q₁(s,a) & Q₂(s,a)', fontsize=11, transform=ax9.transAxes, color='purple')
    ax9.text(0.1, 0.64, '✅ Target Critics: Q̄₁(s,a) & Q̄₂(s,a)', fontsize=11, transform=ax9.transAxes, color='purple')
    ax9.text(0.1, 0.57, '❌ No V-Critic Networks', fontsize=11, transform=ax9.transAxes, color='red')
    ax9.text(0.1, 0.50, '❌ No Entropy Term', fontsize=11, transform=ax9.transAxes, color='red')
    ax9.text(0.1, 0.43, '✅ Policy Delay: Update every 2 steps', fontsize=11, transform=ax9.transAxes)
    ax9.text(0.1, 0.36, '✅ Target Smoothing: Noise clipping', fontsize=11, transform=ax9.transAxes)
    ax9.text(0.1, 0.29, '🔄 Fusion Reward: 4 components', fontsize=11, transform=ax9.transAxes, color='green')
    ax9.text(0.1, 0.22, '📊 Rich State: 5 features/stock', fontsize=11, transform=ax9.transAxes, color='green')
    ax9.text(0.1, 0.15, '⭐ Deterministic policy with exploration', fontsize=11, transform=ax9.transAxes, color='blue')
    ax9.text(0.1, 0.08, f'🔄 Early Stopping: {early_stopping_info["early_stopping_config"]["patience"]} patience', fontsize=11, transform=ax9.transAxes, color='orange')
    ax9.text(0.1, 0.01, f'💾 Best Model: Episode {early_stopping_info["best_episode"]}', fontsize=11, transform=ax9.transAxes, color='green')
    ax9.set_title('⑨ TD3 Twin Delayed DDPG + Early Stopping', fontweight='bold')
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{exp_dir}/plots/enhanced_td3_learning_monitor_early_stopping.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Enhanced TD3 Learning Monitor with Early Stopping created!")

def run_td3_fusion_experiment_with_early_stopping(csv_file_path=None):
    """運行TD3融合實驗 - Twin Delayed DDPG，集成 Early Stopping 機制"""
    
    print("🎯 TD3 Portfolio Management with Early Stopping - Twin Delayed Deep Deterministic Policy Gradient 🎯")
    print("=" * 80)
    print("📈 Rich State Features (5 per stock) + Fusion Reward Function")
    print("🎯 State: Momentum(5d,20d) + Volatility + RSI + Bollinger%B + Current Weights")
    print("🎯 Reward: Sharpe + ActiveSelection + MaxDrawdown + CashTiming")
    print("🔄 Algorithm: TD3 (Twin Delayed DDPG)")
    print("🏗️ Networks: Actor + Target Actor + Twin Critics + Target Critics")
    print("📊 Charts: X-axis using Trading Days (1, 2, 3, ..., N)")
    print("✅ Deterministic Policy with Exploration Noise")
    print("✅ Policy Delay + Target Smoothing + Twin Critics")
    print("❌ No V-Networks (unlike SAC)")
    print("❌ No Entropy Term (deterministic policy)")
    print("🔄 🆕 Early Stopping: 驗證集評估自動停止訓練")
    print("📊 🆕 Best Model Selection: 保存並載入最佳模型")
    print("=" * 80)
    
    setup_gpu()
    
    # 設置實驗環境
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"td3_fusion_early_stop_{timestamp}"
    
    dirs = [exp_dir, f"{exp_dir}/data", f"{exp_dir}/models", f"{exp_dir}/plots"]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # 融合版配置
    config = EnhancedFinancialConfig()
    
    # 默認路徑
    if csv_file_path is None:
        csv_file_path = "eight_stock_prices.csv"
    
    if not os.path.exists(csv_file_path):
        print(f"✗ File not found: {csv_file_path}")
        return
    
    print("\n=== Creating Enhanced Environments with Early Stopping ===")
    start_time = time.time()
    
    try:
        train_env = EnhancedPortfolioEnvironment(csv_file_path, mode='train', config=config)
        val_env = EnhancedPortfolioEnvironment(csv_file_path, mode='val', config=config)
        test_env = EnhancedPortfolioEnvironment(csv_file_path, mode='test', config=config)
        
        print(f"✓ Enhanced environments created in {time.time() - start_time:.1f}s")
        print(f"✓ Detected stocks: {train_env.stock_names}")
        print(f"✓ Enhanced state dimension: {train_env.state_dim}")
        print(f"✓ Testing days: {test_env.n_days}")
        
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return
    
    print(f"\n=== Creating TD3 Agent with Early Stopping Support ===")
    print(f"🔧 Setting up TD3 (Twin Delayed DDPG) agent...")
    
    # 創建TD3 Agent
    agent = TD3Agent(
        state_dim=train_env.state_dim,
        action_dim=train_env.action_dim,
        learning_rate=1e-4
    )
    
    # 🔧 驗證TD3架構
    print(f"✅ Actor network: {hasattr(agent, 'actor')}")
    print(f"✅ Target Actor network: {hasattr(agent, 'actor_target')}")
    print(f"✅ Critic1 network: {hasattr(agent, 'critic1')}")
    print(f"✅ Critic2 network: {hasattr(agent, 'critic2')}")
    print(f"✅ Target Critic1 network: {hasattr(agent, 'critic1_target')}")
    print(f"✅ Target Critic2 network: {hasattr(agent, 'critic2_target')}")
    print(f"❌ No V-Critic (correct): {not hasattr(agent, 'v_critic')}")
    print(f"❌ No Alpha (correct): {not hasattr(agent, 'log_alpha')}")
    print(f"🔄 Early Stopping methods: {hasattr(agent, 'save_best_models')} & {hasattr(agent, 'load_best_models')}")
    
    print(f"\n=== TD3 Training with Early Stopping ===")
    print(f"State features: {train_env.n_stocks} stocks × 5 features + {train_env.n_assets} weights + 1 cash = {train_env.state_dim}")
    print(f"Network architecture: Actor + Target Actor + Twin Critics + Target Critics")
    print(f"Reward focus: Sharpe({config.sharpe_weight:.0%}) + ActiveSelection({config.return_weight:.0%}) + MDD({config.mdd_weight:.0%}) + Cash({config.cash_timing_weight:.0%})")
    print(f"Algorithm: TD3 (Deterministic policy with exploration noise)")
    print(f"Key features: Policy Delay + Target Smoothing + Twin Critics")
    
    # 🆕 Early Stopping 設置
    print(f"\n🔄 Early Stopping Configuration:")
    print(f"  📊 Evaluation Interval: {EARLY_STOPPING_CONFIG['eval_interval']} episodes")
    print(f"  ⏳ Patience: {EARLY_STOPPING_CONFIG['patience']} evaluations")
    print(f"  📏 Minimum Episodes: {EARLY_STOPPING_CONFIG['min_episodes']}")
    print(f"  📈 Metric: {'Sharpe Ratio' if EARLY_STOPPING_CONFIG['use_sharpe_ratio'] else 'Average Reward'}")
    print(f"  💾 Save Best Models: {EARLY_STOPPING_CONFIG['save_best_models']}")
    
    # 訓練參數
    max_episodes = 200  # 最大 episode 數，但可能提前停止
    max_steps_per_episode = len(train_env.features)
    start_training_step = 1000
    training_frequency = 10
    
    # 🆕 Early Stopping 變數
    best_val_score = -np.inf
    best_episode = -1
    patience_counter = 0
    eval_interval = EARLY_STOPPING_CONFIG['eval_interval']
    patience = EARLY_STOPPING_CONFIG['patience']
    min_episodes = EARLY_STOPPING_CONFIG['min_episodes']
    use_sharpe_ratio = EARLY_STOPPING_CONFIG['use_sharpe_ratio']
    save_best_models = EARLY_STOPPING_CONFIG['save_best_models']
    
    # 數據收集
    episode_data = []
    weights_data = []
    training_losses = []
    episode_reward_history = []
    feedback_data = []
    validation_scores = []  # 🆕 記錄驗證分數
    
    global_step = 0
    
    print(f"\n=== Enhanced TD3 Training Loop with Early Stopping ===")
    training_start = time.time()
    
    try:
        for episode in range(max_episodes):
            state = train_env.reset()
            episode_return = 0
            episode_steps = 0
            
            for step in range(max_steps_per_episode):
                # 獲取action（帶探索噪聲）
                action, action_info = agent.get_action(state, add_noise=True)
                
                # 環境step
                next_state, reward, done, info = train_env.step(action)
                
                # 收集反饋數據
                feedback_entry = {
                    "episode": episode,
                    "step": step,
                    "global_step": global_step,
                    "action_info": action_info,
                    "env_info": info,
                    "reward": float(reward),
                    "enhanced_features": True,
                    "reward_approach": "active_selection_fusion",
                    "algorithm": "TD3",
                    "early_stopping": True,  # 🆕 標記使用了 Early Stopping
                }
                feedback_data.append(feedback_entry)
                
                # 添加經驗到回放緩衝區
                agent.add_experience(state, action, reward, next_state, done)
                
                # TD3訓練
                if global_step >= start_training_step and global_step % training_frequency == 0:
                    train_info = agent.train()
                    if train_info:
                        training_losses.append(train_info)
                
                state = next_state
                episode_return += reward
                episode_steps += 1
                global_step += 1
                
                if done:
                    break
            
            # 記錄episode數據
            episode_info = {
                'episode': episode,
                'return': episode_return,
                'portfolio_value': train_env.portfolio_value,
                'final_weights': train_env.weights.copy(),
                'final_cash_ratio': train_env.weights[-1],
                'steps': episode_steps,
                'global_step': global_step
            }
            episode_data.append(episode_info)
            episode_reward_history.extend(train_env.reward_history)
            
            # 🆕 Early Stopping 評估
            if (episode + 1) % eval_interval == 0 and episode >= min_episodes:
                print(f"\n🔍 Validation Evaluation at Episode {episode + 1}")
                
                # 在驗證集上評估
                val_score = evaluate_on_validation(agent, val_env)
                validation_scores.append({
                    'episode': episode + 1,
                    'score': val_score,
                    'metric': 'sharpe_ratio' if use_sharpe_ratio else 'avg_reward'
                })
                
                print(f"  📊 Validation {'Sharpe Ratio' if use_sharpe_ratio else 'Average Reward'}: {val_score:.4f}")
                print(f"  🏆 Best Score So Far: {best_val_score:.4f} (Episode {best_episode})")
                
                # 檢查是否為最佳分數
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_episode = episode + 1
                    patience_counter = 0
                    
                    # 🆕 保存最佳模型
                    if save_best_models:
                        agent.save_best_models(f"{exp_dir}/models")
                    
                    print(f"  ✅ NEW BEST! Score: {val_score:.4f} at Episode {episode + 1}")
                    print(f"  💾 Best models saved to {exp_dir}/models/")
                else:
                    patience_counter += 1
                    print(f"  ⏳ No improvement. Patience: {patience_counter}/{patience}")
                    
                    # 🔄 Early Stopping 條件檢查
                    if patience_counter >= patience:
                        print(f"\n🛑 EARLY STOPPING TRIGGERED!")
                        print(f"  📊 Stopped at Episode: {episode + 1}")
                        print(f"  🏆 Best Score: {best_val_score:.4f}")
                        print(f"  🎯 Best Episode: {best_episode}")
                        print(f"  ⏳ Patience Exceeded: {patience_counter}/{patience}")
                        break
            
            # 進度報告
            if episode % 10 == 0:
                elapsed = time.time() - training_start
                current_weights = train_env.weights
                buffer_size = len(agent.replay_buffer)
                
                print(f"\nEpisode {episode:3d}: Portfolio ${train_env.portfolio_value:8.0f}, "
                      f"Return {episode_return:6.2f}, Time {elapsed:.0f}s")
                print(f"  TD3 State Dim: {train_env.state_dim}, Buffer: {buffer_size}, Training Step: {agent.training_step}")
                
                # 顯示股票權重分布
                print(f"  📊 Stock Weights:")
                for i, (stock, weight) in enumerate(zip(train_env.stock_names[:4], current_weights[:4])):
                    print(f"    {stock}: {weight:.1%}", end="  ")
                if len(train_env.stock_names) > 4:
                    print(f"\n    ... and {len(train_env.stock_names) - 4} more stocks")
                print(f"  💰 Cash: {current_weights[-1]:.1%}")
                
                # 🆕 顯示 Early Stopping 狀態
                if episode >= min_episodes:
                    print(f"  🔄 Early Stopping Status:")
                    print(f"    📊 Best Score: {best_val_score:.4f} (Episode {best_episode})")
                    print(f"    ⏳ Patience: {patience_counter}/{patience}")
                    next_eval = ((episode // eval_interval) + 1) * eval_interval
                    print(f"    📅 Next Evaluation: Episode {next_eval}")
                
                # 展示融合版獎勵成分
                if len(episode_reward_history) > 0:
                    recent_rewards = episode_reward_history[-20:]
                    if recent_rewards and 'components' in recent_rewards[0]:
                        avg_components = {}
                        for component in recent_rewards[0]['components'].keys():
                            component_values = [r['components'][component] for r in recent_rewards 
                                              if not np.isnan(r['components'][component])]
                            if component_values:
                                avg_components[component] = np.mean(component_values)
                        
                        print("🎯 TD3 Fusion Reward Components:")
                        for component, value in avg_components.items():
                            emoji = "📈" if component == "sharpe" else "🎯" if component == "active_selection" else "⬇️" if component == "mdd_penalty" else "⏰"
                            if not np.isnan(value):
                                print(f"  {emoji} {component}: {value:.4f}")
                        print(f"  🏗️ Architecture: TD3 (Twin Delayed DDPG)")
                        print(f"  🔧 Deterministic Policy + Exploration Noise")
    
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 🆕 載入最佳模型進行測試
    print(f"\n=== Loading Best Model for Testing ===")
    if save_best_models and best_episode > 0:
        print(f"🔄 Loading best model from Episode {best_episode} (Score: {best_val_score:.4f})")
        agent.load_best_models(f"{exp_dir}/models")
    else:
        print(f"⚠️ Using current model for testing (no best model saved)")
    
    # 測試階段
    print(f"\n=== Testing Enhanced TD3 Model with Early Stopping ===")
    test_state = test_env.reset()
    test_done = False
    test_steps = 0
    test_feedback_data = []
    
    while not test_done:
        test_action, test_action_info = agent.get_action(test_state, add_noise=False)  # 測試時不加噪聲
        test_next_state, test_reward, test_done, test_info = test_env.step(test_action)
        
        # 收集測試反饋數據
        test_feedback_entry = {
            "step": test_steps,
            "trading_day": test_steps + 1,  # 🆕 添加交易日信息
            "action_info": test_action_info,
            "env_info": test_info,
            "reward": float(test_reward),
            "phase": "testing",
            "enhanced_features": True,
            "reward_approach": "active_selection_fusion",
            "algorithm": "TD3",
            "early_stopping_used": True,  # 🆕 標記使用了 Early Stopping
            "best_episode_used": best_episode,
            "best_score": best_val_score,
        }
        test_feedback_data.append(test_feedback_entry)
        
        test_state = test_next_state
        test_steps += 1
    
    # 保存測試期間的權重數據（用於圖表）
    test_weights_data = test_env.weights_history
    
    # 保存增強版結果
    print(f"\n=== Saving Enhanced TD3 Results with Early Stopping ===")
    
    # 1. 保存episode數據
    episode_df = pd.DataFrame(episode_data)
    episode_df.to_csv(f'{exp_dir}/data/enhanced_td3_episode_data_early_stop.csv', index=False)
    
    # 🆕 2. 保存驗證分數歷史
    if validation_scores:
        validation_df = pd.DataFrame(validation_scores)
        validation_df.to_csv(f'{exp_dir}/data/validation_scores_history.csv', index=False)
    
    # 🆕 3. 保存 Early Stopping 摘要
    early_stopping_summary = {
        'early_stopping_config': EARLY_STOPPING_CONFIG,
        'best_validation_score': float(best_val_score),
        'best_episode': int(best_episode),
        'total_episodes_trained': len(episode_data),
        'patience_counter_final': int(patience_counter),
        'early_stopped': bool(patience_counter >= patience),
        'metric_used': 'sharpe_ratio' if use_sharpe_ratio else 'avg_reward',
        'models_saved': save_best_models,
        'training_time_saved': f"Potentially saved {max_episodes - len(episode_data)} episodes"
    }
    
    with open(f'{exp_dir}/data/early_stopping_summary.json', 'w') as f:
        json.dump(early_stopping_summary, f, indent=2, ensure_ascii=False)
    
    # 4. 保存測試權重數據
    assets = train_env.stock_names + ['CASH']
    test_weights_df = pd.DataFrame(test_weights_data, columns=assets)
    test_weights_df['trading_day'] = range(1, len(test_weights_data) + 1)
    test_weights_df.to_csv(f'{exp_dir}/data/test_weights_by_days_early_stop.csv', index=False)
    
    if training_losses:
        losses_df = pd.DataFrame(training_losses)
        losses_df.to_csv(f'{exp_dir}/data/enhanced_td3_training_losses_early_stop.csv', index=False)
    
    # 5. 保存獎勵歷史
    reward_df = pd.DataFrame([
        {
            'step': r['step'],
            'total_reward': r['total_reward'],
            **r['components']
        } for r in episode_reward_history
    ])
    reward_df.to_csv(f'{exp_dir}/data/enhanced_td3_detailed_rewards_early_stop.csv', index=False)
    
    # 6. 保存反饋數據
    if feedback_data:
        feedback_df = pd.DataFrame(feedback_data)
        feedback_df.to_csv(f'{exp_dir}/data/enhanced_td3_feedback_loop_data_early_stop.csv', index=False)
    
    if test_feedback_data:
        test_feedback_df = pd.DataFrame(test_feedback_data)
        test_feedback_df.to_csv(f'{exp_dir}/data/enhanced_td3_test_feedback_by_days_early_stop.csv', index=False)
    
    # 🆕 7. 保存TD3模型 (最終模型，除了最佳模型)
    agent.actor.save(f'{exp_dir}/models/td3_actor_final.keras')
    agent.actor_target.save(f'{exp_dir}/models/td3_actor_target_final.keras')
    agent.critic1.save(f'{exp_dir}/models/td3_critic1_final.keras')
    agent.critic2.save(f'{exp_dir}/models/td3_critic2_final.keras')
    agent.critic1_target.save(f'{exp_dir}/models/td3_critic1_target_final.keras')
    agent.critic2_target.save(f'{exp_dir}/models/td3_critic2_target_final.keras')
    
    # 🆕 8. 創建TD3學習監控圖表 (包含Early Stopping信息)
    print("🧠 Creating enhanced TD3 learning monitor with Early Stopping info...")
    create_td3_learning_monitor_with_early_stopping(exp_dir, episode_data, training_losses, episode_reward_history, test_weights_data, validation_scores, early_stopping_summary)
    
    # 🆕 9. 創建以天數為x軸的可視化圖表
    print("📊 Creating enhanced TD3 visualizations with trading days as x-axis...")
    create_enhanced_visualizations_with_days(exp_dir, episode_data, test_weights_data, training_losses, train_env.stock_names, test_steps)
    
    # 🆕 10. 創建以天數為x軸的詳細回測對比
    print("📊 Creating detailed TD3 backtest comparison by trading days...")
    backtest_results = create_detailed_backtest_comparison_by_days(exp_dir, test_env, train_env.stock_names)
    
    # 🆕 11. 創建專業風格堆疊圖
    # 我們需要從測試環境中獲取真實的日期索引
    test_dates = test_env.price_subset.index
    create_professional_stacked_chart_final(exp_dir, test_weights_data, train_env.stock_names, test_dates)
    
    # 12. 生成系統輸出
    final_value = test_env.portfolio_value
    total_return = (final_value / 10000.0) - 1
    final_cash_ratio = test_env.weights[-1]
    
    portfolio_values = test_env.portfolio_values_history
    peak_values = np.maximum.accumulate(portfolio_values)
    drawdowns = (np.array(portfolio_values) - peak_values) / peak_values
    max_drawdown = np.min(drawdowns)
    
    # 完整系統輸出
    system_output = {
        "🎯 quant_model_layer": config.output_objectives(),
        "🤖 rl_agent_layer": {
            "algorithm": "TD3 (Twin Delayed DDPG)",
            "state_dim": agent.state_dim,
            "action_dim": agent.action_dim,
            "enhanced_features": True,
            "features_per_stock": 5,
            "fusion_approach": True,
            "algorithm_type": "deterministic_policy_gradient",
            "exploration_method": "gaussian_noise",
            "early_stopping_enabled": True,  # 🆕
            "state_composition": {
                "stock_features": train_env.n_stocks * 5,
                "cash_feature": 1,
                "weight_features": train_env.n_assets,
                "total": train_env.state_dim
            },
            "network_architecture": {
                "actor": "π(a|s) - Deterministic policy network",
                "actor_target": "π̄(a|s) - Target policy network",
                "critic1": "Q₁(s,a) - Twin critic 1",
                "critic2": "Q₂(s,a) - Twin critic 2", 
                "critic1_target": "Q̄₁(s,a) - Target critic 1",
                "critic2_target": "Q̄₂(s,a) - Target critic 2",
                "enhanced_capacity": [512, 256, 128],
                "total_parameters": (agent.actor.count_params() + 
                                   agent.actor_target.count_params() + 
                                   agent.critic1.count_params() + 
                                   agent.critic2.count_params() +
                                   agent.critic1_target.count_params() +
                                   agent.critic2_target.count_params())
            },
            "td3_hyperparameters": {
                "policy_delay": agent.policy_delay,
                "policy_noise": agent.policy_noise,
                "noise_clip": agent.noise_clip,
                "exploration_noise": agent.exploration_noise,
                "tau": agent.tau
            }
        },
        "🏢 trading_environment_layer": {
            "enhanced_features": True,
            "features_per_stock": ["momentum_5d", "momentum_20d", "volatility_20d", "rsi", "bollinger_b"],
            "n_stocks": train_env.n_stocks,
            "state_dim": train_env.state_dim,
            "stock_names": train_env.stock_names,
            "testing_days": test_steps
        },
        "🔄 early_stopping_layer": early_stopping_summary,  # 🆕
        "📊 visualization_enhancements": {
            "x_axis_type": "trading_days",
            "charts_use_days": True,
            "day_range": f"1 to {test_steps}",
            "training_episodes": len(episode_data),
            "early_stopped": bool(patience_counter >= patience)
        },
        "📈 performance_summary": {
            "final_portfolio_value": float(final_value),
            "total_return_pct": float(total_return * 100),
            "max_drawdown_pct": float(max_drawdown * 100),
            "final_cash_ratio": float(final_cash_ratio),
            "testing_days": test_steps,
            "sharpe_ratio": backtest_results['td3_sharpe'],
            "win_rate_pct": backtest_results['win_rate'],
            "enhanced_features_used": True,
            "reward_approach": "active_selection_fusion",
            "algorithm": "TD3",
            "early_stopping_used": True,
            "best_validation_score": float(best_val_score),
            "best_episode": int(best_episode),
            "total_training_episodes": len(episode_data)
        }
    }
    
    # 轉換numpy類型並保存
    system_output = convert_numpy_types(system_output)
    
    with open(f'{exp_dir}/td3_fusion_early_stop_system_output.json', 'w') as f:
        json.dump(system_output, f, indent=2, ensure_ascii=False)
    
    print(f"✅ TD3 Early Stopping experiment completed!")
    print(f"📊 Final Portfolio Value: ${final_value:.2f}")
    print(f"📈 Total Return: {total_return:.1%}")
    print(f"📉 Max Drawdown: {max_drawdown:.1%}")
    print(f"💰 Final Cash Ratio: {final_cash_ratio:.1%}")
    print(f"🎯 Reward Approach: Active Selection Fusion")
    print(f"🎯 Enhanced State Dim: {train_env.state_dim}")
    print(f"📊 Testing Days: {test_steps}")
    print(f"✨ Rich Features: 5 per stock (momentum, volatility, RSI, Bollinger)")
    print(f"📊 Sharpe Ratio: {backtest_results['td3_sharpe']:.2f}")
    print(f"📊 Win Rate: {backtest_results['win_rate']:.1f}%")
    print(f"🏗️ TD3 Architecture: Actor + Target Actor + Twin Critics + Target Critics")
    print(f"🎯 Deterministic Policy with Exploration Noise")
    print(f"✅ Policy Delay: {agent.policy_delay}, Target Smoothing, Twin Critics")
    
    # 🆕 Early Stopping 結果摘要
    print(f"\n🔄 ⭐ EARLY STOPPING SUMMARY ⭐")
    print(f"🏆 Best Validation Score: {best_val_score:.4f}")
    print(f"🎯 Best Episode: {best_episode}")
    print(f"📊 Total Episodes Trained: {len(episode_data)} / {max_episodes}")
    print(f"🔄 Early Stopped: {'Yes' if patience_counter >= patience else 'No'}")
    print(f"⏳ Final Patience Counter: {patience_counter} / {patience}")
    print(f"📏 Metric Used: {'Sharpe Ratio' if use_sharpe_ratio else 'Average Reward'}")
    print(f"💾 Best Models Saved: {'Yes' if save_best_models else 'No'}")
    if patience_counter >= patience:
        episodes_saved = max_episodes - len(episode_data)
        print(f"⚡ Training Efficiency: Saved {episodes_saved} episodes ({episodes_saved/max_episodes*100:.1f}%)")
    
    print(f"\n🔄 Results saved to: {exp_dir}")
    
    print(f"\n📊 ⭐ CHARTS WITH TRADING DAYS AS X-AXIS + EARLY STOPPING ⭐")
    print(f"🧠 enhanced_td3_learning_monitor_early_stopping.png - TD3學習監控 + Early Stopping")
    print(f"📈 enhanced_td3_weight_evolution_by_days.png")
    print(f"📊 enhanced_td3_weight_stacked_by_days.png")
    print(f"💰 enhanced_td3_stocks_vs_cash_by_days.png")
    print(f"📋 enhanced_td3_comprehensive_metrics.png")
    print(f"🏆 enhanced_td3_backtest_comparison_by_days.png")
    print(f"🎨 professional_stacked_chart.png")
    
    print(f"\n🎯 ⭐ TD3架構說明 ⭐")
    print(f"✅ Actor Network: π(a|s) - 確定性策略網絡")
    print(f"✅ Target Actor: π̄(a|s) - 目標策略網絡")
    print(f"✅ Twin Critics: Q₁(s,a) & Q₂(s,a) - 雙重價值函數")
    print(f"✅ Target Critics: Q̄₁(s,a) & Q̄₂(s,a) - 目標價值函數")
    print(f"❌ No V-Critic Networks - 與SAC不同")
    print(f"❌ No Entropy Term - 確定性策略")
    print(f"✅ Policy Delay: 每{agent.policy_delay}步更新策略")
    print(f"✅ Target Smoothing: 目標策略平滑技術")
    print(f"✅ Exploration Noise: {agent.exploration_noise} 高斯噪聲")
    print(f"🔄 Early Stopping: 自動優化訓練效率")
    
    print(f"\n💾 ⭐ TD3模型文件 ⭐")
    print(f"🤖 best_actor.keras - 最佳Actor網絡")
    print(f"🤖 best_critic1.keras - 最佳Critic1網絡")
    print(f"🤖 best_critic2.keras - 最佳Critic2網絡")
    print(f"🔧 td3_*_final.keras - 其他最終網絡")
    
    print(f"\n🔄 ⭐ TD3 + Early Stopping 特性 ⭐")
    print(f"✅ 確定性策略: tanh輸出，無隨機性")
    print(f"✅ 探索噪聲: 訓練時加高斯噪聲")
    print(f"✅ 延遲策略更新: 減少Actor過度更新")
    print(f"✅ 目標策略平滑: 減少目標Q值過估計")
    print(f"✅ 雙重評價網絡: Twin Critics取最小值")
    print(f"✅ 軟更新所有目標網絡: τ={agent.tau}")
    print(f"🔄 自動停止訓練: 驗證集指標導向")
    print(f"💾 最佳模型保存: 確保最優性能")
    print(f"⚡ 訓練效率提升: 避免過度訓練")
    
    print(f"\n💾 ⭐ DATA FILES WITH EARLY STOPPING INFO ⭐")
    print(f"📈 test_weights_by_days_early_stop.csv - Weights with trading day column")
    print(f"📊 validation_scores_history.csv - Validation scores over training")
    print(f"🔄 early_stopping_summary.json - Early stopping configuration & results")
    print(f"🧪 enhanced_td3_test_feedback_by_days_early_stop.csv - Test feedback with day info")
    print(f"📊 enhanced_td3_backtest_comparison_by_days.csv - Detailed comparison")
    print(f"🔄 td3_fusion_early_stop_system_output.json - Complete system output")
    
    return exp_dir

def create_enhanced_visualizations_with_days(exp_dir, episode_data, weights_data, training_losses, stock_names, test_steps):
    """🆕 創建以天數為x軸的增強版可視化圖表"""
    plt.style.use('default')
    
    # 動態資產列表
    assets = stock_names + ['CASH']
    n_assets = len(assets)
    
    # 深色方案
    deep_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#393b79', '#5254a3',
    ]
    cash_color = '#B8860B'
    
    # 分配顏色
    colors = []
    for i, asset in enumerate(assets):
        if asset == 'CASH':
            colors.append(cash_color)
        else:
            colors.append(deep_colors[i % len(deep_colors)])
    
    # 🆕 1. 測試期間權重演化（以天數為x軸）
    plt.figure(figsize=(16, 10))
    
    # 使用測試期間的天數作為x軸
    days = list(range(1, len(weights_data) + 1))
    
    for i, asset in enumerate(assets):
        weights = [w[i] for w in weights_data]
        if asset == 'CASH':
            plt.plot(days, weights, label=asset, color=cash_color, 
                    linewidth=3.5, linestyle='--', alpha=0.9)
        else:
            plt.plot(days, weights, label=asset, color=colors[i], 
                    linewidth=2.5, linestyle='-', alpha=0.85)
    
    plt.axhline(y=1/n_assets, color='gray', linestyle=':', alpha=0.5, 
                label=f'Equal Weight ({100/n_assets:.1f}%)')
    plt.xlabel('Trading Day', fontsize=12)
    plt.ylabel('Weight', fontsize=12)
    plt.title(f'Enhanced TD3 Portfolio Evolution with Early Stopping - Testing Period ({test_steps} Days)', 
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{exp_dir}/plots/enhanced_td3_weight_evolution_by_days.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 🆕 2. 權重演化累積柱狀圖（按天數）
    plt.figure(figsize=(20, 10))
    
    # 採樣顯示（如果天數太多）
    sample_step = max(1, len(weights_data) // 50)
    sample_days = list(range(1, len(weights_data) + 1, sample_step))
    sample_weights = [weights_data[i-1] for i in sample_days]  # 調整索引
    
    weights_by_asset = []
    for i in range(n_assets):
        asset_weights = [w[i] for w in sample_weights]
        weights_by_asset.append(asset_weights)
    
    bottom_values = np.zeros(len(sample_days))
    
    for i, asset in enumerate(assets):
        if asset == 'CASH':
            plt.bar(sample_days, weights_by_asset[i], 
                    bottom=bottom_values, label=asset, color=cash_color, 
                    alpha=0.8, edgecolor='white', linewidth=1.5)
        else:
            plt.bar(sample_days, weights_by_asset[i], 
                    bottom=bottom_values, label=asset, color=colors[i], 
                    alpha=0.85, edgecolor='white', linewidth=0.5)
        bottom_values += np.array(weights_by_asset[i])
    
    plt.xlabel('Trading Day', fontsize=12)
    plt.ylabel('Weight', fontsize=12)
    plt.title(f'Enhanced TD3 Portfolio Weight Evolution with Early Stopping - By Trading Days', 
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{exp_dir}/plots/enhanced_td3_weight_stacked_by_days.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 🆕 3. 股票vs現金對比圖（按天數）
    plt.figure(figsize=(15, 8))
    
    stock_weights = [sum(w[:-1]) for w in weights_data]
    cash_weights = [w[-1] for w in weights_data]
    
    plt.plot(days, stock_weights, label=f'Total Stocks ({len(stock_names)})', 
             color='#C41E3A', linewidth=3.5, alpha=0.9)
    plt.plot(days, cash_weights, label='Cash', 
             color=cash_color, linewidth=3.5, linestyle='--', alpha=0.9)
    
    plt.axhline(y=0.5, color='#2F4F4F', linestyle=':', alpha=0.8, linewidth=2, label='50% Line')
    plt.axhline(y=0.8, color='#191970', linestyle=':', alpha=0.8, linewidth=2, label='Benchmark Stocks (80%)')
    plt.axhline(y=0.2, color='#FF8C00', linestyle=':', alpha=0.8, linewidth=2, label='Benchmark Cash (20%)')
    
    plt.xlabel('Trading Day', fontsize=12)
    plt.ylabel('Weight', fontsize=12)
    plt.title('Enhanced TD3 Stocks vs Cash Allocation with Early Stopping Over Trading Days', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{exp_dir}/plots/enhanced_td3_stocks_vs_cash_by_days.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 🆕 4. 綜合指標圖（修復x軸）
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Enhanced TD3 Comprehensive Metrics with Early Stopping', fontsize=16, fontweight='bold')
    
    # 投資組合價值增長（訓練期間 - 保持episode）
    episodes_list = [data['episode'] for data in episode_data]
    portfolio_values = [data['portfolio_value'] for data in episode_data]
    
    axes[0,0].plot(episodes_list, [(val/10000-1)*100 for val in portfolio_values], 'g-', linewidth=2, label='Enhanced TD3')
    axes[0,0].set_title('Portfolio Value Growth During Training (%)', fontweight='bold')
    axes[0,0].set_xlabel('Training Episode')
    axes[0,0].set_ylabel('Return (%)')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    
    # Episode獎勵（訓練期間 - 保持episode）
    returns = [data['return'] for data in episode_data]
    axes[0,1].plot(episodes_list, returns, 'b-', linewidth=2)
    axes[0,1].set_title('Training Episode Rewards (Fusion Version + Early Stopping)', fontweight='bold')
    axes[0,1].set_xlabel('Training Episode')
    axes[0,1].set_ylabel('Reward')
    axes[0,1].grid(True, alpha=0.3)
    
    # 現金比例（🆕 測試期間 - 改為天數）
    cash_ratios_test = [w[-1]*100 for w in weights_data]
    axes[1,0].plot(days, cash_ratios_test, 'orange', linewidth=2, label='Enhanced TD3')
    axes[1,0].axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Benchmark (20%)')
    axes[1,0].set_title('Cash Allocation During Testing (%)', fontweight='bold')
    axes[1,0].set_xlabel('Trading Day')  # 🆕 改為Trading Day
    axes[1,0].set_ylabel('Cash %')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend()
    
    # TD3訓練損失（訓練指標 - 保持update steps）
    if training_losses:
        actor_losses = [loss['actor_loss'] for loss in training_losses]
        update_steps = list(range(1, len(training_losses) + 1))
        
        axes[1,1].plot(update_steps, actor_losses, 'r-', linewidth=2, label='Actor Loss')
        
        axes[1,1].set_title('Enhanced TD3 Training Progress with Early Stopping', fontweight='bold')
        axes[1,1].set_xlabel('Training Update Step')
        axes[1,1].set_ylabel('Actor Loss', color='r')
        axes[1,1].tick_params(axis='y', labelcolor='r')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{exp_dir}/plots/enhanced_td3_comprehensive_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Enhanced TD3 visualizations with Early Stopping and trading days created!")

def create_detailed_backtest_comparison_by_days(exp_dir, test_env, stock_names):
    """🆕 生成以天數為x軸的詳細回測對比"""
    
    print("📊 Generating detailed backtest comparison by trading days with Early Stopping...")
    
    # 計算基準指標
    benchmark_values = test_env.calculate_equal_weight_benchmark()
    td3_values = test_env.portfolio_values_history
    
    min_length = min(len(benchmark_values), len(td3_values))
    benchmark_values = benchmark_values[:min_length]
    td3_values = td3_values[:min_length]
    
    # 計算收益率序列
    td3_returns = [(td3_values[i] / td3_values[i-1] - 1) for i in range(1, len(td3_values))]
    benchmark_returns = [(benchmark_values[i] / benchmark_values[i-1] - 1) for i in range(1, len(benchmark_values))]
    
    # 計算關鍵指標
    td3_total_return = (td3_values[-1] / 10000.0 - 1) * 100
    benchmark_total_return = (benchmark_values[-1] / 10000.0 - 1) * 100
    
    # 計算最大回撤
    def calculate_max_drawdown(values):
        peak_values = np.maximum.accumulate(values)
        drawdowns = (np.array(values) - peak_values) / peak_values
        return abs(np.min(drawdowns)) * 100
    
    td3_max_drawdown = calculate_max_drawdown(td3_values)
    benchmark_max_drawdown = calculate_max_drawdown(benchmark_values)
    
    # 計算Sharpe Ratio
    risk_free_rate = test_env.daily_cash_return * 252
    
    def calculate_sharpe(returns, risk_free_rate):
        if len(returns) == 0:
            return 0
        excess_returns = np.array(returns) - risk_free_rate/252
        return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8) * np.sqrt(252)
    
    td3_sharpe = calculate_sharpe(td3_returns, risk_free_rate)
    benchmark_sharpe = calculate_sharpe(benchmark_returns, risk_free_rate)
    
    # 其他指標
    td3_final_cash = test_env.weights[-1] * 100
    td3_volatility = np.std(td3_returns) * np.sqrt(252) * 100 if len(td3_returns) > 0 else 0
    benchmark_volatility = np.std(benchmark_returns) * np.sqrt(252) * 100 if len(benchmark_returns) > 0 else 0
    
    # 勝率
    if len(td3_returns) == len(benchmark_returns):
        win_days = sum(1 for i in range(len(td3_returns)) if td3_returns[i] > benchmark_returns[i])
        win_rate = (win_days / len(td3_returns)) * 100
    else:
        win_rate = 0
    
    # 生成對比表格
    comparison_data = {
        'Metric': [
            'Total Return (%)',
            'Max Drawdown (%)', 
            'Sharpe Ratio',
            'Volatility (%)',
            'Final Cash (%)',
            'Win Rate (%)',
            'Trading Days',
            'Final Portfolio Value ($)',
            'Early Stopping Used'
        ],
        'Enhanced TD3': [
            f"{td3_total_return:.1f}%",
            f"{td3_max_drawdown:.1f}%",
            f"{td3_sharpe:.2f}",
            f"{td3_volatility:.1f}%",
            f"{td3_final_cash:.1f}%",
            f"{win_rate:.1f}%",
            f"{len(td3_values)-1}",
            f"${td3_values[-1]:,.0f}",
            "YES ✅"
        ],
        'Equal Weight Benchmark': [
            f"{benchmark_total_return:.1f}%",
            f"{benchmark_max_drawdown:.1f}%", 
            f"{benchmark_sharpe:.2f}",
            f"{benchmark_volatility:.1f}%",
            "20.0%",
            "50.0%",
            f"{len(benchmark_values)-1}",
            f"${benchmark_values[-1]:,.0f}",
            "N/A"
        ]
    }
    
    # 保存為CSV
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(f'{exp_dir}/data/enhanced_td3_backtest_comparison_by_days.csv', index=False)
    
    # 🆕 生成以天數為x軸的可視化對比圖表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('📊 Enhanced TD3 Backtest Performance with Early Stopping (By Trading Days)', fontsize=16, fontweight='bold')
    
    # 1. 累積收益曲線（按天數）
    days = range(1, len(td3_values) + 1)
    td3_cumulative = [(val / 10000 - 1) * 100 for val in td3_values]
    benchmark_cumulative = [(val / 10000 - 1) * 100 for val in benchmark_values]
    
    axes[0,0].plot(days, td3_cumulative, label='Enhanced TD3 (Early Stop)', color='#2E8B57', linewidth=2.5)
    axes[0,0].plot(days, benchmark_cumulative, label='Benchmark', color='#4682B4', linewidth=2.5, linestyle='--')
    axes[0,0].set_title('Cumulative Return Comparison')
    axes[0,0].set_xlabel('Trading Day')
    axes[0,0].set_ylabel('Cumulative Return (%)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. 日收益率對比
    if len(td3_returns) > 0 and len(benchmark_returns) > 0:
        return_days = range(2, len(td3_values) + 1)  # 從第2天開始
        axes[0,1].plot(return_days, [r*100 for r in td3_returns], label='Enhanced TD3 (Early Stop)', color='#2E8B57', alpha=0.7, linewidth=1)
        axes[0,1].plot(return_days, [r*100 for r in benchmark_returns], label='Benchmark', color='#4682B4', alpha=0.7, linewidth=1)
        axes[0,1].set_title('Daily Returns Comparison')
        axes[0,1].set_xlabel('Trading Day')
        axes[0,1].set_ylabel('Daily Return (%)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    
    # 3. 現金配置對比（按天數）
    if hasattr(test_env, 'weights_history') and len(test_env.weights_history) > 0:
        cash_ratios = [w[-1] * 100 for w in test_env.weights_history]
        cash_days = range(1, len(cash_ratios) + 1)
        
        axes[1,0].plot(cash_days, cash_ratios, label='Enhanced TD3 Cash % (Early Stop)', color='#B8860B', linewidth=2.5)
        axes[1,0].axhline(y=20, color='#4682B4', linestyle='--', linewidth=2.5, label='Benchmark Cash (20%)')
        axes[1,0].set_title('Cash Allocation Over Trading Days')
        axes[1,0].set_xlabel('Trading Day')
        axes[1,0].set_ylabel('Cash Allocation (%)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    # 4. 滾動Sharpe Ratio對比
    window = 30  # 30天滾動窗口
    if len(td3_returns) >= window:
        rolling_days = range(window, len(td3_returns) + 1)
        td3_rolling_sharpe = []
        benchmark_rolling_sharpe = []
        
        for i in range(window, len(td3_returns) + 1):
            td3_window_returns = td3_returns[i-window:i]
            benchmark_window_returns = benchmark_returns[i-window:i]
            
            td3_rs = calculate_sharpe(td3_window_returns, risk_free_rate)
            benchmark_rs = calculate_sharpe(benchmark_window_returns, risk_free_rate)
            
            td3_rolling_sharpe.append(td3_rs)
            benchmark_rolling_sharpe.append(benchmark_rs)
        
        axes[1,1].plot(rolling_days, td3_rolling_sharpe, label='Enhanced TD3 (Early Stop)', color='#2E8B57', linewidth=2)
        axes[1,1].plot(rolling_days, benchmark_rolling_sharpe, label='Benchmark', color='#4682B4', linewidth=2, linestyle='--')
        axes[1,1].set_title(f'Rolling Sharpe Ratio ({window}-Day Window)')
        axes[1,1].set_xlabel('Trading Day')
        axes[1,1].set_ylabel('Sharpe Ratio')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{exp_dir}/plots/enhanced_td3_backtest_comparison_by_days.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印對比表格
    print("\n" + "="*80)
    print("📊 ENHANCED TD3 BACKTEST COMPARISON WITH EARLY STOPPING (BY TRADING DAYS)")
    print("="*80)
    
    print(f"{'Metric':<25} {'Enhanced TD3':<15} {'Benchmark':<20}")
    print("-" * 80)
    
    for i, metric in enumerate(comparison_data['Metric']):
        our = comparison_data['Enhanced TD3'][i]
        bench = comparison_data['Equal Weight Benchmark'][i]
        print(f"{metric:<25} {our:<15} {bench:<20}")
    
    print("="*80)
    
    return {
        'td3_total_return': td3_total_return,
        'benchmark_total_return': benchmark_total_return,
        'td3_sharpe': td3_sharpe,
        'benchmark_sharpe': benchmark_sharpe,
        'win_rate': win_rate,
        'trading_days': len(td3_values) - 1,
        'comparison_df': comparison_df
    }

def create_professional_stacked_chart_final(exp_dir, weights_data, stock_names, dates):
    """
    創建專業的堆疊面積圖 - 完全匹配你的參考圖片風格，加上Early Stopping標記
    """
    print("🎨 Generating professional stacked chart with Early Stopping (final version)...")
    
    try:
        # 1. 數據準備和驗證
        assets = stock_names + ['Cash']  # 注意這裡用 'Cash' 而不是 'CASH'
        weights_array = np.array(weights_data)
        
        print(f"📊 原始數據形狀: {weights_array.shape}")
        print(f"📊 資產列表: {assets}")
        
        # 確保是2D數組
        if weights_array.ndim == 1:
            weights_array = weights_array.reshape(1, -1)
        
        # 檢查並修復維度不匹配問題
        if weights_array.shape[1] != len(assets):
            print(f"⚠️ 維度不匹配: weights有{weights_array.shape[1]}列，但需要{len(assets)}列")
            if weights_array.shape[1] == len(assets) - 1:
                # 如果缺少現金列，添加它
                cash_weights = 1 - np.sum(weights_array, axis=1, keepdims=True)
                weights_array = np.column_stack([weights_array, cash_weights])
                print("✅ 自動添加了現金列")
            else:
                # 調整資產列表匹配數據
                assets = assets[:weights_array.shape[1]]
                print(f"✅ 調整資產列表為: {assets}")
        
        # 創建DataFrame
        df = pd.DataFrame(weights_array, columns=assets)
        
        # 處理日期索引
        if isinstance(dates, (pd.DatetimeIndex, list, np.ndarray)):
            try:
                if len(dates) >= len(df):
                    df.index = pd.to_datetime(dates[:len(df)])
                else:
                    # 如果日期不夠，生成默認日期
                    df.index = pd.date_range('2019-03-14', periods=len(df), freq='D')
                    print("⚠️ 日期數量不足，使用默認日期範圍")
            except:
                # 如果日期轉換失敗，使用步數作為索引
                df.index = range(len(df))
                print("⚠️ 日期轉換失敗，使用步數索引")
        else:
            df.index = range(len(df))
            print("⚠️ 使用步數索引")
        
        print(f"✅ DataFrame創建成功: {df.shape}")
        
        # 2. 可選的重採樣
        if len(df) > 200:
            original_len = len(df)
            df = df.resample('W').first().dropna() if hasattr(df.index, 'freq') else df.iloc[::7]
            print(f"📅 重採樣: {original_len} -> {len(df)} 數據點")
        
        # 3. 創建圖表 - 使用與你參考圖片相似的風格
        fig, ax = plt.subplots(figsize=(12, 6))  # 與參考圖片相似的比例
        
        # 設置顏色 - 使用柔和的顏色，類似參考圖片
        colors = [
            '#FF9999',  # 淺紅/粉色 (類似你圖片中的顏色)
            '#66B2FF',  # 淺藍色
            '#FFB366',  # 淺橙色  
            '#FF66FF',  # 淺紫色
            '#66FFB2',  # 淺綠色
            '#B366FF',  # 藍紫色
            '#FFFF66',  # 淺黃色
            '#66FFFF'   # 淺青色
        ]
        
        # 使用stackplot創建堆疊面積圖
        y_data = [df[asset].values for asset in assets]
        
        # 如果索引是數值型（步數），直接使用
        if isinstance(df.index[0], (int, np.integer)):
            x_data = df.index
        else:
            x_data = df.index
        
        stack = ax.stackplot(x_data, *y_data, 
                           labels=assets,
                           colors=colors[:len(assets)],
                           alpha=0.8)
        
        # 4. 格式化圖表 - 匹配參考圖片風格
        
        # Y軸格式化為權重（0-1）
        ax.set_ylim(0, 1)
        ax.set_ylabel('Weights', fontsize=12)
        
        # 如果使用日期索引
        if hasattr(df.index, 'year'):
            # 格式化X軸日期
            if len(df) > 50:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(df)//10)))
            else:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # 設置標題 - 類似參考圖片，加上Early Stopping標記
            start_date = df.index.min().strftime('%Y-%m-%d')
            end_date = df.index.max().strftime('%Y-%m-%d')
            ax.set_title(f'TD3 + Early Stopping - Portfolio Weights - OOS {start_date} to {end_date}', 
                        fontsize=14, color='gray')
        else:
            # 使用步數索引
            ax.set_xlabel('Time Steps')
            ax.set_title('Portfolio Weights Over Time (TD3 + Early Stopping)', fontsize=14, color='gray')
        
        # 圖例設置 - 放在右側，類似參考圖片
        legend = ax.legend(title='Stock', loc='center left', bbox_to_anchor=(1, 0.5), 
                          frameon=True, fancybox=True, shadow=False, fontsize=10)
        legend.get_title().set_fontsize(12)
        
        # 移除頂部和右側邊框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('lightgray')
        ax.spines['bottom'].set_color('lightgray')
        
        # 設置網格 - 淺色，類似參考圖片
        ax.grid(True, alpha=0.3, color='lightgray', linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # 調整布局
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)  # 為右側圖例留空間
        
        # 保存圖表
        os.makedirs(f'{exp_dir}/plots', exist_ok=True)
        plt.savefig(f'{exp_dir}/plots/professional_stacked_chart.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("✅ 專業堆疊面積圖創建成功（含Early Stopping）！")
        print(f"💾 保存位置: {exp_dir}/plots/professional_stacked_chart.png")
        
        return {
            'success': True,
            'data_points': len(df),
            'assets': assets,
            'date_range': f"{df.index.min()} to {df.index.max()}" if hasattr(df.index, 'year') else f"0 to {len(df)}"
        }
        
    except Exception as e:
        print(f"❌ 創建圖表時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    # 運行TD3實驗，集成Early Stopping機制
    print("🚀 Running TD3 Twin Delayed DDPG Experiment with Early Stopping...")
    exp_dir = run_td3_fusion_experiment_with_early_stopping("nine_stock_prices.csv")
    print(f"\n🎉 TD3 Early Stopping experiment completed: {exp_dir}")
    print(f"📊 Twin Delayed DDPG with deterministic policy and Early Stopping!")
    print(f"📊 All charts now use Trading Days (1, 2, 3, ..., N) as x-axis!")
    print(f"🏗️ TD3: Actor + Target Actor + Twin Critics + Target Critics")
    print(f"🎯 Deterministic Policy + Exploration Noise + Policy Delay + Target Smoothing")
    print(f"🔄 Early Stopping: 自動優化訓練，避免過度訓練")
    print(f"💾 Best Model Selection: 載入最佳驗證性能模型進行測試")