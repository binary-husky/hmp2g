# cython: language_level=3
# Cython配置，指定使用Python3语法级别
import copy  # 导入深拷贝模块

import numpy as np  # 导入numpy库，用于数值计算

# 从其他模块导入基础类和配置
from ALGORITHM.common.traj import TRAJ_BASE  # 轨迹基类
from UTIL.colorful import *  # 导入颜色工具（可能用于日志）
from UTIL.tensor_ops import my_view  # 导入张量视图操作函数

from .foundation import AlgorithmConfig  # 导入算法配置参数


class trajectory(TRAJ_BASE):
    """轨迹类，继承自TRAJ_BASE，管理单个环境轨迹数据"""

    def __init__(self, traj_limit, env_id):
        """初始化轨迹对象
        Args:
            traj_limit: 轨迹最大长度限制
            env_id: 环境实例ID
        """
        super().__init__(traj_limit, env_id)
        self.reference_track_name = "value"  # 设置参考轨迹名称（用于价值函数）

    def early_finalize(self):
        """预终止处理：标记需要奖励引导"""
        assert not self.readonly_lock  # 确保轨迹未锁定
        self.need_reward_bootstrap = True  # 启用奖励引导

    def set_terminal_obs(self, tobs):
        """设置终止状态观测值
        Args:
            tobs: 终止观测数据
        """
        self.tobs = copy.deepcopy(tobs)  # 深拷贝防止引用修改

    def cut_tail(self):
        """裁剪轨迹尾部冗余数据，处理NaN无效点"""
        super().cut_tail()  # 调用父类裁剪方法
        TJ = lambda key: getattr(self, key)  # 快捷方法获取轨迹属性

        # 获取参考轨迹数据（如价值函数）
        reference_track = getattr(self, self.reference_track_name)

        # 处理奖励引导场景（当前设计不应进入）
        if self.need_reward_bootstrap:
            assert False, "it should not go here if everything goes as expected"
            # 找到最后一个非NaN时间点作为引导点
            T = np.where(~np.isnan(reference_track.squeeze()))[0][-1]
            # 保存引导值和终止观测
            self.boot_strap_value = {
                "bootstrap_value": TJ("value").squeeze()[T].copy(),
            }
            assert not hasattr(self, "tobs")  # 确保未设置过终止观测
            self.set_terminal_obs(TJ("g_obs")[T].copy())  # 设置终止观测
            reference_track[T] = np.nan  # 标记该点为无效

        # 判断轨迹有效性（是否存在有效数据）
        p_invalid = np.isnan(my_view(reference_track, [0, -1])).any(axis=-1)
        p_valid = ~p_invalid  # 有效时间点掩码

        if p_invalid.all():  # 全部无效则废弃轨迹
            self.deprecated_flag = True
            return

        # 奖励前推：将无效时间点的奖励合并到前一个有效点
        reward = TJ("reward")
        for i in reversed(range(self.time_pointer)):
            if p_invalid[i] and i != 0:  # 无效且非首帧
                reward[i - 1] += reward[i]  # 累加奖励
                reward[i] = np.nan  # 清空当前奖励
        setattr(self, "reward", reward)  # 更新奖励属性

        # 剪裁所有轨迹数据，保留有效时间点
        for key in self.key_dict:
            setattr(self, key, TJ(key)[p_valid])
        return

    def reward_push_forward(self, dead_mask):
        """基于死亡掩码前推奖励（用于终端状态处理）
        Args:
            dead_mask: 死亡状态掩码数组
        """
        if AlgorithmConfig.gamma_in_reward_forwarding:
            # 使用衰减因子γ的奖励前推
            gamma = AlgorithmConfig.gamma_in_reward_forwarding_value
            for i in reversed(range(self.time_pointer)):
                if i == 0:
                    continue
                # 死亡状态下将奖励乘γ前推
                self.reward[i - 1] += np.where(dead_mask[i], self.reward[i] * gamma, 0)
                self.reward[i] = np.where(
                    dead_mask[i], 0, self.reward[i]
                )  # 清空死亡帧奖励
        else:
            # 无衰减的纯奖励前推
            for i in reversed(range(self.time_pointer)):
                if i == 0:
                    continue
                # 死亡状态下直接前推奖励
                self.reward[i - 1] += np.where(dead_mask[i], self.reward[i], 0)
                self.reward[i] = np.where(dead_mask[i], 0, self.reward[i])
        return

    def finalize(self):
        """轨迹最终化处理：计算威胁矩阵和GAE回报"""
        self.readonly_lock = True  # 锁定轨迹不可修改
        assert not self.deprecated_flag  # 确保轨迹有效
        TJ = lambda key: getattr(self, key)  # 属性访问快捷方式

        # 检查奖励数据有效性
        assert not np.isnan(TJ("reward")).any()

        # 构建死亡掩码（通过观测数据判断）
        tmp = np.isnan(my_view(self.obs, [0, 0, -1]))
        dead_mask = tmp.all(-1)  # 全维度为NaN才标记死亡

        # 奖励前推处理
        self.reward_push_forward(dead_mask)

        # 构建威胁矩阵（记录连续存活步数）
        threat = np.zeros(shape=dead_mask.shape) - 1  # 初始化威胁矩阵
        assert dead_mask.shape[0] == self.time_pointer  # 确保维度一致

        # 反向遍历计算存活步数
        for i in reversed(range(self.time_pointer)):
            if i + 1 < self.time_pointer:
                threat[: (i + 1)] += (~(dead_mask[i + 1] & dead_mask[i])).astype(int)
            elif i + 1 == self.time_pointer:
                threat[:] += (~dead_mask[i]).astype(int)

        # 威胁值裁剪
        SAFE_LIMIT = 11
        threat = np.clip(threat, -1, SAFE_LIMIT)
        setattr(self, "threat", np.expand_dims(threat, -1))  # 添加维度

        # 使用GAE计算回报
        self.gae_finalize_return(
            reward_key="reward", value_key="value", new_return_name="return"
        )

    def gae_finalize_return(self, reward_key, value_key, new_return_name):
        """使用GAE(广义优势估计)计算回报
        Args:
            reward_key: 奖励属性名
            value_key: 价值属性名
            new_return_name: 返回值存储名称
        """
        # 获取GAE参数
        gamma = AlgorithmConfig.gamma
        tau = AlgorithmConfig.tau

        # 获取奖励和价值数据
        rewards = getattr(self, reward_key)
        value = getattr(self, value_key)
        length = rewards.shape[0]  # 获取轨迹长度

        # 确保维度一致
        assert rewards.shape[0] == value.shape[0]
        if rewards.ndim == value.ndim - 1:
            rewards = np.expand_dims(rewards, -1)  # 扩展维度对齐

        # 初始化回报属性
        setattr(self, new_return_name, np.zeros_like(value))
        self.key_dict.append(new_return_name)  # 注册新属性

        returns = getattr(self, new_return_name)
        # 获取引导值（若需要）
        boot_strap = (
            0
            if not self.need_reward_bootstrap
            else self.boot_strap_value["bootstrap_" + value_key]
        )

        # 反向计算GAE
        gae = 0
        for step in reversed(range(length)):
            if step == length - 1:  # 最后一帧
                value_preds_delta = rewards[step] + gamma * boot_strap - value[step]
            else:  # 中间帧
                value_preds_delta = (
                    rewards[step] + gamma * value[step + 1] - value[step]
                )
            gae = value_preds_delta + gamma * tau * gae  # 计算GAE
            returns[step] = gae + value[step]  # 回报=优势+价值


# 轨迹池管理器
class TrajPoolManager(object):
    def __init__(self):
        self.cnt = 0  # 计数器

    def absorb_finalize_pool(self, pool):
        """处理轨迹池：裁剪、过滤、最终化
        Args:
            pool: 轨迹池列表
        Returns:
            task: 任务类型
            pool: 处理后的轨迹池
        """
        for traj_handle in pool:
            traj_handle.cut_tail()  # 裁剪轨迹
        # 过滤无效轨迹
        pool = list(filter(lambda traj: not traj.deprecated_flag, pool))
        for traj_handle in pool:
            traj_handle.finalize()  # 最终化轨迹
        self.cnt += 1
        task = ["train"]  # 默认任务
        return task, pool


# 轨迹管理基类
class TrajManagerBase(object):
    def __init__(self, n_env, traj_limit):
        """初始化轨迹管理系统
        Args:
            n_env: 环境实例数量
            traj_limit: 轨迹长度限制
        """
        self.n_env = n_env
        self.traj_limit = traj_limit
        self.update_cnt = 0  # 更新计数
        self.traj_pool = []  # 轨迹池
        self.registered_keys = []  # 注册的数据键
        # 为每个环境创建实时轨迹
        self.live_trajs = [
            trajectory(self.traj_limit, env_id=i) for i in range(self.n_env)
        ]
        self.live_traj_frame = [0] * self.n_env  # 帧计数器
        self._traj_lock_buf = None  # 轨迹锁缓冲区
        self.patience = 1000  # 完整性检查耐心值

    def __check_integraty(self, traj_frag):
        """检查轨迹片段完整性"""
        if self.patience < 0:
            return
        self.patience -= 1  # 递减检查次数
        # 注册新出现的键
        for key in traj_frag:
            if key not in self.registered_keys and not key.startswith("_"):
                self.registered_keys.append(key)
        # 验证所有注册键都存在
        for key in self.registered_keys:
            assert key in traj_frag, (
                "this key sometimes disappears from the traj_frag:",
                key,
            )

    def batch_update(self, traj_frag):
        """批量更新轨迹数据
        Args:
            traj_frag: 轨迹片段
        """
        self.__check_integraty(traj_frag)
        # 提取控制标志
        done = traj_frag.pop("_DONE_")
        skip = traj_frag.pop("_SKIP_")
        tobs = traj_frag.pop("_TOBS_")

        # 类型转换：布尔值转列表
        if isinstance(done, bool):
            done = [done] * self.n_env
        if isinstance(skip, bool):
            skip = [skip for i in range(self.n_env)]
        n_active = sum(~skip)
        # feed
        cnt = 0
        for env_i in range(self.n_env):
            if skip[env_i]:
                continue  # 跳过冻结环境
            frag_index = cnt
            cnt += 1
            traj_handle = self.live_trajs[env_i]
            # 存储轨迹数据
            for key in traj_frag:
                self.traj_remember(
                    traj_handle,
                    key=key,
                    content=traj_frag[key],
                    frag_index=frag_index,
                    n_active=n_active,
                )
            # 更新帧计数和时间指针
            self.live_traj_frame[env_i] += 1
            traj_handle.time_shift()
            # 处理完成状态
            if done[env_i]:
                assert tobs[env_i] is not None  # 验证终止观测存在
                traj_handle.set_terminal_obs(tobs[env_i])  # 设置终止观测
                self.traj_pool.append(traj_handle)  # 添加到轨迹池
                # 重置环境轨迹
                self.live_trajs[env_i] = trajectory(self.traj_limit, env_id=env_i)
                self.live_traj_frame[env_i] = 0

    def traj_remember(self, traj, key, content, frag_index, n_active):
        """存储轨迹数据（支持嵌套结构）
        Args:
            traj: 轨迹对象
            key: 数据键
            content: 数据内容
            frag_index: 片段索引
            n_active: 激活环境数量
        """
        if content is None:
            traj.remember(key, None)
        elif isinstance(content, dict):  # 处理嵌套字典
            for sub_key in content:
                self.traj_remember(
                    traj, f"{key}>{sub_key}", content[sub_key], frag_index, n_active
                )
        else:
            assert n_active == len(content), "数据长度不匹配"
            traj.remember(key, content[frag_index])  # 存储对应索引数据


# 批量轨迹管理器
class BatchTrajManager(TrajManagerBase):
    def __init__(self, n_env, traj_limit, trainer_hook):
        """初始化批量轨迹管理器
        Args:
            n_env: 环境数量
            traj_limit: 轨迹长度限制
            trainer_hook: 训练回调函数
        """
        super().__init__(n_env, traj_limit)
        self.trainer_hook = trainer_hook  # 训练回调
        self.train_traj_needed = AlgorithmConfig.train_traj_needed  # 训练所需轨迹数
        self.pool_manager = TrajPoolManager()  # 轨迹池管理器

    def update(self, traj_frag, index):
        """更新指定索引的轨迹数据
        Args:
            traj_frag: 轨迹片段
            index: 环境索引列表
        """
        assert traj_frag is not None
        for j, env_i in enumerate(index):
            traj_handle = self.live_trajs[env_i]
            for key in traj_frag:
                if traj_frag[key] is None:
                    assert False, key  # 检测空值
                if isinstance(traj_frag[key], dict):  # 处理嵌套字典
                    for sub_key in traj_frag[key]:
                        content = traj_frag[key][sub_key][j]
                        traj_handle.remember(f"{key}>{sub_key}", content)
                else:
                    content = traj_frag[key][j]
                    traj_handle.remember(key, content)
            # 更新计数器和时间指针
            self.live_traj_frame[env_i] += 1
            traj_handle.time_shift()
        return

    def feed_traj_framedata(self, traj_frag, require_hook=False):
        """轨迹数据入口函数
        Args:
            traj_frag: 轨迹片段
            require_hook: 是否需要钩子函数
        Returns:
            解锁函数或None
        """
        assert self._traj_lock_buf is None  # 确保无锁
        if require_hook:  # 需要分阶段接收数据
            assert "_SKIP_" in traj_frag
            assert "_DONE_" not in traj_frag
            assert "reward" not in traj_frag
            self._traj_lock_buf = traj_frag  # 缓存未完成数据
            return self.unlock_fn  # 返回解锁函数
        else:  # 直接处理完整数据
            assert "_DONE_" in traj_frag
            assert "_SKIP_" in traj_frag
            self.batch_update(traj_frag=traj_frag)
            return

    def clear_traj_pool(self):
        """清空轨迹池并执行更新"""
        print(f"do update {self.update_cnt}")
        _, self.traj_pool = self.pool_manager.absorb_finalize_pool(self.traj_pool)
        self.traj_pool = []  # 重置轨迹池
        return self.update_cnt

    def train_and_clear_traj_pool(self):
        """执行训练并清空轨迹池"""
        print(f"do update {self.update_cnt}")
        current_task_l, self.traj_pool = self.pool_manager.absorb_finalize_pool(
            self.traj_pool
        )
        for current_task in current_task_l:
            ppo_update_cnt = self.trainer_hook(self.traj_pool, current_task)  # 执行训练
        self.traj_pool = []  # 清空轨迹池
        self.update_cnt += 1
        return self.update_cnt

    def can_exec_training(self):
        """检查是否满足训练条件"""
        return len(self.traj_pool) >= self.train_traj_needed

    def unlock_fn(self, traj_frag):
        """解锁并完成轨迹数据"""
        assert self._traj_lock_buf is not None
        traj_frag.update(self._traj_lock_buf)  # 合并缓存数据
        self._traj_lock_buf = None  # 清空锁
        assert "_DONE_" in traj_frag
        assert "_SKIP_" in traj_frag
        self.batch_update(traj_frag=traj_frag)  # 完成数据更新
