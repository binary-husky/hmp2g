# 导入依赖库
import numpy as np  # 数值计算库

from MISSION.env_router import make_env_function  # 环境工厂函数
from UTIL.colorful import print亮红  # 彩色打印工具

# 文档说明：
# 使用进程池控制多个环境同步步进
# SuperPool.add_target: 在每个进程初始化EnvAutoReset对象
# SuperPool.exec_target: 在进程内执行对象方法
# 例如：
# self.SuperPool.add_target(name='env', lam=EnvAutoReset, args_list=env_args_dict_list)
# self.SuperPool.exec_target(name='env', dowhat='step', args_list=actions)


class EnvAutoReset(object):
    """在子进程中执行的环境包装器"""

    def __init__(self, env_args_dict):
        # 解析环境参数
        env_name = env_args_dict["env_name"]  # 环境名称
        proc_index = env_args_dict["proc_index"]  # 进程索引

        # 创建环境初始化函数
        env_init_fn = make_env_function(env_name=env_name, rank=proc_index)

        # 初始化实际环境
        self._env = env_init_fn()

        # 状态标记
        self._cold_start = True  # 首次启动标志
        self._suffer_reset = False  # 自动重置标志

        # 缓存空间信息
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

        # 步进/重置缓存
        self._step_cache = None
        self._reset_cache = None

    def step(self, act):
        """执行环境步进操作"""
        # 处理跳过指令（通过NaN动作触发）
        if np.isnan(act).any():
            assert self._suffer_reset
            assert self._step_cache is not None
            return self._step_cache

        # 正常步进逻辑
        ob, reward, done, info = self._env.step(act)

        # 警告：避免返回列表类型的观测数据
        if isinstance(ob, list):
            print("[shm_env.py] Warning, ob is list, which is low efficient")
            ob = np.array(ob, dtype=object)

        # 处理环境终止逻辑
        if np.any(done):
            # 1. 存储终止状态到info
            if info is None:
                info = {"obs-echo": ob}
            else:
                assert isinstance(info, dict)
                info.update({"obs-echo": ob.copy()})

            # 2. 自动重置环境
            ob = self._reset_cache = self._real_reset()

            # 3. 处理特殊环境（如星际争霸）的元组返回值
            if isinstance(ob, tuple):
                ob, info_reset = ob
                info = self.dict_update(info, info_reset)
        else:
            self._suffer_reset = False
            self._reset_cache = None

        # 缓存当前状态用于异常恢复
        self._step_cache = [ob, reward, done, info]
        return (ob, reward, done, info)

    def dict_update(self, info, info_reset):
        """合并两个info字典"""
        for key in info_reset:
            if key in info:
                info[key + "-echo"] = info.pop(key)
        info.update(info_reset)
        return info

    def reset(self):
        """环境重置逻辑"""
        if self._cold_start:
            # 首次初始化
            self._cold_start = False
            return self._real_reset()
        elif self._suffer_reset:
            # 返回已缓存的重置结果
            assert self._reset_cache is not None
            return self._reset_cache
        else:
            # 手动重置警告
            print("[shm_env.py] We do not recommand resetting manually.")
            return self._real_reset()

    def _real_reset(self):
        """实际重置方法"""
        self._suffer_reset = True
        return self._env.reset()

    # 其他辅助方法
    def sleep(self):
        return self._env.sleep()

    def render(self):
        return self._env.render()

    def close(self):
        return None

    # 获取空间信息的方法
    def get_act_space(self):
        return self.action_space

    def get_obs_space(self):
        return self.observation_space

    def get_act_space_str(self):
        return str(self.action_space)

    def get_obs_space_str(self):
        return str(self.observation_space)

    def __del__(self):
        # 析构函数
        if hasattr(self, "env"):
            del self._env


class SuperpoolEnv(object):
    """主进程环境管理器"""

    def __init__(self, process_pool, env_args_dict_list, spaces=None):
        # 绑定进程池
        self.SuperPool = process_pool

        # 环境数量
        self.num_envs = len(env_args_dict_list)

        # 环境标记
        self.env_name_marker = env_args_dict_list[0][0]["marker"]
        self.env = "env" + self.env_name_marker

        # 添加环境目标
        self.SuperPool.add_target(
            name=self.env, lam=EnvAutoReset, args_list=env_args_dict_list
        )

        # 获取空间信息（尝试直接传输，失败则用字符串）
        try:
            self.observation_space = self.SuperPool.exec_target(
                name=self.env, dowhat="get_obs_space"
            )[0]
            self.action_space = self.SuperPool.exec_target(
                name=self.env, dowhat="get_act_space"
            )[0]
        except:
            print亮红(
                "[shm_env.py] Gym Space is unable to transfer between processes, using string instead"
            )
            self.observation_space = self.SuperPool.exec_target(
                name=self.env, dowhat="get_obs_space_str"
            )[0]
            self.action_space = self.SuperPool.exec_target(
                name=self.env, dowhat="get_act_space_str"
            )[0]

    def get_space(self):
        """获取空间信息"""
        return {"obs_space": self.observation_space, "act_space": self.action_space}

    def step(self, actions):
        """批量执行步进操作"""
        results = self.SuperPool.exec_target(
            name=self.env, dowhat="step", args_list=actions
        )
        obs, rews, dones, infos = zip(*results)

        try:
            return np.stack(obs), np.stack(rews), np.stack(dones), np.stack(infos)
        except:
            raise RuntimeError(
                "[shm_env.py] Unaligned obs/reward/done is illegal!", obs, rews, dones
            )

    def reset(self):
        """批量执行重置操作"""
        results = self.SuperPool.exec_target(name=self.env, dowhat="reset")
        if isinstance(results[0], tuple):
            # 处理特殊环境的元组返回值
            obs, infos = zip(*results)
            return np.stack(obs), np.stack(infos)
        else:
            # 标准环境返回
            return np.stack(results)

    def sleep(self):
        # 弃用的睡眠接口
        self.SuperPool.exec_target(name=self.env, dowhat="sleep")
