# 导入基础依赖库
import copy
import json
import os
import pickle
import platform
import subprocess
import sys
import time

import numpy as np
import redis  # 用于旧版Pymarl进程间通信

# 项目内部模块导入
from config import GlobalConfig  # 全局配置
from UTIL.network import TcpServerP2P, UnixTcpServerP2P, find_free_port  # 网络工具


class AlgorithmConfig:
    """
    算法配置类：定义与Pymarl2算法交互的关键参数
    """

    pymarl2_alg_select = "qmix"  # 当前选择的算法（参考pymarl2配置）
    pymarl2_runner_select = "efficient_parallel_runner"  # 运行器类型
    state_compat = "pad"  # 状态兼容模式：pad/obs_mean/obs_cat
    pymarl_config_injection = {}  # 注入到pymarl的额外配置
    use_shell = ""  # 使用的外壳环境名称
    use_shell_normalization = False  # 是否启用外壳环境归一化
    load_checkpoint = False  # 是否加载检查点
    load_specific_checkpoint = ""  # 特定检查点路径
    action_converter = (
        "ALGORITHM.common.converter->LegacyUmapActionConverter"  # 动作转换器路径
    )


def encrpt_string(s):
    """
    简单字符串加密函数：通过字符ASCII码相加实现混淆
    @param s: 原始字符串
    @return: 加密后的字符串（格式：数字_数字_..._）
    """
    k = "".join(["@"] * 1000)  # 固定密钥字符串
    encry_str = ""
    for i, j in zip(s, k):  # 逐字符加密
        temp = str(ord(i) + ord(j)) + "_"
        encry_str = encry_str + temp
    return encry_str


def decrpt_string(p):
    """
    解密函数：将加密字符串还原
    @param p: 加密后的字符串
    @return: 解密后的原始字符串
    """
    k = "".join(["@"] * 1000)
    dec_str = ""
    for i, j in zip(p.split("_")[:-1], k):  # 忽略最后一个空元素
        temp = chr(int(i) - ord(j))  # ASCII差值还原字符
        dec_str = dec_str + temp
    return dec_str


class PymarlFoundation:
    """
    核心类：实现Pymarl2算法与环境的高效并行交互
    通过TCP/Unix套接字与子进程通信，支持多线程环境控制
    """

    def init_pymarl(self):
        """
        初始化Pymarl环境：启动子进程并建立通信通道
        1. 验证运行器类型是否匹配
        2. 创建日志文件
        3. 生成唯一标识符
        4. 注入全局配置
        5. 启动Pymarl子进程
        6. 创建通信服务器
        """
        assert AlgorithmConfig.pymarl2_runner_select == "efficient_parallel_runner", (
            "if not qmix, please use PymarlFoundation below"
        )
        # 创建日志文件
        fp = open("%s/pymarl.log" % GlobalConfig.logdir, "w+")
        import atexit
        import uuid

        self.remote_uuid = uuid.uuid1().hex  # 生成唯一进程标识

        # 注入全局配置到pymarl配置系统
        if "config.py->GlobalConfig" not in AlgorithmConfig.pymarl_config_injection:
            AlgorithmConfig.pymarl_config_injection["config.py->GlobalConfig"] = {}

        AlgorithmConfig.pymarl_config_injection["config.py->GlobalConfig"].update(
            {
                "HmpRoot": os.getcwd(),  # 当前工作目录
                "ExpNote": GlobalConfig.note,  # 实验备注
                "draw_mode": GlobalConfig.draw_mode,  # 绘图模式
                "logdir": GlobalConfig.logdir,  # 日志目录
                "n_thread": GlobalConfig.num_threads,  # 线程数
                "seed": GlobalConfig.seed,  # 随机种子
                "activate_logger": GlobalConfig.activate_logger,  # 是否启用日志
                "train_time_testing": GlobalConfig.train_time_testing,  # 训练时测试
                "test_interval": GlobalConfig.test_interval,  # 测试间隔
                "test_only": GlobalConfig.test_only,  # 仅测试模式
                "test_epoch": GlobalConfig.test_epoch,  # 测试轮次
            }
        )

        # Windows特殊处理：分配兼容端口
        if platform.system() == "Windows":
            compat_windows_port = find_free_port()
            AlgorithmConfig.pymarl_config_injection["config.py->GlobalConfig"].update(
                {"compat_windows_port": compat_windows_port}
            )

        # 启动Pymarl子进程
        assert os.path.exists("./THIRDPARTY/pymarl2/pymarl2src/main.py")
        subprocess.Popen(
            [
                sys.executable,  # 当前Python解释器路径
                "./THIRDPARTY/pymarl2/pymarl2src/main.py",
                "--force",  # 强制运行
                "--config=%s" % AlgorithmConfig.pymarl2_alg_select,  # 算法配置
                "--env-config=HMP_compat",  # 环境配置
                "--pymarl_config_injection=%s"
                % encrpt_string(
                    json.dumps(AlgorithmConfig.pymarl_config_injection)
                ),  # 注入配置
                "--env_uuid=%s" % self.remote_uuid,  # 环境唯一标识
            ],
            stdout=fp,  # 输出重定向到日志
        )

        # 根据系统选择通信方式
        if platform.system() == "Windows":
            unix_path = ("localhost", compat_windows_port)
            self.remote_link_server = TcpServerP2P(unix_path, obj="pickle")  # TCP通信
        else:
            unix_path = "TEMP/Sockets/unix/%s" % self.remote_uuid
            self.remote_link_server = UnixTcpServerP2P(
                unix_path, obj="pickle"
            )  # Unix套接字

        # 注册退出清理函数
        atexit.register(lambda: self.__del__())
        time.sleep(5)  # 等待初始化完成

    def reset_confirm_all(self):
        """
        确认所有环境重置完成：收集所有线程的初始状态
        @return: 包含每个线程状态、可用动作和观测值的列表
        """
        assert self.team_intel["Env-Suffered-Reset"].all()  # 确保所有环境已重置
        res = [
            {
                "state": self.get_state_of(i),
                "avail_actions": self.get_avail_actions_of(i),
                "obs": self.get_obs_of(i),
            }
            for i in range(self.n_thread)  # 遍历所有线程
        ]
        return res

    def step_all(self, action, remote_active_flag):
        """
        执行所有环境步进：处理输入动作并存储
        @param action: 动作数组
        @param remote_active_flag: 活动标志位
        """
        assert (
            remote_active_flag == ~self.team_intel["ENV-PAUSE"]
        ).all()  # 验证暂停状态
        for i in range(self.n_thread):
            if self.team_intel["ENV-PAUSE"][i]:  # 暂停状态处理
                assert action[i] is None
                self.current_actions[i] = np.array([np.nan] * self.n_agent)  # 填充NaN
            else:
                assert action[i] is not None
                self.current_actions[i] = action[i]  # 存储有效动作

        # 区分测试和训练模式存储标志位
        if self.team_intel["Test-Flag"]:
            self.remote_active_flag_T = remote_active_flag
        else:
            self.remote_active_flag = remote_active_flag
        return None  # 异步处理不立即返回

    def get_step_future(self):
        """
        获取步进结果：组装包含奖励、终止状态等的环境反馈
        @return: 步进结果列表或测试阶段标识
        """
        res = []
        # 选择测试/训练模式的活动标志
        remote_active_flag = (
            self.remote_active_flag_T
            if self.team_intel["Test-Flag"]
            else self.remote_active_flag
        )

        if remote_active_flag is None:
            return "entering_test_phase"  # 测试阶段标识

        for i, active in enumerate(remote_active_flag):
            if not active:
                continue
            # 构建环境信息字典
            env_info = self.team_intel["Latest-Team-Info"][i].copy()
            # 移除冗余字段
            for key in [
                "obs-echo",
                "state-echo",
                "state",
                "avail-act-echo",
                "avail-act",
            ]:
                if key in env_info:
                    env_info.pop(key)
            env_info["testing"] = self.team_intel["Test-Flag"]  # 添加测试标记

            res.append(
                {
                    "state": self.get_state_of(i),
                    "avail_actions": self.get_avail_actions_of(i),
                    "obs": self.get_obs_of(i),
                    "reward": self.team_intel["Latest-Reward"][i],
                    "terminated": self.team_intel["Env-Suffered-Reset"][i],
                    "info": env_info,
                }
            )

        # 清理标志位
        if self.team_intel["Test-Flag"]:
            self.remote_active_flag_T = None
        else:
            self.remote_active_flag_T = None
        return res

    def get_env_info(self):
        """
        获取环境基本信息：状态/观测维度、动作空间等
        @return: 包含环境元数据的字典
        """
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.get_n_agents(),
            "episode_limit": self.get_episode_limit(),
        }
        return env_info

    def __del__(self):
        """析构函数：清理资源"""
        print("PymarlFoundation end, cleaning")
        self.remote_link_server.__del__()  # 关闭通信连接

    def __init__(self, n_agent, n_thread, space, mcv, team=None):
        """
        初始化方法
        @param n_agent: 智能体数量
        @param n_thread: 线程数量
        @param space: 环境空间定义
        @param mcv: 共享内存变量（未使用）
        @param team: 团队信息（未使用）
        """
        # 基础属性初始化
        self.n_thread = n_thread
        self.n_agent = n_agent
        self.handler = [None for _ in range(self.n_thread)]
        self.remote_pymarl_start_cmd = ""
        self.remote_pymarl_interface = ""
        self.team_intel = None  # 团队智能体信息
        self.space = space  # 环境空间
        self.uuid2threads = {"thread_cnt": 0}  # UUID到线程的映射
        self.current_actions = [None for _ in range(self.n_thread)]  # 当前动作
        self.previous_action = None  # 前一次动作
        self.previous_ENV_PAUSE = None  # 前一次暂停状态
        self.ScenarioConfig = GlobalConfig.ScenarioConfig  # 场景配置

        self.init_pymarl()  # 初始化Pymarl连接

        # 外壳环境初始化（如果启用）
        if AlgorithmConfig.use_shell != "":
            if AlgorithmConfig.use_shell == "mini_shell_uhmap":
                from .mini_shell_uhmap import ShellEnv

                self.shell = ShellEnv(self, n_agent, n_thread, space, mcv, team)
            else:
                assert False, "未知的外壳环境类型"

    def get_current_mode(self):
        """获取当前模式（训练/测试）"""
        return "Testing" if self.team_intel["Test-Flag"] else "Training"

    def get_state_size(self):
        """
        获取状态空间大小：根据兼容模式处理不同来源的数据
        @return: 状态向量维度
        """
        if AlgorithmConfig.state_compat == "native":
            try:
                return self.space["obs_space"]["state_shape"]
            except:
                info = self.team_intel["Latest-Team-Info"][0]
                if "state" not in info:
                    return 0
                else:
                    return info["state"].shape[-1]
        else:
            return self.space["obs_space"]["state_shape"]

    def get_obs_size(self):
        """
        获取观测空间大小：优先从space获取，失败则从最新观测数据推导
        @return: 观测向量维度
        """
        try:
            return self.space["obs_space"]["obs_shape"]
        except:
            obs = self.team_intel["Latest-Obs"][0]
            return obs.shape[-1]

    def get_n_agents(self):
        """
        获取智能体数量：优先从space获取，失败则返回初始化值
        @return: 智能体数量
        """
        try:
            self.n_agent = self.space["act_space"]["n_agents"]
            return self.n_agent
        except:
            return self.n_agent

    def get_episode_limit(self):
        """获取单个episode的最大步数"""
        return int(self.ScenarioConfig.MaxEpisodeStep)

    def get_total_actions(self):
        """
        获取总动作数：优先从space获取，失败则使用场景配置
        @return: 动作空间大小
        """
        try:
            self.n_actions = self.space["act_space"]["n_actions"]
            return self.n_actions
        except:
            assert self.ScenarioConfig.use_simple_action_space
            self.n_actions = self.ScenarioConfig.n_actions
            return self.ScenarioConfig.n_actions

    def confirm_reset(self):
        """确认环境重置完成（供Pymarl调用）"""
        return True

    def close(self):
        """关闭环境（占位函数）"""
        return

    # PymarlFoundation 类续（方法实现）
    # @basic_io_call
    def get_stats_of(self):
        """
        获取当前环境统计信息：移除冗余字段后返回环境状态信息
        用于监控和调试，过滤掉重复或不必要的观测状态数据
        """
        which_env = self.get_env_with_currentuuid()  # 获取当前UUID对应的环境索引
        env_info = self.team_intel["Latest-Team-Info"][which_env].copy()
        # 移除冗余字段以优化传输效率
        for key in ["obs-echo", "state-echo", "state", "avail-act-echo", "avail-act"]:
            if key in env_info:
                env_info.pop(key)
        return env_info

    # @basic_io_call
    def get_state_of(self, which_env):
        """
        获取指定环境的状态数据：根据暂停状态返回最新或缓存状态
        @param which_env: 环境索引
        @return: 状态向量
        """
        if self.team_intel["Env-Suffered-Reset"][which_env]:  # 如果环境已重置
            if not self.team_intel["ENV-PAUSE"][which_env]:  # 非暂停状态
                return self.team_intel["Latest-Team-Info"][which_env][
                    "state"
                ]  # 返回最新状态
            else:  # 暂停状态
                return self.team_intel["Latest-Team-Info"][which_env][
                    "state-echo"
                ]  # 返回缓存状态
        # 正常运行状态
        return self.team_intel["Latest-Team-Info"][which_env]["state"]

    # @basic_io_call
    def get_avail_actions_of(self, which_env):
        """
        获取指定环境可用动作：优先使用环境提供的动作掩码
        @param which_env: 环境索引
        @return: 可用动作矩阵 (n_agent x n_actions)
        """
        if "avail-act" in self.team_intel["Latest-Team-Info"][which_env]:
            return self.team_intel["Latest-Team-Info"][which_env]["avail-act"]
        else:
            # 默认返回全可用动作
            return np.ones((self.n_agent, self.n_actions))

    # @basic_io_call
    def get_obs_of(self, which_env):
        """
        获取指定环境观测值：根据暂停状态返回最新或缓存观测
        @param which_env: 环境索引
        @return: 观测向量
        """
        if self.team_intel["Env-Suffered-Reset"][which_env]:  # 如果环境已重置
            if not self.team_intel["ENV-PAUSE"][which_env]:  # 非暂停状态
                return self.team_intel["Latest-Obs"][which_env]  # 返回最新观测
            else:  # 暂停状态
                return self.team_intel["Terminal-Obs-Echo"][which_env]  # 返回缓存观测
        # 正常运行状态
        return self.team_intel["Latest-Obs"][which_env]

    def deal_with_pymarl(self):
        """
        核心通信循环：等待并处理Pymarl子进程请求
        持续监听通信通道，执行收到的命令直到所有动作完成
        """
        while any([act is None for act in self.current_actions]):
            dgram = self.remote_link_server.wait_next_dgram()  # 等待数据包
            cmd = dgram[0]
            args = dgram[1:]
            assert isinstance(cmd, str)
            res = getattr(self, cmd)(*args)  # 执行对应方法
            self.remote_link_server.reply_last_client(res)  # 返回结果

    def clear_actions(self):
        """
        清理动作缓存：重置所有动作并处理暂停环境的动作
        对暂停环境使用NaN填充动作以保持一致性
        """
        self.current_actions = [None for i in range(self.n_thread)]
        for ith, paused in enumerate(self.team_intel["ENV-PAUSE"]):
            if paused:
                assert self.previous_action[ith] is not None
                self.current_actions[ith] = self.previous_action[ith] + np.nan

    def interact_with_env(self, team_intel):
        """
        主交互接口：接收环境状态并返回动作决策
        @param team_intel: 环境状态信息
        @return: 动作列表和更新后的环境状态
        """
        self.team_intel = team_intel  # 更新环境状态
        self.current_actions = [None] * self.n_thread  # 初始化动作缓存

        self.deal_with_pymarl()  # 处理Pymarl请求

        # 保存历史状态用于后续处理
        self.previous_action = np.array(self.current_actions)
        self.previous_ENV_PAUSE = copy.deepcopy(team_intel["ENV-PAUSE"])
        # 转换动作数组维度 (thread x agent) -> (agent x thread)
        ret_action_list = np.swapaxes(np.array(self.current_actions), 0, 1)
        return ret_action_list, team_intel

    def get_env_with_currentuuid(self):
        """
        UUID到环境映射：实现线程安全的环境标识分配
        @return: 对应的环境索引
        """
        uuid = self.current_uuid
        if uuid not in self.uuid2threads:
            self.uuid2threads[uuid] = self.uuid2threads["thread_cnt"]
            self.uuid2threads["thread_cnt"] += 1
            assert self.uuid2threads["thread_cnt"] <= self.n_thread, (
                "线程访问超限，请检查配置!"
            )
        return self.uuid2threads[uuid]


# PymarlFoundationOld 类定义（兼容旧版Pymarl）
class PymarlFoundationOld:
    def init_pymarl(self):
        """
        初始化旧版Pymarl：使用Redis作为通信中间件
        适用于非高效并行运行器的场景
        """
        assert AlgorithmConfig.pymarl2_runner_select != "efficient_parallel_runner"
        fp = open("%s/pymarl.log" % GlobalConfig.logdir, "w+")
        import atexit
        import uuid

        self.remote_uuid = uuid.uuid1().hex  # 生成唯一进程标识
        # 使用Redis进行进程间通信（需单独安装服务）
        self.redis = redis.Redis(host="127.0.0.1", port=6379)

        # 注入全局配置到pymarl配置系统
        if "config.py->GlobalConfig" not in AlgorithmConfig.pymarl_config_injection:
            AlgorithmConfig.pymarl_config_injection["config.py->GlobalConfig"] = {}

        AlgorithmConfig.pymarl_config_injection["config.py->GlobalConfig"].update(
            {
                "HmpRoot": os.getcwd(),  # 当前工作目录
                "ExpNote": GlobalConfig.note,  # 实验备注
                "draw_mode": GlobalConfig.draw_mode,  # 绘图模式
                "logdir": GlobalConfig.logdir,  # 日志目录
                "n_thread": GlobalConfig.num_threads,  # 线程数
                "seed": GlobalConfig.seed,  # 随机种子
                "activate_logger": GlobalConfig.activate_logger,  # 是否启用日志
                "train_time_testing": GlobalConfig.train_time_testing,  # 训练时测试
                "test_interval": GlobalConfig.test_interval,  # 测试间隔
                "test_only": GlobalConfig.test_only,  # 仅测试模式
                "test_epoch": GlobalConfig.test_epoch,  # 测试轮次
            }
        )

        # 启动Pymarl子进程
        assert os.path.exists("./THIRDPARTY/pymarl2/pymarl2src/main.py")
        subprocess.Popen(
            [
                sys.executable,  # 当前Python解释器路径
                "./THIRDPARTY/pymarl2/pymarl2src/main.py",
                "--force",  # 强制运行
                "--config=%s" % AlgorithmConfig.pymarl2_alg_select,  # 算法配置
                "--env-config=HMP_compat",  # 环境配置
                "--pymarl_config_injection=%s"
                % encrpt_string(
                    json.dumps(AlgorithmConfig.pymarl_config_injection)
                ),  # 注入配置
                "--env_uuid=%s" % self.remote_uuid,  # 环境唯一标识
            ],
            stdout=fp,  # 输出重定向到日志
        )

        # 注册退出清理函数
        atexit.register(lambda: self.__del__())
        time.sleep(5)  # 等待初始化完成

    def __del__(self):
        """
        析构函数：清理Redis资源
        """
        print("PymarlFoundation end, cleaning redis")
        self.redis.delete(">>hmp%s" % self.remote_uuid)
        for uuid, which_env in self.uuid2threads.items():
            self.redis.delete("<<hmp%s" % uuid)

    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        """
        初始化方法（旧版）
        @param n_agent: 智能体数量
        @param n_thread: 线程数量
        @param space: 环境空间定义
        @param mcv: 共享内存变量（未使用）
        @param team: 团队信息（未使用）
        """
        # 基础属性初始化
        self.n_thread = n_thread
        self.n_agent = n_agent
        self.handler = [None for _ in range(self.n_thread)]
        self.remote_pymarl_start_cmd = ""
        self.remote_pymarl_interface = ""
        self.team_intel = None  # 团队智能体信息
        self.space = space  # 环境空间
        self.uuid2threads = {"thread_cnt": 0}  # UUID到线程的映射
        self.current_actions = [None for _ in range(self.n_thread)]  # 当前动作
        self.previous_action = None  # 前一次动作
        self.previous_ENV_PAUSE = None  # 前一次暂停状态
        self.ScenarioConfig = GlobalConfig.ScenarioConfig  # 场景配置
        self.register_step_call = [False for _ in range(self.n_thread)]  # 步进调用标记

        self.init_pymarl()  # 初始化Pymarl连接

        # 外壳环境初始化（如果启用）
        if AlgorithmConfig.use_shell != "":
            if AlgorithmConfig.use_shell == "mini_shell_uhmap":
                from .mini_shell_uhmap import ShellEnv

                self.shell = ShellEnv(self, n_agent, n_thread, space, mcv, team)
            else:
                assert False, "未知的外壳环境类型"

    def basic_io(self):
        """
        基础IO操作：从Redis读取请求并执行
        用于旧版Pymarl通信协议
        """
        _, buf = self.redis.brpop(">>hmp%s" % self.remote_uuid)
        cmd_arg = pickle.loads(buf)
        cmd, args, uuid = cmd_arg
        self.current_uuid = uuid
        res = getattr(self, cmd)(*args)
        if cmd == "step_of":  # 步进操作需要延迟处理
            pass
        elif cmd == "close":  # 关闭请求
            raise ReferenceError
        else:
            self.redis.lpush("<<hmp%s" % uuid, pickle.dumps(res))

    def step_callback_pymarl(self):
        """
        步进回调处理：将步进结果写入Redis
        用于异步返回环境反馈
        """
        for uuid, which_env in self.uuid2threads.items():
            if uuid == "thread_cnt":
                continue
            if not self.register_step_call[which_env]:
                continue
            self.register_step_call[which_env] = False

            # 构建返回结果
            reward = self.team_intel["Latest-Reward"][which_env]
            terminated = self.team_intel["Env-Suffered-Reset"][which_env]
            env_info = self.team_intel["Latest-Team-Info"][which_env].copy()
            env_info["Test-Flag"] = self.team_intel["Test-Flag"]  # 添加测试标记

            # 移除冗余字段
            for key in [
                "obs-echo",
                "state-echo",
                "state",
                "avail-act-echo",
                "avail-act",
            ]:
                if key in env_info:
                    env_info.pop(key)

            res = (reward, terminated, env_info)
            self.redis.lpush("<<hmp%s" % uuid, pickle.dumps(res))

    # @basic_io_call
    def step_of(self, act):
        """
        步进操作（旧版）：记录动作并标记步进完成
        @param act: 执行的动作
        """
        which_env = self.get_env_with_currentuuid()
        self.current_actions[which_env] = act
        self.register_step_call[which_env] = True

    # @basic_io_call
    def get_state_size(self):
        """
        获取状态空间大小（旧版）：逻辑同PymarlFoundation
        """
        if AlgorithmConfig.state_compat == "native":
            try:
                return self.space["obs_space"]["state_shape"]
            except:
                info = self.team_intel["Latest-Team-Info"][0]
                if "state" not in info:
                    return 0
                else:
                    return info["state"].shape[-1]
        else:
            return self.space["obs_space"]["state_shape"]

    # @basic_io_call
    def get_obs_size(self):
        """
        获取观测空间大小（旧版）：逻辑同PymarlFoundation
        """
        try:
            return self.space["obs_space"]["obs_shape"]
        except:
            obs = self.team_intel["Latest-Obs"][0]
            return obs.shape[-1]

    # @basic_io_call
    def get_n_agents(self):
        """
        获取智能体数量（旧版）：逻辑同PymarlFoundation
        """
        try:
            self.n_agent = self.space["act_space"]["n_agents"]
            return self.n_agent
        except:
            return self.n_agent

    # @basic_io_call
    def get_episode_limit(self):
        """
        获取最大步数（旧版）：逻辑同PymarlFoundation
        """
        return int(self.ScenarioConfig.MaxEpisodeStep)

    # @basic_io_call
    def get_total_actions(self):
        """
        获取总动作数（旧版）：逻辑同PymarlFoundation
        """
        try:
            self.n_actions = self.space["act_space"]["n_actions"]
            return self.n_actions
        except:
            assert self.ScenarioConfig.use_simple_action_space
            self.n_actions = self.ScenarioConfig.n_actions
            return self.ScenarioConfig.n_actions

    # @basic_io_call
    def confirm_reset(self):
        """
        确认环境重置（旧版）：逻辑同PymarlFoundation
        """
        return True

    # @basic_io_call
    def close(self):
        """
        关闭环境（旧版）：逻辑同PymarlFoundation
        """
        return

    # @basic_io_call
    def get_stats_of(self):
        """
        获取统计信息（旧版）：添加测试标记后返回
        """
        which_env = self.get_env_with_currentuuid()
        env_info = self.team_intel["Latest-Team-Info"][which_env].copy()
        env_info["Test-Flag"] = self.team_intel["Test-Flag"]
        for key in ["obs-echo", "state-echo", "state", "avail-act-echo", "avail-act"]:
            if key in env_info:
                env_info.pop(key)
        return env_info

    # @basic_io_call
    def get_state_of(self):
        """
        获取状态（旧版）：无参数版本，逻辑同带参数版本
        """
        which_env = self.get_env_with_currentuuid()
        if self.team_intel["Env-Suffered-Reset"][which_env]:
            if not self.team_intel["ENV-PAUSE"][which_env]:
                return self.team_intel["Latest-Team-Info"][which_env]["state"]
            else:
                return self.team_intel["Latest-Team-Info"][which_env]["state-echo"]
        return self.team_intel["Latest-Team-Info"][which_env]["state"]

    # @basic_io_call
    def get_avail_actions_of(self):
        """
        获取可用动作（旧版）：逻辑同PymarlFoundation
        """
        which_env = self.get_env_with_currentuuid()
        if "avail-act" in self.team_intel["Latest-Team-Info"][which_env]:
            return self.team_intel["Latest-Team-Info"][which_env]["avail-act"]
        else:
            return np.ones((self.n_agent, self.n_actions))

    # @basic_io_call
    def get_obs_of(self):
        """
        获取观测值（旧版）：逻辑同PymarlFoundation
        """
        which_env = self.get_env_with_currentuuid()
        if self.team_intel["Env-Suffered-Reset"][which_env]:
            if not self.team_intel["ENV-PAUSE"][which_env]:
                return self.team_intel["Latest-Obs"][which_env]
            else:
                return self.team_intel["Terminal-Obs-Echo"][which_env]
        return self.team_intel["Latest-Obs"][which_env]

    def deal_with_pymarl(self):
        """
        处理Pymarl请求（旧版）：持续处理Redis队列中的请求
        """
        while any([act is None for act in self.current_actions]):
            self.basic_io()

    def clear_actions(self):
        """
        清理动作（旧版）：逻辑同PymarlFoundation
        """
        self.current_actions = [None for i in range(self.n_thread)]
        for ith, paused in enumerate(self.team_intel["ENV-PAUSE"]):
            if paused:
                assert self.previous_action[ith] is not None
                self.current_actions[ith] = self.previous_action[ith] + np.nan

    def interact_with_env(self, team_intel):
        """
        与环境交互（旧版）：逻辑同PymarlFoundation
        """
        self.team_intel = team_intel
        self.step_callback_pymarl()
        assert not any(self.register_step_call)

        self.clear_actions()
        self.deal_with_pymarl()

        self.previous_action = np.array(self.current_actions)
        self.previous_ENV_PAUSE = copy.deepcopy(team_intel["ENV-PAUSE"])
        ret_action_list = np.swapaxes(np.array(self.current_actions), 0, 1)
        return ret_action_list, team_intel

    def get_env_with_currentuuid(self):
        """
        获取环境索引（旧版）：逻辑同PymarlFoundation
        """
        uuid = self.current_uuid
        if uuid not in self.uuid2threads:
            self.uuid2threads[uuid] = self.uuid2threads["thread_cnt"]
            self.uuid2threads["thread_cnt"] += 1
            assert self.uuid2threads["thread_cnt"] <= self.n_thread, (
                "accessed by too many threads, check setting!"
            )
        return self.uuid2threads[uuid]
