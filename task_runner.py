"""
Author: Fu Qingxu,CASIA
Description: HMP task runner, coordinates environments and algorithms
Notes before you read code:
In general, HMP task runner can operate two ways:
    self.align_episode = False: threads immediately restart at terminal state, threads do not wait each other
    self.align_episode = True: threads pause at terminal state, waiting until all threads terminate, then reset
"""

import os
import time

import numpy as np

from config import GlobalConfig as cfg  # 全局配置参数
from MISSION.env_router import make_parallel_envs  # 并行环境创建函数
from UTIL.colorful import *  # 彩色打印工具


class Runner(object):
    def __init__(self, process_pool):
        """
        Runner类构造函数，初始化强化学习任务的核心组件

        参数:
        process_pool: 进程池对象，用于管理并行环境
        """

        # 保存进程池引用
        self.process_pool = process_pool

        # 创建并行环境（调用MISSION.env_router.make_parallel_envs）
        # 支持多线程/进程环境隔离与同步
        self.envs = make_parallel_envs(process_pool)

        # 初始化日志记录器
        # 使用config.GlobalConfig中的note字段作为日志标识
        self.mcv = self.get_a_logger(cfg.note)

        # 动态导入多团队决策模块
        # 根据配置选择并行决策或单线程决策
        if cfg.mt_parallel:
            from multi_team_parallel import MMPlatform  # 并行决策模块
        else:
            from multi_team import MMPlatform  # 单线程决策模块

        # 实例化平台控制器
        # 参数说明:
        #   - mcv: 日志记录器
        #   - envs: 环境接口
        # 功能:
        #   1. 管理团队间信息访问
        #   2. 控制决策流程
        self.platform_controller = MMPlatform(self.mcv, self.envs)

        # 运行时状态容器，存储实时数据：
        # - 观测数据
        # - 奖励记录
        # - 环境状态
        # - 其他元信息
        self.info_runner = {}

        # 计算智能体总数：
        # sum(cfg.ScenarioConfig.N_AGENT_EACH_TEAM)
        # 示例：[5,5] -> 10个智能体
        self.n_agent = sum(cfg.ScenarioConfig.N_AGENT_EACH_TEAM)

        # 团队总数：len(N_AGENT_EACH_TEAM)
        # 示例：[5,5] -> 2个团队
        self.n_team = len(cfg.ScenarioConfig.N_AGENT_EACH_TEAM)

        # 奖励模式设置：
        # False=按智能体奖励，True=按团队奖励
        self.RewardAsUnity = False
        if hasattr(cfg.ScenarioConfig, "RewardAsUnity"):
            self.RewardAsUnity = cfg.ScenarioConfig.RewardAsUnity

        # 测试环境休眠控制：
        # 当不进行测试时释放资源
        self.test_env_sleepy = False
        if hasattr(cfg.ScenarioConfig, "CanTurnOff"):
            self.test_env_sleepy = cfg.ScenarioConfig.CanTurnOff

        # 基础配置参数
        self.n_thread = cfg.num_threads  # 并行线程数
        self.n_frame = cfg.n_parallel_frame  # 每帧迭代次数
        self.note = cfg.note  # 实验备注信息

        # 心跳动画开关：
        # 仅在终端环境中显示
        self.hb_on = cfg.heartbeat_on and stdout.isatty()

        # 迭代计数器
        self.current_n_frame = 0  # 当前帧数
        self.current_n_episode = 0  # 当前episode数
        self.max_n_episode = cfg.max_n_episode  # 最大episode数

        # 测试相关配置
        self.train_time_testing = cfg.train_time_testing  # 训练期间测试开关
        self.test_interval = cfg.test_interval  # 测试间隔
        self.test_only = cfg.test_only  # 是否仅测试模式

        # 线程同步模式：
        # True=所有线程完成episode后统一重置
        # False=线程独立运行
        self.align_episode = cfg.align_episode

        # 提前终止标志
        self._exit_early_ = False

        # 初始化感兴趣智能体日志记录
        self._init_interested_agent_logging()

        self.start_time = None

    # -------------------------------------------------------------------------
    # ------------------------------ Major Loop -------------------------------
    # -------------------------------------------------------------------------
    def run(self):
        # 初始化运行时状态信息，包含对环境的重置
        self.init_runner()

        # 记录性能测试起始时间
        tic = time.time()

        # 记录程序起始时间
        self.start_time = time.time()

        # 主循环：按帧数(n_frame)控制训练迭代
        for cnt in range(self.n_frame):
            # 1. 获取动作：调用算法控制器生成动作列表
            # 参数说明:
            # - self.info_runner: 包含最新观测/奖励/环境状态的运行时数据
            # 返回值:
            # - actions_list: 各线程的动作指令
            # - self.info_runner: 更新后的状态信息
            actions_list, self.info_runner = self.platform_controller.act(
                self.info_runner
            )

            # 2. 执行环境步进：
            # 参数说明:
            # - actions_list: 从算法控制器获取的动作指令
            # 返回值:
            # - obs: 观测数据 (n_thread × n_agent × ...)
            # - reward: 奖励值 (n_thread × n_team/n_agent)
            # - done: 终止标志 (n_thread × bool)
            # - info: 环境信息 (dict)
            obs, reward, done, info = self.envs.step(actions_list)

            # 3. 更新运行时状态：
            # 处理观测/奖励统计、环境重置、测试会话触发等逻辑
            self.info_runner = self.update_runner(done, obs, reward, info)

            # 4. 性能监控：
            # - 计算帧率
            # - 打印心跳动画
            # - 检查提前终止条件
            toc = time.time()
            dt = toc - tic
            tic = toc

            if self.hb_on:
                print(
                    "\r [task runner]: FPS %d, episode steping %s       "
                    % (self.get_fps(dt), self.heartbeat()),
                    end="",
                    flush=True,
                )

            if self._exit_early_:
                print("exit_early")
                break

        # 训练循环结束
        return

    def init_runner(self):
        """初始化运行时状态字典"""
        # 测试模式标志
        self.info_runner["Test-Flag"] = self.test_only

        # 奖励统计缓冲区
        self.info_runner["Recent-Reward-Sum"] = []  # 近期奖励总和
        self.info_runner["Recent-Win"] = []  # 近期胜利次数
        self.info_runner["Recent-Team-Ranking"] = []  # 团队排名记录

        # 环境初始化
        obs_info = self.envs.reset()  # 重置所有环境线程
        # 解析观测数据：处理可能的元组返回值
        if isinstance(obs_info, tuple):
            self.info_runner["Latest-Obs"], self.info_runner["Latest-Team-Info"] = (
                obs_info
            )
        else:
            self.info_runner["Latest-Obs"] = obs_info
            self.info_runner["Latest-Team-Info"] = None

        # 环境状态跟踪
        self.info_runner["Env-Suffered-Reset"] = np.array(
            [True for _ in range(self.n_thread)]
        )  # 环境重置标志
        self.info_runner["ENV-PAUSE"] = np.array(
            [False for _ in range(self.n_thread)]
        )  # 环境暂停状态
        self.info_runner["Current-Obs-Step"] = np.array(
            [0 for _ in range(self.n_thread)]
        )  # 步数计数器

        # 奖励追踪
        self.info_runner["Latest-Reward"] = np.zeros(
            shape=(self.n_thread, self.n_agent)
        )  # 即时奖励
        self.info_runner["Latest-Reward-Sum"] = np.zeros(
            shape=(self.n_thread, self.n_agent)
        )  # 累积奖励

        # 任务统计
        self.info_runner["Thread-Episode-Cnt"] = np.array(
            [0 for _ in range(self.n_thread)]
        )  # 线程任务计数

        # 团队胜率统计初始化
        for i in range(self.n_team):
            self.info_runner[f"t{i}_win_cnt_avg"] = []

        # 支持团队级奖励模式
        if self.RewardAsUnity:
            self.info_runner["Latest-Reward"] = np.zeros(
                shape=(self.n_thread, self.n_team)
            )  # 团队级奖励
            self.info_runner["Latest-Reward-Sum"] = np.zeros(
                shape=(self.n_thread, self.n_team)
            )  # 累积团队奖励

        return

    def update_runner(self, done, obs, reward, info):
        """
        更新运行时状态的核心方法，处理以下关键任务：
        1. 状态同步：更新观测/奖励/步数等实时数据
        2. 环境生命周期管理：处理环境重置与线程暂停
        3. 奖励统计：维护近期奖励/胜率等指标
        4. 测试会话控制：触发周期性测试流程
        5. 终止条件检查：判断是否达到最大训练轮次

        参数:
        done: 环境终止标志 (n_thread × bool)
        obs: 观测数据 (n_thread × n_agent × ...)
        reward: 即时奖励 (n_thread × n_team/n_agent)
        info: 环境附加信息 (dict)
        """

        # 获取当前线程暂停状态 (True表示暂停)
        P = self.info_runner["ENV-PAUSE"]
        # 生成运行线程掩码 (True表示正在运行)
        R = ~P

        # 验证info非空
        assert info is not None

        # 初始化团队信息（仅首次调用）
        if self.info_runner["Latest-Team-Info"] is None:
            self.info_runner["Latest-Team-Info"] = info

        # 更新实时观测数据（仅运行中的线程）
        self.info_runner["Latest-Obs"][R] = obs[R]
        self.info_runner["Latest-Team-Info"][R] = info[R]

        # 更新即时奖励与累计奖励
        self.info_runner["Latest-Reward"][R] = reward[R]
        self.info_runner["Latest-Reward-Sum"][R] += reward[R]

        # 更新步数计数器
        self.info_runner["Current-Obs-Step"][R] += 1

        # 逐线程处理环境重置逻辑
        for i in range(self.n_thread):
            # 记录环境是否完成一次完整episode
            self.info_runner["Env-Suffered-Reset"][i] = done[i].all()

            # 如果线程未重置或处于暂停状态，跳过处理
            if P[i] or (not self.info_runner["Env-Suffered-Reset"][i]):
                continue

            # 当前线程完成一个episode，执行以下操作：

            # 1. 更新总帧数统计
            self.current_n_frame += self.info_runner["Current-Obs-Step"][i]

            # 2. 更新总episode计数
            self.current_n_episode += 1

            # 3. 记录本次episode的累计奖励
            self.info_runner["Recent-Reward-Sum"].append(
                self.info_runner["Latest-Reward-Sum"][i].copy()
            )

            # 4. 提取终止状态信息
            term_info = self.info_runner["Latest-Team-Info"][i]

            # 5. 更新胜利/排名统计
            win = 1 if "win" in term_info and term_info["win"] == True else 0
            self.info_runner["Recent-Win"].append(win)

            # 6. 多团队场景记录排名信息
            if "team_ranking" in term_info:
                self.info_runner["Recent-Team-Ranking"].append(
                    term_info["team_ranking"].copy()
                )

            # 7. 重置线程级统计
            self.info_runner["Latest-Reward-Sum"][i] = 0
            self.info_runner["Current-Obs-Step"][i] = 0
            self.info_runner["Thread-Episode-Cnt"][i] += 1

            # 8. 对齐模式下暂停已完成线程
            if self.align_episode:
                self.info_runner["ENV-PAUSE"][i] = True

            # 9. 定期监控特定智能体
            if self.current_n_episode % self.report_interval == 0:
                self._checkout_interested_agents(self.info_runner)
                self.info_runner["Recent-Reward-Sum"] = []
                self.info_runner["Recent-Win"] = []
                self.info_runner["Recent-Team-Ranking"] = []

            # 10. 触发测试会话
            if (
                self.train_time_testing
                and (not self.test_only)
                and (self.current_n_episode % self.test_interval == 0)
            ):
                self.platform_controller.before_terminate(self.info_runner)
                self.start_a_test_run()

                # 预估剩余时间
                remaining_time = self.estimate_remaining_time()
                print(remaining_time)

        # 所有线程完成当前episode后重启
        if self.align_episode and self.info_runner["ENV-PAUSE"].all():
            self.info_runner["ENV-PAUSE"][:] = False

        # 检查最大episode限制
        if self.current_n_episode >= self.max_n_episode:
            self._exit_early_ = True

        return self.info_runner

    def estimate_remaining_time(self):
        """
        基于当前进度预估剩余运行时间

        返回:
        str: 预计结束时间字符串，格式为YYYY-MM-DD HH:MM:SS
        或 None: 若无法预估
        """
        import time
        from datetime import datetime, timedelta

        if self.current_n_episode == 0:
            return "无法预估 - 尚未完成任何回合"

        # 计算已用时间（秒）
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # 计算每回合平均耗时
        avg_time_per_episode = elapsed_time / self.current_n_episode

        # 计算剩余回合数
        remaining_episodes = self.max_n_episode - self.current_n_episode
        if remaining_episodes <= 0:
            return (
                "任务已完成"
                if self.current_n_episode >= self.max_n_episode
                else "即将完成"
            )

        # 预估剩余时间
        estimated_seconds = remaining_episodes * avg_time_per_episode
        estimated_end_time = datetime.now() + timedelta(seconds=estimated_seconds)
        estimated_end = estimated_end_time.strftime("%Y-%m-%d %H:%M:%S")

        minutes = int(estimated_seconds // 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 24)
        remaining_time = (
            f"{days}天{hours}小时{minutes}分" if days else f"{hours}小时{minutes}分"
        )

        progress = self.current_n_episode / self.max_n_episode * 100

        return f"进度: {progress:.1f}% | 剩余时间：{remaining_time} | 预计结束: {estimated_end}"

    # ------------------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------ About TEST RUN routine, almost a Mirror of above ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------------------
    # -- I know these code below might merge with above for simplicity --
    # -- But I decide not, in order to make it easier to read and debug --
    if cfg.train_time_testing:

        def start_a_test_run(self):
            # 打印测试开始提示信息（靛色高亮）
            print靛("\r[task runner]: test run started!")

            # 初始化测试运行器相关状态
            self.init_test_runner()
            # loop until all env is done

            # 确保测试总轮次数能被线程数整除，保证每个线程执行相同数量的测试回合
            assert cfg.test_epoch % self.n_thread == 0, (
                "please set test_epoch as (n_thread * N)!"
            )
            # assert self.n_thread%cfg.test_epoch == 0, ('please set test_epoch as (n_thread * N)!')
            # 计算每个线程需要执行的测试回合数
            num_runs = cfg.test_epoch // self.n_thread

            # 打印总测试回合数信息
            print靛(
                "\r[task runner]: test run is going to run %d episode" % cfg.test_epoch
            )

            # 开始测试循环
            while True:
                # 通过平台控制器获取各线程的动作列表，并更新测试运行器状态
                actions_list, self.test_info_runner = self.platform_controller.act(
                    self.test_info_runner
                )

                # 执行环境步进，获取观测数据、奖励、完成状态和附加信息
                obs, reward, done, info = self.test_envs.step(actions_list)

                # 更新测试运行器状态
                self.test_info_runner = self.update_test_runner(done, obs, reward, info)

                # 如果启用心跳监测，打印当前观测步数
                if self.hb_on:
                    print(
                        "\r [task runner]: testing %s  "
                        % self.heartbeat(
                            style=3, beat=self.test_info_runner["Current-Obs-Step"]
                        ),
                        end="",
                        flush=True,
                    )

                # 检查所有线程是否完成指定测试回合
                if (self.test_info_runner["Thread-Episode-Cnt"] >= num_runs).all():
                    # 获取各智能体的胜率和奖励
                    t_win_rates, t_rewards = self._checkout_interested_agents(
                        self.test_info_runner, testing=True
                    )

                    # 通知平台控制器测试结束
                    self.platform_controller.before_terminate(self.test_info_runner)

                    # 通知各团队测试结果（胜率和平均奖励）
                    self.platform_controller.notify_teams(
                        "test done", win_rate=t_win_rates, mean_reward=t_rewards
                    )

                    # 如果启用睡眠模式，关闭测试环境
                    if self.test_env_sleepy:
                        self.test_envs.sleep()

                    # 退出测试循环
                    return

        def init_test_runner(self):
            """
            # 核心功能 初始化测试运行器状态
            # 安全机制:
            - 惰性初始化测试环境
            - 状态变量完整性校验
            # 设计意图: 与训练模式分离的独立测试环境管理
            """
            if not hasattr(self, "test_envs"):  # 惰性初始化测试环境
                self.test_envs = make_parallel_envs(  # 创建并行测试环境
                    self.process_pool,
                    marker="test",  # 标记为测试环境
                )  # 平行环境

            # 初始化测试状态字典
            self.test_info_runner = {}  # 存储测试时的实时观测/奖励等信息

            # 测试环境暂停状态数组（False=运行中）
            self.test_info_runner["ENV-PAUSE"] = np.array(
                [False for _ in range(self.n_thread)]  # 初始所有线程运行
            )
            self.test_info_runner["Test-Flag"] = True
            self.test_info_runner["Recent-Win"] = []
            self.test_info_runner["Recent-Reward-Sum"] = []
            self.test_info_runner["Recent-Team-Ranking"] = []
            test_obs_info = (
                self.test_envs.reset()
            )  # assume only the first time reset is manual
            (
                self.test_info_runner["Latest-Obs"],
                self.test_info_runner["Latest-Team-Info"],
            ) = (
                test_obs_info
                if isinstance(test_obs_info, tuple)
                else (test_obs_info, None)
            )
            self.test_info_runner["Env-Suffered-Reset"] = np.array(
                [True for _ in range(self.n_thread)]  # 初始全部标记为已重置
            )

            # 初始化奖励统计容器
            self.test_info_runner["Latest-Reward"] = np.zeros(  # 即时奖励
                shape=(self.n_thread, self.n_agent)  # 线程×智能体量
            )
            self.test_info_runner["Latest-Reward-Sum"] = np.zeros(  # 累积奖励
                shape=(self.n_thread, self.n_agent)  # 线程×智能体量
            )

            # 步数计数器初始化
            self.test_info_runner["Current-Obs-Step"] = np.array(
                [0 for _ in range(self.n_thread)]  # 初始步数为0
            )

            # 线程任务计数器初始化
            self.test_info_runner["Thread-Episode-Cnt"] = np.array(
                [0 for _ in range(self.n_thread)]  # 初始任务数为0
            )

            # 支持团队级奖励模式
            if self.RewardAsUnity:  # 如果启用团队奖励模式
                self.test_info_runner["Latest-Reward"] = np.zeros(  # 团队奖励
                    shape=(self.n_thread, self.n_team)  # 线程×团队量
                )
                self.test_info_runner["Latest-Reward-Sum"] = np.zeros(  # 累积团队奖励
                    shape=(self.n_thread, self.n_team)  # 线程×团队量
                )
            return

        def update_test_runner(self, done, obs, reward, info):
            """
            # 核心功能 测试模式运行时状态更新
            # 处理流程:
            1. 状态同步更新
            2. 环境生命周期管理
            3. 奖励统计维护
            4. 测试终止条件判断
            """
            # 获取当前线程暂停状态 (True=暂停)
            P = self.test_info_runner["ENV-PAUSE"]
            # 生成运行线程掩码 (True=运行中)
            R = ~P

            # 安全检查：确保环境信息非空
            assert info is not None

            # 初始化团队信息（仅首次调用）
            if self.test_info_runner["Latest-Team-Info"] is None:
                self.test_info_runner["Latest-Team-Info"] = info

            # 更新实时观测数据（仅运行中的线程）
            self.test_info_runner["Latest-Obs"][R] = obs[R]  # 存储最新观测
            self.test_info_runner["Latest-Team-Info"][R] = info[R]  # 存储团队信息

            # 更新即时奖励与累计奖励
            self.test_info_runner["Latest-Reward"][R] = reward[R]  # 即时奖励
            self.test_info_runner["Latest-Reward-Sum"][R] += reward[R]  # 累积奖励

            # 更新步数计数器
            self.test_info_runner["Current-Obs-Step"][R] += 1  # 运行线程步数递增

            # 逐线程处理环境重置逻辑
            for i in range(self.n_thread):  # 遍历所有线程
                # 记录环境是否完成一次完整episode
                self.test_info_runner["Env-Suffered-Reset"][i] = done[
                    i
                ].all()  # 全部智能体完成

                # 如果线程未重置或处于暂停状态，跳过处理
                if P[i] or (not self.test_info_runner["Env-Suffered-Reset"][i]):
                    continue  # 跳过未完成线程

                # 当前线程完成一个episode，执行以下操作：

                # 1. 记录本次episode的累计奖励
                self.test_info_runner["Recent-Reward-Sum"].append(  # 存入奖励记录
                    self.test_info_runner["Latest-Reward-Sum"][
                        i
                    ].copy()  # 深拷贝防止覆盖
                )

                # 2. 重置线程级奖励统计
                self.test_info_runner["Latest-Reward-Sum"][i] = 0  # 累计奖励清零
                self.test_info_runner["Current-Obs-Step"][i] = 0  # 步数计数器清零
                self.test_info_runner["Thread-Episode-Cnt"][i] += 1  # episode计数递增

                # 3. 提取终止状态信息
                term_info = self.test_info_runner["Latest-Team-Info"][i]  # 终止信息

                # 4. 记录胜利状态（单胜利条件）
                win = 1 if "win" in term_info and term_info["win"] == True else 0
                self.test_info_runner["Recent-Win"].append(win)  # 存入胜利记录

                # 5. 多团队场景记录排名信息
                if "team_ranking" in term_info:  # 如果存在排名信息
                    self.test_info_runner["Recent-Team-Ranking"].append(  # 保存排名
                        term_info["team_ranking"].copy()  # 深拷贝防止后续修改影响
                    )

                # 6. 对齐模式下暂停已完成线程
                if self.align_episode:  # 如果启用同步模式
                    self.test_info_runner["ENV-PAUSE"][i] = True  # 暂停当前线程

            # 所有线程完成当前episode后重启
            if (
                self.align_episode and self.test_info_runner["ENV-PAUSE"].all()
            ):  # 全部暂停
                self.test_info_runner["ENV-PAUSE"][:] = False  # 解除所有暂停

            return self.test_info_runner  # 返回更新后的状态字典

    def _init_interested_agent_logging(self):
        """
        # 核心功能 初始化感兴趣智能体日志配置
        # 设计意图: 为特定智能体/团队建立独立的性能监控
        """
        # 记录奖励间隔（设计意图：控制日志记录频率）
        self.report_interval = cfg.report_reward_interval

        # 感兴趣智能体ID列表（设计意图：用于监控特定智能体）
        self.interested_agents_uid = cfg.interested_agent_uid

        # 感兴趣团队索引（设计意图：定位需重点监控的团队）
        self.interested_team = cfg.interested_team

        # 历史最高奖励记录（设计意图：性能基准追踪）
        self.top_rewards = None

        # 测试阶段历史最高奖励记录（设计意图：测试性能对比）
        self.test_top_rewards = None

        return

    def _checkout_interested_agents(self, info_runner, testing=False):
        """
        # 核心功能 智能体性能评估与日志记录
        # 处理流程:
        1. 计算平均奖励
        2. 更新历史最高奖励
        3. 统计胜率/排名
        4. 输出可视化结果
        """
        # (1) 记录时间戳（非测试模式）
        if not testing:
            self.mcv.rec(self.current_n_episode, "time")  # 时间序列记录

        # 构建日志前缀（测试/训练区分）
        prefix = "test " if testing else ""

        # 提取近期奖励数据
        recent_rewards = np.stack(info_runner["Recent-Reward-Sum"])  # 堆叠奖励数据

        # 初始化团队奖励列表
        mean_reward_each_team = []

        # 团队级奖励统计
        if self.RewardAsUnity:
            for interested_team in range(self.n_team):
                # 计算团队平均奖励
                mean_reward_each_team.append(
                    recent_rewards[:, interested_team].mean().copy()
                )
        else:
            for interested_team in range(self.n_team):
                # 获取团队成员ID并计算奖励
                tean_agent_uid = cfg.ScenarioConfig.AGENT_ID_EACH_TEAM[interested_team]
                mean_reward_each_team.append(
                    recent_rewards[:, tean_agent_uid].mean().copy()  # 单智能体奖励均值
                )

        # 记录各团队平均奖励
        for team in range(self.n_team):
            self.mcv.rec(mean_reward_each_team[team], f"{prefix}reward of=team-{team}")

        # (2) 更新历史最高奖励
        if not testing:
            if self.top_rewards is None:
                self.top_rewards = mean_reward_each_team  # 初始化历史记录
            top_rewards_list_pointer = self.top_rewards
        else:
            if self.test_top_rewards is None:
                self.test_top_rewards = mean_reward_each_team  # 测试历史记录初始化
            top_rewards_list_pointer = self.test_top_rewards

        # 更新历史最高值并记录
        for team in range(self.n_team):
            if mean_reward_each_team[team] > top_rewards_list_pointer[team]:
                top_rewards_list_pointer[team] = mean_reward_each_team[team]
            self.mcv.rec(
                top_rewards_list_pointer[team], f"{prefix}top reward of=team-{team}"
            )

        # (3) 胜率统计
        teams_ranking = info_runner["Recent-Team-Ranking"]
        win_rate_each_team = [0] * self.n_team

        if len(teams_ranking) > 0:
            for team in range(self.n_team):
                # 提取当前团队排名数据
                rank_itr_team = np.array(teams_ranking)[:, team]
                # 计算排名首位比例（设计意图：多团队场景胜率定义）
                win_rate = (rank_itr_team == 0).mean()
                win_rate_each_team[team] = win_rate
                # 记录团队排名胜率
                self.mcv.rec(win_rate, f"{prefix}top-rank ratio of=team-{team}")
        else:
            team = 0
            assert self.n_team == 1, "There is only one team"  # 安全检查
            # 单团队场景直接统计胜利次数
            win_rate_each_team[team] = np.array(info_runner["Recent-Win"]).mean()
            win_rate = np.array(info_runner["Recent-Win"]).mean()
            # 记录单团队胜利率
            self.mcv.rec(win_rate, f"{prefix}win rate of=team-{team}")

        # 更新累计胜率统计（非测试模式）
        if not testing:
            for i in range(self.n_team):
                # 追加最新胜率
                self.info_runner[f"t{i}_win_cnt_avg"].append(win_rate_each_team[i])
                # 计算移动平均
                ti = np.array(self.info_runner[f"t{i}_win_cnt_avg"]).mean()
                # 记录移动平均胜率
                self.mcv.rec(ti, f"{prefix}acc win ratio of=team-{i}")

        # 可视化数据展示
        self.mcv.rec_show()

        # 构建输出信息
        if testing:
            print_info = [
                f"\r[task runner]: Test result at episode {self.current_n_episode}."
            ]
        else:
            print_info = [
                f"\r[task runner]: ({self.note}) Finished episode {self.current_n_episode}, frame {self.current_n_frame}."
            ]
        # 添加各团队统计信息
        for team in range(self.n_team):
            print_info.append(
                " | team-%d: win rate: %.3f, recent reward %.3f"
                % (team, win_rate_each_team[team], mean_reward_each_team[team])
            )
        # 靛色输出最终信息
        print靛("".join(print_info))

        return win_rate_each_team, mean_reward_each_team  # 返回统计结果

    def conclude_experiment(self):
        """
        # 核心功能 实验数据持久化存储
        # 处理流程:
        1. 获取实验数据
        2. 序列化保存到指定路径
        """
        result = self.mcv.rec_get()  # 获取所有记录数据
        logdir = cfg.logdir  # 日志目录路径

        def objdump(obj):
            import pickle  # 导入序列化模块

            # 构建输出文件路径
            with open(f"{logdir}/experiment_conclusion.pkl", "wb+") as f:
                pickle.dump(obj, f)  # 执行序列化操作
            return

        objdump(result)  # 执行数据保存
        return

    @staticmethod
    def get_a_logger(note):
        """
        # 核心功能 创建可视化日志记录器
        # 参数说明:
        - note: 日志标识符
        # 返回值: mcom日志对象
        """
        from VISUALIZE.mcom import mcom  # 导入可视化模块

        logdir = cfg.logdir  # 获取日志目录
        if cfg.activate_logger:
            # 创建日志记录器实例
            mcv = mcom(
                path="%s/logger/" % logdir,  # 日志存储路径
                digit=16,  # 浮点数精度
                rapid_flush=True,  # 快速刷新缓冲区
                draw_mode=cfg.draw_mode,  # 绘图模式
                tag="[task_runner.py]",  # 日志标签
                resume_mod=cfg.resume_mod,  # 断点续训模式
            )
            cfg.data_logger = mcv  # 全局引用
        mcv.rec_init(color="b")  # 初始化蓝色记录器
        return mcv  # 返回日志对象

    def heartbeat(self, style=0, beat=None):
        """
        # 核心功能 生成终端心跳动画
        # 支持4种样式：
        0: 点阵
        1: 方块旋转
        2: 柱状图
        3: 花朵图标
        """
        width = os.get_terminal_size().columns  # 获取终端宽度

        # 定义不同样式符号集
        if style == 0:
            sym = ["⠁", "⠈", "⠐", "⠠", "⢀", "⡀", "⠄", "⠂"]  # 点阵样式
        elif style == 1:
            sym = ["◐ ", "◓ ", "◑ ", "◒ "]  # 方块旋转
        elif style == 2:
            sym = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]  # 柱状图
        elif style == 3:
            sym = ["💐", "🌷", "🌸", "🌹", "🌺", "🌻", "🌼"]  # 花朵样式

        # 处理心跳数据源
        if beat is None:
            beat = self.info_runner["Current-Obs-Step"]  # 默认使用当前步数
        beat = beat % len(sym)  # 循环索引
        beat = beat[: int(width * 0.2)]  # 限制最大长度
        beat.astype(int)  # 确保整数索引
        beat = [sym[t] for t in beat]  # 映射到符号
        return "".join(beat)  # 返回拼接的动画字符串

    def get_fps(self, dt):
        """
        # 核心功能 计算帧率（每秒帧数）
        # 参数说明:
        - dt: 时间差（秒）
        # 返回值: 平滑后的帧率
        """
        # 计算瞬时帧率
        new_fps = int(self.n_thread / dt)

        # 初始化平滑值或更新移动平均
        if not hasattr(self, "fps_smooth"):
            self.fps_smooth = new_fps  # 首次赋值
        else:
            # 指数加权移动平均（98%旧值 + 2%新值）
            self.fps_smooth = self.fps_smooth * 0.98 + new_fps * 0.02
        return int(self.fps_smooth)  # 返回整数帧率
