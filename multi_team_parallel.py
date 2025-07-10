# 导入依赖库
import atexit  # 用于注册程序退出时执行的函数
import copy  # 用于深拷贝对象
import importlib  # 用于动态导入模块

import numpy as np  # 数值计算库

from UTIL.data_struct import UniqueList  # 自定义数据结构
from UTIL.shm_pool import SmartPool  # 共享内存进程池（用于并行计算）


class alg_parallel_wrapper(object):
    """算法并行封装器，包装单个团队的算法实例"""

    def __init__(self, t_name, n_agent, n_thread, space, mcv, team) -> None:
        """初始化算法包装器

        Args:
            t_name: 算法类路径（如'module->Class'）
            n_agent: 代理数量
            n_thread: 线程数量
            space: 环境空间信息
            mcv: 可视化记录器（None时自动初始化）
            team: 团队编号
        """
        self.team = team
        # 初始化可视化记录器
        if mcv is None:
            mcv = self.init_alg_logger()
        # 解析算法模块和类
        module_, class_ = t_name.split("->")
        init_f = getattr(importlib.import_module(module_), class_)
        # 创建算法实例
        self.alg = init_f(n_agent, n_thread, space, mcv, team)
        self._hook_deligate_ = None  # 回调钩子

    def interact_with_env(self, _input_):
        """与环境交互生成动作

        Args:
            _input_: 包含观测值等信息的字典

        Returns:
            动作数组和元信息字典
        """
        # 调用底层算法的交互方法
        _act_, _t_intel_ = self.alg.interact_with_env(_input_)

        # 过滤非内部状态信息（保留_开头的键）
        for k in list(_t_intel_.keys()):
            if not k.startswith("_"):
                _t_intel_.pop(k)

        # 处理回调钩子
        if "_hook_" in _t_intel_ and _t_intel_["_hook_"] is not None:
            self._hook_deligate_ = _t_intel_.pop("_hook_")
            _t_intel_["_hook_"] = "call_hook_deligate"
        return _act_, _t_intel_

    def call_hook_deligate(self, callback_arg):
        """执行回调函数"""
        assert self._hook_deligate_ is not None
        self._hook_deligate_(callback_arg)  # 执行回调
        self._hook_deligate_ = None  # 重置钩子

    def notify_teams(self, message, kargs):
        """向团队发送通知"""
        if (not hasattr(self.alg, "on_notify")) or (not callable(self.alg.on_notify)):
            return
        self.alg.on_notify(message, **kargs)  # 调用算法的on_notify方法

    def init_alg_logger(self):
        """初始化可视化记录器"""
        from config import GlobalConfig as cfg
        from VISUALIZE.mcom import mcom

        logdir = cfg.logdir
        if cfg.activate_logger:
            mcv = mcom(
                path=f"{logdir}/logger/{self.team}/",
                image_path=f"{logdir}/team-{self.team}.jpg",
                rapid_flush=True,
                draw_mode=cfg.draw_mode,
                tag="[multi_team_parallel.py]",
                resume_mod=cfg.resume_mod,
            )
        mcv.rec_init(color="k")  # 初始化记录器
        return mcv


class MMPlatform(object):
    """多团队并行计算平台"""

    def __init__(self, mcv, envs):
        """初始化平台

        Args:
            mcv: 可视化记录器
            envs: 环境对象
        """
        from config import GlobalConfig

        # 读取场景配置
        self.n_t = GlobalConfig.ScenarioConfig.N_TEAM  # 团队数量
        n_agents_each_t = GlobalConfig.ScenarioConfig.N_AGENT_EACH_TEAM  # 各团队代理数
        self.t_member_list = (
            GlobalConfig.ScenarioConfig.AGENT_ID_EACH_TEAM
        )  # 团队成员ID
        self.t_name = GlobalConfig.ScenarioConfig.TEAM_NAMES  # 团队算法名称

        # 验证配置一致性
        assert self.n_t == len(self.t_name), "团队数量与算法名称数量不匹配"
        assert self.n_t == len(UniqueList(self.t_name)), "团队名称重复"

        # 读取运行时配置
        self.align_episode = GlobalConfig.align_episode  # 是否对齐episode
        self.n_thread = GlobalConfig.num_threads  # 线程数
        # 动作顺序兼容性配置
        self.legacy_act_order = True
        if GlobalConfig.mt_act_order == "new_method":
            self.legacy_act_order = False

        # 观测/奖励/动作空间配置
        self.RewardAsUnity = False
        if hasattr(GlobalConfig.ScenarioConfig, "RewardAsUnity"):
            self.RewardAsUnity = GlobalConfig.ScenarioConfig.RewardAsUnity
        self.ActAsUnity = False
        if hasattr(GlobalConfig.ScenarioConfig, "ActAsUnity"):
            self.ActAsUnity = GlobalConfig.ScenarioConfig.ActAsUnity
        self.ObsAsUnity = False
        if hasattr(GlobalConfig.ScenarioConfig, "ObsAsUnity"):
            self.ObsAsUnity = GlobalConfig.ScenarioConfig.ObsAsUnity

        # 获取环境空间信息
        space = envs.get_space()

        # 构建算法参数列表
        arg_list = []
        for t in range(self.n_t):
            assert len(self.t_member_list[t]) == n_agents_each_t[t]
            assert "->" in self.t_name[t]
            arg_list.append(
                (
                    self.t_name[t],  # 算法类路径
                    n_agents_each_t[t],  # 代理数量
                    self.n_thread,  # 线程数
                    space,  # 空间信息
                    None,  # 记录器（后续自动初始化）
                    t,  # 团队编号
                )
            )

        # 初始化并行执行器
        print("[multi_team_parallel] distributing algorithm to independent process")
        self.alg_parallel_exe = SmartPool(
            fold=1,
            proc_num=self.n_t,
            base_seed=GlobalConfig.seed,  # 固定随机种子
        )
        # 注册进程终止回调
        atexit.register(self.alg_parallel_exe.party_over)

        # 添加算法目标到执行器
        self.alg_parallel_exe.add_target(
            name="alg_parallel_exe",
            lam=alg_parallel_wrapper,  # 目标类
            args_list=arg_list,  # 参数列表
        )
        print("[multi_team_parallel] distribution is done")

    def act(self, runner_info):
        """生成动作

        Args:
            runner_info: 运行时信息字典

        Returns:
            动作数组和更新后的runner_info
        """
        actions_list = []  # 存储动作结果

        # 准备各团队输入数据
        _t_intel_feed_list_ = []
        for t_name, t_members, t_index in zip(
            self.t_name, self.t_member_list, range(self.n_t)
        ):
            # 分割观测/奖励等信息
            _t_intel_ = self._split_intel(runner_info, t_members, t_name, t_index)
            _t_intel_feed_list_.append(_t_intel_)

        # 并行执行各团队算法
        results = self.alg_parallel_exe.exec_target(
            name="alg_parallel_exe",
            dowhat="interact_with_env",
            args_list=_t_intel_feed_list_,
            ensure_safe=True,
        )

        # 解析结果
        _act_mt_, _t_intel_mt_ = zip(*results)

        # 处理各团队结果
        for t_name, t_members, _act_, _t_intel_, t_index in zip(
            self.t_name, self.t_member_list, _act_mt_, _t_intel_mt_, range(self.n_t)
        ):
            # 拼接动作
            actions_list = self._append_act_to_list(_act_, actions_list, t_members)

            # 处理内部状态回传
            if _t_intel_ is None:
                continue
            for key in _t_intel_:
                if key.startswith("_") and key.endswith("_"):
                    self._update_runner(
                        runner_info,
                        runner_info["ENV-PAUSE"],
                        t_name,
                        key,
                        _t_intel_[key],
                    )

        # 调整动作数组维度
        actions_list = np.swapaxes(np.array(actions_list, dtype=np.double), 0, 1)

        # 处理暂停线程的动作
        ENV_PAUSE = runner_info["ENV-PAUSE"]
        if ENV_PAUSE.any() and self.align_episode:
            actions_list[ENV_PAUSE, :] = np.nan
        return actions_list, runner_info

    def before_terminate(self, runner_info):
        """终止前处理，目前仅做数据分割预处理"""
        # 遍历所有团队
        for t_name, t_members, t_index in zip(
            self.t_name, self.t_member_list, range(self.n_t)
        ):
            # 分割观测/奖励等信息（当前未实际使用）
            self._split_intel(runner_info, t_members, t_name, t_index)

    def _update_runner(self, runner_info, ENV_PAUSE, t_name, key, content):
        """更新运行时信息字典

        Args:
            runner_info: 运行时信息字典
            ENV_PAUSE: 环境暂停标志数组
            t_name: 团队名称
            key: 要更新的键
            content: 新内容
        """
        u_key = t_name + key  # 拼接团队名称和键名
        # 如果内容满足长度要求且存在暂停线程
        if (
            (u_key in runner_info)
            and hasattr(content, "__len__")
            and len(content) == self.n_thread
            and ENV_PAUSE.any()
        ):
            # 仅更新非暂停线程的内容
            runner_info[u_key][~ENV_PAUSE] = content[~ENV_PAUSE]
            return
        # 否则直接赋值
        runner_info[u_key] = content
        return

    def _split_intel(self, runner_info, t_members, t_name, t_index):
        """分割团队相关数据（观测、奖励等）

        Args:
            runner_info: 运行时信息
            t_members: 团队成员ID列表
            t_name: 团队名称
            t_index: 团队索引

        Returns:
            包含团队专用数据的字典
        """
        Team_Info = None
        ter_obs_echo = None

        # 如果存在团队信息
        if runner_info["Latest-Team-Info"] is not None:
            assert isinstance(runner_info["Latest-Team-Info"][0], dict)
            Team_Info = runner_info["Latest-Team-Info"]
            # 构造终止观测回声数组
            ter_obs_echo = np.array([None for _ in range(self.n_thread)], dtype=object)
            for thread_idx, done in enumerate(runner_info["Env-Suffered-Reset"]):
                if done and ("obs-echo" in Team_Info[thread_idx]):
                    # 分割单个线程的观测回声
                    ter_obs_echo[thread_idx] = self.__split_obs_thread(
                        Team_Info[thread_idx]["obs-echo"], t_index
                    )

        # 深拷贝团队信息并移除obs-echo字段
        Team_Info_Downstream = copy.deepcopy(Team_Info)
        for i in range(len(Team_Info_Downstream)):
            if "obs-echo" in Team_Info_Downstream[i]:
                Team_Info_Downstream[i].pop("obs-echo")

        # 分割观测数据
        o = self.__split_obs(runner_info["Latest-Obs"], t_index)
        # 获取奖励数据
        reward = runner_info["Latest-Reward"]

        # 构造返回字典
        t_intel_basic = {
            "Team_Name": t_name,
            "Latest-Obs": o,
            "Latest-Team-Info": Team_Info_Downstream,
            "Env-Suffered-Reset": runner_info["Env-Suffered-Reset"],
            "Terminal-Obs-Echo": ter_obs_echo,
            "ENV-PAUSE": runner_info["ENV-PAUSE"],
            "Test-Flag": runner_info["Test-Flag"],
            # 根据配置分割奖励数据
            "Latest-Reward": reward[:, t_members]
            if not self.RewardAsUnity
            else reward[:, t_index],
            "Current-Obs-Step": runner_info["Current-Obs-Step"],
        }

        # 处理回调钩子
        key = f"{t_name}_hook_"
        if (key in runner_info) and (runner_info[key] is not None):
            t_intel_basic["_hook_"] = runner_info[key]
            self.deal_with_hook(t_intel_basic["_hook_"], t_intel_basic, t_index)
            runner_info[key] = None
            t_intel_basic["_hook_"] = None
            t_intel_basic.pop("_hook_")

        return t_intel_basic

    def _append_act_to_list(self, _act_, actions_list, t_members):
        """将动作添加到结果列表

        Args:
            _act_: 输入动作数组
            actions_list: 动作列表
            t_members: 团队成员列表

        Returns:
            更新后的动作列表
        """
        if not self.legacy_act_order:
            # 调整维度顺序
            _act_ = np.swapaxes(_act_, 0, 1)
        # 验证动作数量匹配
        assert _act_.shape[0] == len(t_members), "动作数量与代理数量不匹配!"
        # 根据配置选择拼接方式
        append_op = actions_list.append if self.ActAsUnity else actions_list.extend
        append_op(_act_)
        return actions_list

    def deal_with_hook(self, hook, t_intel_basic, t_index):
        """处理算法回调钩子

        Args:
            hook: 回调对象
            t_intel_basic: 团队数据字典
            t_index: 团队索引
        """
        # 构造回调参数
        arg = {
            "reward": t_intel_basic["Latest-Reward"],
            "done": t_intel_basic["Env-Suffered-Reset"],
            "info": t_intel_basic["Latest-Team-Info"],
            "Latest-Obs": t_intel_basic["Latest-Obs"],
            "Terminal-Obs-Echo": t_intel_basic["Terminal-Obs-Echo"],
        }
        # 根据钩子类型执行不同操作
        if hook == "call_hook_deligate":
            # 通过进程池执行回调
            self.alg_parallel_exe.exec_target(
                name="alg_parallel_exe",
                dowhat="call_hook_deligate",
                args_list=[arg],
                index_list=[t_index],
                ensure_safe=True,
            )
        else:
            # 直接调用回调函数
            hook(arg)

    def notify_teams(self, message, **kargs):
        """向所有团队发送通知

        Args:
            message: 通知消息
            **kargs: 附加参数
        """
        # 构造参数列表（每个团队都使用相同参数）
        args_list = [(message, kargs)] * self.n_t
        # 并行执行通知
        self.alg_parallel_exe.exec_target(
            name="alg_parallel_exe",
            dowhat="notify_teams",
            args_list=args_list,
            ensure_safe=True,
        )

    def __split_obs(self, obs, t_index):
        """分割观测数据（批量处理）

        Args:
            obs: 原始观测数据 [n_thread, n_team/n_agent, coredim]
            t_index: 团队索引

        Returns:
            分割后的观测数据
        """
        if obs[0] is None:
            return None
        elif self.ObsAsUnity:
            # 按团队分割
            return obs[:, t_index]
        else:  # 默认按成员分割
            return obs[:, self.t_member_list[t_index]]

    def __split_obs_thread(self, obs, t_index):
        """分割单个线程的观测数据

        Args:
            obs: 原始观测数据 [n_thread, n_team/n_agent, coredim]
            t_index: 团队索引

        Returns:
            分割后的观测数据
        """
        if self.ObsAsUnity:
            # 按团队分割
            return obs[t_index]
        else:  # 默认按成员分割
            return obs[self.t_member_list[t_index]]
