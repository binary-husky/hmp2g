# 导入必要的库
import importlib  # 用于动态导入模块

import numpy as np  # 数值计算库

from UTIL.data_struct import UniqueList  # 自定义数据结构，用于去重列表


class MMPlatform(object):
    def __init__(self, mcv, envs):
        """初始化多团队强化学习平台"""
        from config import GlobalConfig  # 导入全局配置

        # 从配置中读取团队相关参数
        self.n_t = GlobalConfig.ScenarioConfig.N_TEAM  # 团队总数
        n_agents_each_t = (
            GlobalConfig.ScenarioConfig.N_AGENT_EACH_TEAM
        )  # 每个团队的智能体数量
        self.t_member_list = (
            GlobalConfig.ScenarioConfig.AGENT_ID_EACH_TEAM
        )  # 团队成员ID列表
        self.t_name = GlobalConfig.ScenarioConfig.TEAM_NAMES  # 团队名称列表

        # 验证配置一致性
        assert self.n_t == len(self.t_name), "团队数量与名称数量不匹配"
        assert self.n_t == len(UniqueList(self.t_name)), "团队名称重复"

        # 读取其他配置参数
        self.align_episode = GlobalConfig.align_episode  # 是否对齐episode
        self.n_thread = GlobalConfig.num_threads  # 线程数量
        self.legacy_act_order = True  # 动作顺序标志
        if GlobalConfig.mt_act_order == "new_method":
            self.legacy_act_order = False

        # 初始化状态表示配置
        self.RewardAsUnity = False  # 奖励是否按团队计算
        if hasattr(GlobalConfig.ScenarioConfig, "RewardAsUnity"):
            self.RewardAsUnity = GlobalConfig.ScenarioConfig.RewardAsUnity
        self.ActAsUnity = False  # 动作是否按团队统一
        if hasattr(GlobalConfig.ScenarioConfig, "ActAsUnity"):
            self.ActAsUnity = GlobalConfig.ScenarioConfig.ActAsUnity
        self.ObsAsUnity = False  # 观测是否按团队统一
        if hasattr(GlobalConfig.ScenarioConfig, "ObsAsUnity"):
            self.ObsAsUnity = GlobalConfig.ScenarioConfig.ObsAsUnity

        # 获取环境空间信息
        space = envs.get_space()  # 获取观测空间和动作空间

        # 动态加载并初始化各团队算法
        self.algo_foundations = []
        for t in range(self.n_t):
            # 验证团队成员数量
            assert len(self.t_member_list[t]) == n_agents_each_t[t]
            assert "->" in self.t_name[t]  # 团队名称需包含'->'分隔符

            # 解析模块和类名
            module_, class_ = self.t_name[t].split("->")
            init_f = getattr(importlib.import_module(module_), class_)

            # 初始化算法实例
            self.algo_foundations.append(
                init_f(
                    n_agent=n_agents_each_t[t],
                    n_thread=self.n_thread,
                    space=space,
                    mcv=mcv,
                    team=t,
                )
            )

    def act(self, runner_info: dict):
        """生成各团队动作"""
        actions_list = []
        # 遍历每个团队
        for t_name, t_members, algo_fdn, t_index in zip(
            self.t_name, self.t_member_list, self.algo_foundations, range(self.n_t)
        ):
            # 分割当前团队的信息
            _t_intel_ = self._split_intel(runner_info, t_members, t_name, t_index)
            # 与环境交互生成动作
            _act_, _t_intel_ = algo_fdn.interact_with_env(_t_intel_)
            # 合并动作结果
            actions_list = self._append_act_to_list(_act_, actions_list, t_members)

            # 处理内部状态更新
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

        # 调整动作数组形状
        actions_list = np.swapaxes(np.array(actions_list, dtype=np.double), 0, 1)

        # 处理暂停线程的动作填充
        ENV_PAUSE = runner_info["ENV-PAUSE"]
        if ENV_PAUSE.any() and self.align_episode:
            actions_list[ENV_PAUSE, :] = np.nan
        return actions_list, runner_info

    def _split_intel(
        self, runner_info: dict, t_members: range, t_name: str, t_index: int
    ):
        """分割全局信息到指定团队"""
        # 初始化团队信息
        Team_Info = None
        ter_obs_echo = None

        # 加载团队终止信息
        if runner_info["Latest-Team-Info"] is not None:
            assert isinstance(runner_info["Latest-Team-Info"][0], dict)
            Team_Info = runner_info["Latest-Team-Info"]
            # 处理环境重置后的观测
            ter_obs_echo = np.array(
                [
                    self.__split_obs_thread(Team_Info[thread_idx]["obs-echo"], t_index)
                    if done and ("obs-echo" in Team_Info[thread_idx])
                    else None
                    for thread_idx, done in enumerate(runner_info["Env-Suffered-Reset"])
                ],
                dtype=object,
            )

        # 分割观测和奖励
        o = self.__split_obs(runner_info["Latest-Obs"], t_index)
        reward = runner_info["Latest-Reward"]

        # 构建团队基础信息字典
        t_intel_basic = {
            "Team_Name": t_name,
            "Latest-Obs": o,
            "Latest-Team-Info": Team_Info,
            "Env-Suffered-Reset": runner_info["Env-Suffered-Reset"],
            "Terminal-Obs-Echo": ter_obs_echo,
            "ENV-PAUSE": runner_info["ENV-PAUSE"],
            "Test-Flag": runner_info["Test-Flag"],
            "Latest-Reward": reward[:, t_members]
            if not self.RewardAsUnity
            else reward[:, t_index],
            "Current-Obs-Step": runner_info["Current-Obs-Step"],
        }

        # 处理团队特定键值
        for key in runner_info:
            if t_name not in key:
                continue
            s_key = key.replace(t_name, "")
            t_intel_basic[s_key] = runner_info[key]
            # 处理钩子函数
            if s_key == "_hook_" and t_intel_basic["_hook_"] is not None:
                self.deal_with_hook(t_intel_basic["_hook_"], t_intel_basic)
                runner_info[key] = None
                t_intel_basic.pop("_hook_")

        return t_intel_basic

    def before_terminate(self, runner_info: dict):
        """在平台终止前执行团队信息预处理"""
        # 遍历所有团队
        for t_name, t_members, t_index in zip(
            self.t_name, self.t_member_list, range(self.n_t)
        ):
            # 分割团队相关信息（奖励/观测等）
            # 将全局runner_info分割为各团队专用数据
            self._split_intel(runner_info, t_members, t_name, t_index)

    def _update_runner(self, runner_info: dict, ENV_PAUSE, t_name, key, content):
        """更新runner_info中的团队特定状态"""
        u_key = t_name + key  # 构建带团队名称的键
        # 条件更新逻辑：
        if (
            (u_key in runner_info)  # 键存在
            and hasattr(content, "__len__")  # 内容可长度检查
            and len(content) == self.n_thread  # 内容长度匹配线程数
            and ENV_PAUSE.any()  # 存在线程暂停
        ):
            # 非暂停线程更新内容
            runner_info[u_key][~ENV_PAUSE] = content[~ENV_PAUSE]
            return
        # 全量更新
        runner_info[u_key] = content

    def _append_act_to_list(
        self, _act_: np.ndarray, actions_list: np.ndarray, t_members: range
    ):
        """将团队动作结果合并到全局动作列表"""
        # 处理动作顺序兼容性
        if not self.legacy_act_order:
            _act_ = np.swapaxes(_act_, 0, 1)  # 轴交换

        # 验证动作数量匹配
        assert _act_.shape[0] == len(t_members), (
            "number of actions differs number of agents, Try to switch mt_act_order!"
        )

        # 根据模式选择合并方式
        append_op = actions_list.append if self.ActAsUnity else actions_list.extend
        append_op(_act_)  # 执行合并操作
        return actions_list

    def deal_with_hook(self, hook: callable, t_intel_basic: dict):
        """执行算法注册的回调钩子函数"""
        # 传递关键运行时信息给钩子函数
        hook(
            {
                "reward": t_intel_basic["Latest-Reward"],  # 奖励信息
                "done": t_intel_basic["Env-Suffered-Reset"],  # 终止标志
                "info": t_intel_basic["Latest-Team-Info"],  # 环境信息
                "Latest-Obs": t_intel_basic["Latest-Obs"],  # 观测数据
                "Terminal-Obs-Echo": t_intel_basic["Terminal-Obs-Echo"],  # 终止观测
            }
        )

    def notify_teams(self, message, **kargs):
        """向所有团队发送通知消息"""
        # 遍历所有团队算法
        for t_index, algo_fdn in enumerate(self.algo_foundations):
            # 检查通知接口有效性
            if (not hasattr(algo_fdn, "on_notify")) or (
                not callable(algo_fdn.on_notify)
            ):
                continue

            # 构造团队专属参数
            team_kargs = {k: v[t_index] for k, v in kargs.items()}
            # 发送通知
            algo_fdn.on_notify(message, **team_kargs)

    def __split_obs(self, obs, t_index):
        """按团队分割观测数据（批量处理）"""
        # 输入格式: [n_thread, n_team/n_agent, coredim]
        if obs[0] is None:  # 空观测处理
            o = None
        elif self.ObsAsUnity:  # 团队统一观测模式
            o = obs[:, t_index]  # [n_thread, coredim]
        else:  # 默认智能体级观测模式
            o = obs[:, self.t_member_list[t_index]]  # [n_thread, n_agent, coredim]
        return o

    def __split_obs_thread(self, obs, t_index):
        """按团队分割单线程观测数据"""
        # 输入格式: [n_thread, n_team/n_agent, coredim]
        if self.ObsAsUnity:  # 团队统一观测模式
            o = obs[t_index]  # [coredim]
        else:  # 默认智能体级观测模式
            o = obs[self.t_member_list[t_index]]  # [n_agent, coredim]
        return o
