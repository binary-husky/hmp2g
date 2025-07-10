# 导入必要的库
import importlib  # 动态导入模块

import numpy as np

from ALGORITHM.common.norm import DynamicNormFix  # 动态归一化层
from config import GlobalConfig  # 全局配置参数
from UTIL.colorful import *  # 彩色输出工具
from UTIL.tensor_ops import my_view, repeat_at  # 张量操作工具

from .pymarl2_compat import AlgorithmConfig  # 算法配置


class ShellEnv(object):
    """
    强化学习环境外壳，负责：
    - 观测/状态处理
    - 动作转换
    - 归一化处理
    - 与底层RL基础架构交互
    """

    def __init__(self, rl_foundation, n_agent, n_thread, space, mcv, team):
        """
        初始化环境外壳

        参数:
            rl_foundation: RL基础架构对象
            n_agent: 智能体数量
            n_thread: 线程数量
            space: 空间参数（未使用）
            mcv: 多线程上下文变量（未使用）
            team: 团队ID（0或1）
        """
        # 基础属性初始化
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.team = team
        self.rl_foundation = rl_foundation

        # 从全局配置获取观测参数
        self.n_entity = GlobalConfig.ScenarioConfig.obs_n_entity  # 实体数量
        self.obssize = GlobalConfig.ScenarioConfig.obs_vec_length  # 单个实体观测维度

        # 根据配置确定状态空间大小
        if AlgorithmConfig.state_compat == "pad":
            # 填充模式：固定大小4
            self.state_size = 4
        elif AlgorithmConfig.state_compat == "obs_cat":
            # 拼接模式：所有实体观测拼接
            self.state_size = self.n_entity * self.obssize * self.n_agent
        elif AlgorithmConfig.state_compat == "obs_mean":
            # 平均模式：实体观测平均
            self.state_size = self.n_entity * self.obssize
        else:
            assert False, "compat method error"

        # 初始化动作转换器
        # 通过配置字符串动态导入类
        module_, class_ = AlgorithmConfig.action_converter.split("->")
        init_f = getattr(importlib.import_module(module_), class_)
        self.action_converter = init_f(
            SELF_TEAM_ASSUME=team,  # 当前团队ID
            OPP_TEAM_ASSUME=(1 - team),  # 对方团队ID
            # 对方团队智能体数量
            OPP_NUM_ASSUME=GlobalConfig.ScenarioConfig.N_AGENT_EACH_TEAM[1 - team],
        )

        # 设置RL基础架构的空间参数
        self.rl_foundation.space = {
            "act_space": {
                "n_actions": len(self.action_converter.dictionary_args),  # 动作数量
            },
            "obs_space": {
                "obs_shape": self.n_entity * self.obssize,  # 观测形状
                "state_shape": self.state_size,  # 状态形状
            },
        }

        # 保存原始交互方法并替换为当前方法
        self.rl_foundation.interact_with_env_real = self.rl_foundation.interact_with_env
        self.rl_foundation.interact_with_env = self.interact_with_env
        self.patience = 2000  # 耐心参数（未使用？）

        # 获取对方团队智能体类型列表
        opp_type_list = [
            a["type"]
            for a in GlobalConfig.ScenarioConfig.SubTaskConfig.agent_list
            if a["team"] != self.team
        ]
        # 验证动作转换器参数是否正确
        self.action_converter.confirm_parameters_are_correct(
            team, self.n_agent, len(opp_type_list)
        )

        # 根据配置决定是否启用归一化
        if AlgorithmConfig.use_shell_normalization:
            # 启用动态归一化
            self.obs_norm = DynamicNormFix(
                input_size=self.n_entity * self.obssize,  # 输入维度
                only_for_last_dim=True,  # 仅归一化最后一维
                exclude_nan=True,  # 排除NaN值
            ).to(GlobalConfig.device)  # 移动到指定设备

            # 状态归一化器
            self.state_norm = DynamicNormFix(
                input_size=self.state_size, only_for_last_dim=True, exclude_nan=True
            ).to(GlobalConfig.device)

            # 归一化函数指针
            self.obs_norm_fn = self.obs_norm.np_forward
            self.state_norm_fn = self.state_norm.np_forward
        else:
            # 禁用归一化时使用恒等函数
            self.obs_norm_fn = self.state_norm_fn = lambda x: x

    def interact_with_env(self, StateRecall):
        """
        与环境交互的核心方法

        参数:
            StateRecall: 状态回忆对象（包含最新观测等信息）

        返回:
            act_converted: 转换后的动作
            team_intel: 团队情报信息
        """
        # 第一次调用时初始化agent_type等属性
        if not hasattr(self, "agent_type"):
            # 获取团队智能体UID
            self.agent_uid = GlobalConfig.ScenarioConfig.AGENT_ID_EACH_TEAM[self.team]

            # 从最新团队信息中提取智能体类型
            self.agent_type = [
                agent_meta["type"]
                for agent_meta in StateRecall["Latest-Team-Info"][0]["dataArr"]
                if agent_meta["uId"] in self.agent_uid
            ]

            # 为每个智能体类型获取可用动作掩码
            self.avail_act = np.stack(
                tuple(
                    self.action_converter.get_tp_avail_act(tp) for tp in self.agent_type
                )
            )

            # 扩展维度以适配线程数
            self.avail_act = repeat_at(
                self.avail_act, insert_dim=0, n_times=self.n_thread
            )

        # 观测数据处理
        StateRecall["Latest-Obs"] = np.nan_to_num(
            self.obs_norm_fn(my_view(StateRecall["Latest-Obs"], [0, 0, -1])), 0
        )

        # 终止观测回声处理
        StateRecall["Terminal-Obs-Echo"] = [
            np.nan_to_num(my_view(np.array(t, dtype=float), [0, -1]), 0)
            if t is not None
            else None
            for t in StateRecall["Terminal-Obs-Echo"]
        ]

        # 状态构建
        for i, d in enumerate(StateRecall["Latest-Team-Info"]):
            if AlgorithmConfig.state_compat == "pad":
                # 填充模式：初始化零状态
                d["state"] = np.zeros(self.state_size)
                d["state-echo"] = np.zeros(self.state_size)

            elif AlgorithmConfig.state_compat == "obs_cat":
                # 拼接模式：观测向量展平后归一化
                d["state"] = self.state_norm_fn(StateRecall["Latest-Obs"][i].flatten())
                d["state-echo"] = np.zeros_like(d["state"])

                # 如果存在终止观测回声
                if StateRecall["Terminal-Obs-Echo"][i] is not None:
                    d["state-echo"] = self.state_norm_fn(
                        StateRecall["Terminal-Obs-Echo"][i].flatten()
                    )

            elif AlgorithmConfig.state_compat == "obs_mean":
                # 平均模式：观测向量平均后归一化
                d["state"] = self.state_norm_fn(StateRecall["Latest-Obs"][i].mean(0))
                d["state-echo"] = np.zeros_like(d["state"])

                # 如果存在终止观测回声
                if StateRecall["Terminal-Obs-Echo"][i] is not None:
                    d["state-echo"] = self.state_norm_fn(
                        StateRecall["Terminal-Obs-Echo"][i].mean(0)
                    )
            else:
                assert False, "compat method error"

            # 添加可用动作掩码
            d["avail-act"] = self.avail_act[i]

        # 调用底层RL框架的交互方法
        ret_action_list, team_intel = self.rl_foundation.interact_with_env_real(
            StateRecall
        )
        # 调整动作列表维度顺序
        ret_action_list = np.swapaxes(ret_action_list, 0, 1)

        # 处理环境暂停标志
        R = ~StateRecall["ENV-PAUSE"]

        # 转换动作格式
        act_converted = np.array(
            [
                [
                    # 如果不是NaN，转换动作
                    self.action_converter.convert_act_arr(
                        self.agent_type[agentid], int(act)
                    )
                    if not np.isnan(act)
                    # 如果是NaN，转换0动作并保留NaN
                    else self.action_converter.convert_act_arr(
                        self.agent_type[agentid], 0
                    )
                    + np.nan
                    for agentid, act in enumerate(th)
                ]
                for th in ret_action_list
            ]
        )

        # 如果不是新动作顺序方法，调整维度
        if GlobalConfig.mt_act_order != "new_method":
            act_converted = np.swapaxes(act_converted, 0, 1)

        return act_converted, team_intel
