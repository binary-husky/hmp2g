# 导入numpy库用于数值计算
import numpy as np

# 从配置模块导入全局配置
from config import GlobalConfig

# 从动作集查找模块导入动作编码函数
from MISSION.uhmap.actset_lookup import encode_action_as_digits

# 导入颜色输出工具
from UTIL.colorful import *

# 导入张量操作工具函数
from UTIL.tensor_ops import gather_righthand, my_view, repeat_at

# 导入基础算法配置
from .foundation import AlgorithmConfig

# 导入异构网络分配模块
from .hete_assignment import select_nets_for_shellenv


# 配置类，控制是否启用可用动作限制
class ShellEnvConfig:
    add_avail_act = False  # 默认不启用


# 动作转换器类（旧版）
class ActionConvertLegacy:
    def __init__(self, SELF_TEAM_ASSUME, OPP_TEAM_ASSUME, OPP_NUM_ASSUME) -> None:
        # 保存队伍标识
        self.SELF_TEAM_ASSUME = SELF_TEAM_ASSUME
        self.OPP_TEAM_ASSUME = OPP_TEAM_ASSUME
        self.OPP_NUM_ASSUME = OPP_NUM_ASSUME

        # 定义动作字典：(主命令, 子命令, 坐标参数, UID, 目标队伍, 索引)
        self.dictionary_args = [
            ("N/A", "N/A", None, None, None, None, None, None),  # 0 空动作
            ("Idle", "DynamicGuard", None, None, None, None, None, None),  # 1 动态警戒
            ("Idle", "StaticAlert", None, None, None, None, None, None),  # 2 静态警戒
            (
                "Idle",
                "AsFarAsPossible",
                None,
                None,
                None,
                None,
                None,
                None,
            ),  # 4 尽可能远
            (
                "Idle",
                "StayWhenTargetInRange",
                None,
                None,
                None,
                None,
                None,
                None,
            ),  # 5 目标范围内停留
            (
                "SpecificMoving",
                "Dir+X",
                None,
                None,
                None,
                None,
                None,
                None,
            ),  # 7 向X正方向移动
            (
                "SpecificMoving",
                "Dir+Y",
                None,
                None,
                None,
                None,
                None,
                None,
            ),  # 8 向Y正方向移动
            (
                "SpecificMoving",
                "Dir-X",
                None,
                None,
                None,
                None,
                None,
                None,
            ),  # 9 向X负方向移动
            (
                "SpecificMoving",
                "Dir-Y",
                None,
                None,
                None,
                None,
                None,
                None,
            ),  # 10 向Y负方向移动
        ]
        # 动态添加针对每个对手的攻击动作
        for i in range(self.OPP_NUM_ASSUME):
            self.dictionary_args.append(
                ("SpecificAttacking", "N/A", None, None, None, None, OPP_TEAM_ASSUME, i)
            )

    def convert_act_arr(self, type, a):
        # UAV支援类型处理
        if type == "RLA_UAV_Support":
            args = self.dictionary_args[a]
            # 覆盖错误的攻击动作
            if args[0] == "SpecificAttacking":
                return encode_action_as_digits(
                    "N/A", "N/A", None, None, None, None, None, None
                )
            # 修正无效的空闲动作
            if args[0] == "Idle":
                return encode_action_as_digits(
                    "Idle", "StaticAlert", None, None, None, None, None, None
                )
            return encode_action_as_digits(*args)
        else:
            return encode_action_as_digits(*self.dictionary_args[a])

    def get_tp_avail_act(self, type):
        DISABLE = 0  # 动作禁用标志
        ENABLE = 1  # 动作启用标志
        n_act = len(self.dictionary_args)
        ret = np.zeros(n_act) + ENABLE  # 默认全部启用

        # 禁用所有巡逻移动动作
        for i in range(n_act):
            args = self.dictionary_args[i]
            if args[0] == "PatrolMoving":
                ret[i] = DISABLE

            # UAV支援类型特殊限制
            if type == "RLA_UAV_Support":
                if args[0] == "PatrolMoving":
                    ret[i] = DISABLE
                if args[0] == "SpecificAttacking":
                    ret[i] = DISABLE
                if args[0] == "Idle":
                    ret[i] = DISABLE
                if args[1] == "StaticAlert":
                    ret[i] = ENABLE  # 启用静态警戒子动作
        return ret

    def confirm_parameters_are_correct(self, team, agent_num, opp_agent_num):
        # 参数校验
        assert team == self.SELF_TEAM_ASSUME  # 队伍标识匹配
        assert self.SELF_TEAM_ASSUME + self.OPP_TEAM_ASSUME == 1  # 队伍总数验证
        assert opp_agent_num == self.OPP_NUM_ASSUME  # 对方代理数量验证


# 统计列表中不同元素的数量
def count_list_type(x):
    type_cnt = {}
    for xx in x:
        if xx not in type_cnt:
            type_cnt[xx] = 0
        type_cnt[xx] += 1
    return len(type_cnt)  # 返回不同类型的数量


# 环境包装器类
class ShellEnvWrapper(object):
    def __init__(
        self,
        n_agent,
        n_thread,
        space,
        mcv,
        rl_functional,
        alg_config,
        ScenarioConfig,
        team,
    ):
        # 基础参数初始化
        self.n_agent = n_agent  # 代理数量
        self.n_thread = n_thread  # 线程数量
        self.team = team  # 队伍标识
        self.space = space  # 空间配置
        self.mcv = mcv  # 多线程控制器
        self.rl_functional = rl_functional  # RL功能模块

        # 根据配置选择观察维度
        if GlobalConfig.ScenarioConfig.EntityOriented:
            self.core_dim = GlobalConfig.ScenarioConfig.obs_vec_length
        else:
            self.core_dim = space["obs_space"]["obs_shape"]

        self.n_entity_placeholder = alg_config.n_entity_placeholder  # 实体占位符数量

        # 可用动作标志初始化
        self.AvailActProvided = False
        if hasattr(ScenarioConfig, "AvailActProvided"):
            self.AvailActProvided = ScenarioConfig.AvailActProvided

        # 创建动作转换器实例
        self.action_converter = ActionConvertLegacy(
            SELF_TEAM_ASSUME=team,
            OPP_TEAM_ASSUME=(1 - team),
            OPP_NUM_ASSUME=GlobalConfig.ScenarioConfig.N_AGENT_EACH_TEAM[1 - team],
        )

        # 异构代理类型处理
        agent_type_list = [
            a["type"] for a in GlobalConfig.ScenarioConfig.SubTaskConfig.agent_list
        ]
        opp_type_list = [
            a["type"]
            for a in GlobalConfig.ScenarioConfig.SubTaskConfig.agent_list
            if a["team"] != self.team
        ]
        self_type_list = [
            a["type"]
            for a in GlobalConfig.ScenarioConfig.SubTaskConfig.agent_list
            if a["team"] == self.team
        ]

        # 类型编码函数
        def str_array_to_num(str_arr):
            out_arr = []
            buffer = {}
            for str in str_arr:
                if str not in buffer:
                    buffer[str] = len(buffer)
                out_arr.append(buffer[str])
            return out_arr

        # 类型编码转换
        self.HeteAgentType = str_array_to_num(agent_type_list)
        self.hete_type = np.array(self.HeteAgentType)[
            GlobalConfig.ScenarioConfig.AGENT_ID_EACH_TEAM[team]
        ]
        self.n_hete_types = count_list_type(self.hete_type)  # 统计类型数量

        # 参数校验
        assert self.n_agent == len(self_type_list)  # 代理数量匹配
        self.action_converter.confirm_parameters_are_correct(
            team, self.n_agent, len(opp_type_list)
        )  # 参数正确性验证

        self.patience = 2000  # 耐心值
        self.epsiode_cnt = 0  # 回合计数器

    def cold_start_warmup(self, StateRecall):
        # 初始化代理UID
        self.agent_uid = GlobalConfig.ScenarioConfig.AGENT_ID_EACH_TEAM[self.team]
        # 获取代理类型
        self.agent_type = [
            agent_meta["type"]
            for agent_meta in StateRecall["Latest-Team-Info"][0]["dataArr"]
            if agent_meta["uId"] in self.agent_uid
        ]
        # 如果启用可用动作限制
        if ShellEnvConfig.add_avail_act:
            # 生成可用动作矩阵
            self.avail_act = np.stack(
                tuple(
                    self.action_converter.get_tp_avail_act(tp) for tp in self.agent_type
                )
            )
            # 扩展线程维度
            self.avail_act = repeat_at(
                self.avail_act, insert_dim=0, n_times=self.n_thread
            )

    def interact_with_env(self, StateRecall):
        # 首次执行时预热
        if not hasattr(self, "agent_type"):
            self.cold_start_warmup(StateRecall)

        # 初始化动作数组为-1
        act = np.zeros(shape=(self.n_thread, self.n_agent), dtype=int) - 1

        # 读取并重塑观测数据
        obs = StateRecall["Latest-Obs"]
        obs = my_view(obs, [0, 0, -1, self.core_dim])
        # 掩码无效观测值
        obs[(obs == 0).all(-1)] = np.nan

        # 环境状态掩码
        P = StateRecall["ENV-PAUSE"]  # 停止掩码
        R = ~P  # 运行掩码
        RST = StateRecall["Env-Suffered-Reset"]  # 重置掩码

        # 如果需要训练
        if not StateRecall["Test-Flag"]:
            self.rl_functional.train()

        # 如果所有环境线程重置
        if RST.all():
            # 快速测试退出条件
            if (
                AlgorithmConfig.allow_fast_test
                and GlobalConfig.test_only
                and (self.epsiode_cnt > GlobalConfig.report_reward_interval)
            ):
                import sys

                sys.exit(0)

            self.epsiode_cnt += self.n_thread  # 更新回合计数

            # 策略共振处理
            eprsn_yita = (
                self.rl_functional.stage_planner.yita
                if AlgorithmConfig.policy_resonance
                else 0
            )
            EpRsn = np.random.rand(self.n_thread) < eprsn_yita
            StateRecall["_EpRsn_"] = EpRsn

            # 异构代理识别
            StateRecall["_hete_type_"] = repeat_at(self.hete_type, 0, self.n_thread)

            # 网络选择
            StateRecall["_hete_pick_"], StateRecall["_gp_pick_"] = (
                select_nets_for_shellenv(
                    n_types=self.n_hete_types,
                    policy=self.rl_functional.policy,
                    hete_type_list=self.hete_type,
                    n_thread=self.n_thread,
                    n_gp=AlgorithmConfig.hete_n_net_placeholder,
                    testing=StateRecall["Test-Flag"],
                )
            )
            # 打印检查点信息
            print(
                [
                    (t["win_rate"], t["ckpg_cnt"])
                    for t in self.rl_functional.policy.ckpg_info
                ]
            )

        # 准备RL算法输入
        I_StateRecall = {
            "obs": obs[R],  # 观测数据
            "avail_act": self.avail_act[R],  # 可用动作
            "Test-Flag": StateRecall["Test-Flag"],  # 测试标志
            "_EpRsn_": StateRecall["_EpRsn_"][R],  # 共振标志
            "_hete_pick_": StateRecall["_hete_pick_"][R],  # 网络选择
            "_hete_type_": StateRecall["_hete_type_"][R],  # 类型信息
            "_gp_pick_": StateRecall["_gp_pick_"][R],  # 组别选择
            "threads_active_flag": R,  # 活动线程
            "Latest-Team-Info": StateRecall["Latest-Team-Info"][R],  # 最新队伍信息
        }

        # 如果提供可用动作
        if self.AvailActProvided:
            avail_act = np.array(
                [
                    info["avail-act"]
                    for info in np.array(
                        StateRecall["Latest-Team-Info"][R], dtype=object
                    )
                ]
            )
            I_StateRecall.update({"avail_act": avail_act})  # 更新可用动作

        # 执行RL算法交互
        act_active, internal_recall = self.rl_functional.interact_with_env_genuine(
            I_StateRecall
        )

        # 获取决策结果
        act[R] = act_active

        # 验证动作有效性
        if ShellEnvConfig.add_avail_act and self.patience > 0:
            self.patience -= 1
            assert (
                gather_righthand(self.avail_act, repeat_at(act, -1, 1), check=False)[R]
                == 1
            ).all()

        # 转换动作格式
        act_converted = np.array(
            [
                [
                    self.action_converter.convert_act_arr(self.agent_type[agentid], act)
                    for agentid, act in enumerate(th)
                ]
                for th in act
            ]
        )

        # 调整维度顺序
        actions_list = (
            act_converted
            if GlobalConfig.mt_act_order == "new_method"
            else np.swapaxes(act_converted, 0, 1)
        )

        # 注册回调钩子
        if not StateRecall["Test-Flag"]:
            StateRecall["_hook_"] = internal_recall["_hook_"]
            assert StateRecall["_hook_"] is not None
        else:
            if AlgorithmConfig.policy_matrix_testing:
                StateRecall["_hook_"] = internal_recall["_hook_"]
                assert StateRecall["_hook_"] is not None

        # 返回动作列表和更新后的状态
        return actions_list, StateRecall
