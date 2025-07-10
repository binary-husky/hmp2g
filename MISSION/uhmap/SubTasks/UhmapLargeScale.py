# 导入必要的库和模块
import copy
import inspect  # 用于获取对象的源代码信息
import json
import os

import numpy as np  # 数值计算库

# 自定义模块导入
from UTIL.tensor_ops import my_view, repeat_at  # 张量操作工具函数

from ...common.base_env import RawObsArray  # 原始观测数组容器
from ..actset_lookup import AgentPropertyDefaults  # 代理属性默认值
from ..agent import Agent  # 代理类定义
from ..uhmap_env_wrapper import ScenarioConfig, UhmapEnv  # 场景配置和基础环境类
from .SubtaskCommonFn import UhmapCommonFn  # 子任务通用功能
from .UhmapLargeScaleConf import SubTaskConfig  # 当前子任务的配置类


class UhmapLargeScale(UhmapCommonFn, UhmapEnv):
    def __init__(self, rank) -> None:
        """
        初始化方法
        Args:
            rank: 环境实例的排序标识
        """
        super().__init__(rank)
        # 获取观测空间维度
        self.observation_space = self.make_obs(get_shape=True)
        # 绑定子任务配置
        self.SubTaskConfig = SubTaskConfig
        # 验证配置文件匹配性
        assert (
            os.path.basename(inspect.getfile(SubTaskConfig))
            == type(self).__name__ + "Conf.py"
        ), "make sure you have imported the correct SubTaskConfig class"

    def reset(self):
        """
        环境重置方法
        在每个episode开始时初始化所有智能体
        Returns:
            观测数据
        """
        UhmapEnv.reset(self)
        self.t = 0  # 时间步计数器
        # 随机生成初始旋转角度（弧度制）
        pos_ro = np.random.rand() * 2 * np.pi * 0.25

        # 初始化代理配置数组
        AgentSettingArray = []

        # 统计各团队代理数量
        n_team_agent = {}
        for i, agent_info in enumerate(self.SubTaskConfig.agent_list):
            team = agent_info["team"]
            if team not in n_team_agent:
                n_team_agent[team] = 0
            # 分配唯一ID和团队内ID
            self.SubTaskConfig.agent_list[i]["uid"] = i
            self.SubTaskConfig.agent_list[i]["tid"] = n_team_agent[team]
            n_team_agent[team] += 1

        self.n_team_agent = n_team_agent  # 保存团队代理数量统计

        # 生成所有代理的初始配置
        for i, agent_info in enumerate(self.SubTaskConfig.agent_list):
            team = agent_info["team"]
            agent_info["n_team_agent"] = n_team_agent[team]
            # 动态获取初始化函数
            init_fn = getattr(self, agent_info["init_fn_name"])
            AgentSettingArray.append(init_fn(agent_info, pos_ro))

        # 创建代理实例
        self.agents = [
            Agent(team=a["team"], team_id=a["tid"], uid=a["uid"])
            for a in self.SubTaskConfig.agent_list
        ]

        # 发送重置命令到Unreal Engine
        resp = self.client.send_and_wait_reply(
            json.dumps(
                {
                    "valid": True,
                    "DataCmd": "reset",
                    "NumAgents": len(self.SubTaskConfig.agent_list),
                    "AgentSettingArray": AgentSettingArray,
                    "TimeStepMax": ScenarioConfig.MaxEpisodeStep,
                    "TimeStep": 0,
                    "Actions": None,
                }
            )
        )
        resp = json.loads(resp)

        # 验证代理生成结果
        assert len(resp["dataArr"]) == len(AgentSettingArray), (
            "Illegal agent initial position. 非法的智能体初始化位置，一部分智能体没有生成."
        )
        return self.parse_response_ob_info(resp)

    def extract_key_gameobj(self, resp):
        """
        提取关键游戏对象信息
        Args:
            resp: UE返回的响应数据
        Returns:
            关键对象数组
        """
        keyObjArr = resp["dataGlobal"]["keyObjArr"]
        return keyObjArr

    def gen_reward_and_win(self, resp):
        """
        生成奖励和胜负判定
        Args:
            resp: UE返回的响应数据
        Returns:
            (奖励列表, 胜负结果)
        """
        reward = [0] * self.n_teams
        events = resp["dataGlobal"]["events"]
        WinningResult = None

        for event in events:
            event_parsed = self.parse_event(event)

            # 处理摧毁事件
            if event_parsed["Event"] == "Destroyed":
                team = self.find_agent_by_uid(event_parsed["UID"]).team
                reward[team] -= 0.005  # 本队损失
                reward[1 - team] += 0.010  # 敌方收益

            # 处理回合结束事件
            if event_parsed["Event"] == "EndEpisode":
                EndReason = event_parsed["EndReason"]
                WinTeam = int(event_parsed["WinTeam"])

                if WinTeam < 0:  # 超时结束
                    # 计算存活代理数量
                    agents_left_each_team = [0 for _ in range(self.n_teams)]
                    for a in self.agents:
                        if a.alive:
                            agents_left_each_team[a.team] += 1

                    WinTeam = np.argmax(agents_left_each_team)

                    # 处理平局情况
                    if (
                        agents_left_each_team[WinTeam]
                        == agents_left_each_team[1 - WinTeam]
                    ):
                        # 比较HP总和
                        hp_each_team = [0 for _ in range(self.n_teams)]
                        for a in self.agents:
                            if a.alive:
                                hp_each_team[a.team] += a.hp
                        WinTeam = np.argmax(hp_each_team)

                        # 再次平局则标记为-1
                        if hp_each_team[WinTeam] == hp_each_team[1 - WinTeam]:
                            WinTeam = -1

                # 生成奖励
                if WinTeam >= 0:
                    WinningResult = {
                        "team_ranking": [0, 1] if WinTeam == 0 else [1, 0],
                        "end_reason": EndReason,
                    }
                    reward[WinTeam] += 5
                else:
                    WinningResult = {"team_ranking": [-1, -1], "end_reason": EndReason}

        return reward, WinningResult

    @staticmethod
    def item_random_mv(src, dst, prob, rand=False):
        """
        随机移动数组元素
        Args:
            src: 源数组
            dst: 目标数组
            prob: 移动概率
            rand: 是否随机打乱
        Returns:
            (修改后的src, 修改后的dst)
        """
        assert len(src.shape) == 1
        assert len(dst.shape) == 1
        if rand:
            np.random.shuffle(src)
        len_src = len(src)
        n_mv = (np.random.rand(len_src) < prob).sum()
        # 移动元素索引
        item_mv = src[range(len_src - n_mv, len_src)]
        src = src[range(0, 0 + len_src - n_mv)]
        dst = np.concatenate((item_mv, dst))
        return src, dst

    @staticmethod
    def get_binary_array(n_int, n_bits=8, dtype=np.float32):
        """
        将整数转换为二进制数组
        Args:
            n_int: 输入整数数组
            n_bits: 二进制位数
            dtype: 输出数据类型
        Returns:
            二进制数组
        """
        arr = np.zeros((*n_int.shape, n_bits), dtype=dtype)
        for i in range(n_bits):
            arr[:, i] = (n_int % 2 == 1).astype(int)
            n_int = n_int / 2
        return arr

    def make_obs(self, resp=None, get_shape=False):
        """
        生成观测数据
        Args:
            resp: UE响应数据
            get_shape: 是否仅获取维度
        Returns:
            观测数组
        """
        CORE_DIM = 23  # 核心特征维度
        assert ScenarioConfig.obs_vec_length == CORE_DIM
        if get_shape:
            return CORE_DIM

        # 配置参数
        OBS_RANGE_PYTHON_SIDE = SubTaskConfig.OBS_RANGE_PYTHON_SIDE
        MAX_NUM_OPP_OBS = SubTaskConfig.MAX_NUM_OPP_OBS
        MAX_NUM_ALL_OBS = SubTaskConfig.MAX_NUM_ALL_OBS
        MAX_OBJ_NUM_ACCEPT = SubTaskConfig.MAX_OBJ_NUM_ACCEPT

        # 获取位置矩阵
        pos3d_arr = np.zeros(shape=(self.n_agents, 3), dtype=np.float32)
        for i, agent in enumerate(self.agents):
            pos3d_arr[i] = agent.pos3d
        dis_mat = resp["dataGlobal"]["distanceMat"]  # 距离矩阵
        alive_all = np.array([agent.alive for agent in self.agents])  # 存活状态

        # 团队归属信息
        team_belonging = np.array([agent.team for agent in self.agents])

        # 构建原始观测数据
        obs_arr = RawObsArray(key="Agent")

        if not hasattr(self, "uid_binary"):
            self.uid_binary = self.get_binary_array(
                np.arange(max(self.n_agents, len(self.key_obj))), 10
            )

        # 收集代理基础观测
        for i, agent in enumerate(self.agents):
            obs_arr.append(self.uid_binary[i])  # UID编码
            obs_arr.append([agent.index, agent.team, agent.alive, agent.uid_remote])
            obs_arr.append(agent.pos3d)  # 位置信息
            obs_arr.append(agent.vel3d)  # 速度信息
            obs_arr.append([agent.hp, agent.yaw, agent.max_speed])  # 状态信息

        obs_ = obs_arr.get()
        new_obs = my_view(obs_, [self.n_agents, -1])  # 重塑观测维度

        # 构建完整的观测矩阵
        OBS_ALL_AGENTS = np.zeros(
            shape=(self.n_agents, MAX_NUM_OPP_OBS + MAX_NUM_ALL_OBS, CORE_DIM)
        )

        # 为每个代理构建观测
        for i, agent in enumerate(self.agents):
            if not agent.alive:
                OBS_ALL_AGENTS[i, :] = np.nan
                continue

            dis2all = dis_mat[i, :]  # 到所有代理的距离
            is_ally = team_belonging == agent.team  # 盟友判断

            # 敌方观测处理
            a2h_dis = dis2all[~is_ally]
            h_alive = alive_all[~is_ally]
            h_feature = new_obs[~is_ally]
            h_iden_sort = np.argsort(a2h_dis)[:MAX_NUM_OPP_OBS]
            a2h_dis_sorted = a2h_dis[h_iden_sort]
            h_alive_sorted = h_alive[h_iden_sort]
            h_vis_mask = (a2h_dis_sorted <= OBS_RANGE_PYTHON_SIDE) & h_alive_sorted

            # 敌方观测索引处理
            h_vis_index = h_iden_sort[h_vis_mask]
            h_invis_index = h_iden_sort[~h_vis_mask]
            h_vis_index, h_invis_index = self.item_random_mv(
                src=h_vis_index, dst=h_invis_index, prob=0, rand=True
            )
            h_ind = np.concatenate((h_vis_index, h_invis_index))
            h_msk = np.concatenate((h_vis_index < 0, h_invis_index >= 0))
            a2h_feature_sort = h_feature[h_ind]
            a2h_feature_sort[h_msk] = 0

            # 补足观测长度
            if len(a2h_feature_sort) < MAX_NUM_OPP_OBS:
                a2h_feature_sort = np.concatenate(
                    (
                        a2h_feature_sort,
                        np.ones(
                            shape=(MAX_NUM_OPP_OBS - len(a2h_feature_sort), CORE_DIM)
                        )
                        + np.nan,
                    ),
                    axis=0,
                )

            # 盟友观测处理
            a2f_dis = dis2all[is_ally]
            f_alive = alive_all[is_ally]
            f_feature = new_obs[is_ally]
            f_iden_sort = np.argsort(a2f_dis)[:MAX_NUM_ALL_OBS]
            a2f_dis_sorted = a2f_dis[f_iden_sort]
            f_alive_sorted = f_alive[f_iden_sort]
            f_vis_mask = (a2f_dis_sorted <= OBS_RANGE_PYTHON_SIDE) & f_alive_sorted

            # 盟友索引处理
            f_vis_index = f_iden_sort[f_vis_mask]
            self_vis_index = f_vis_index[:1]
            f_vis_index = f_vis_index[1:]
            f_invis_index = f_iden_sort[~f_vis_mask]
            f_vis_index, f_invis_index = self.item_random_mv(
                src=f_vis_index, dst=f_invis_index, prob=0, rand=True
            )
            f_ind = np.concatenate((self_vis_index, f_vis_index, f_invis_index))
            f_msk = np.concatenate(
                (self_vis_index < 0, f_vis_index < 0, f_invis_index >= 0)
            )
            self_ally_feature_sort = f_feature[f_ind]
            self_ally_feature_sort[f_msk] = 0

            # 补足观测长度
            if len(self_ally_feature_sort) < MAX_NUM_ALL_OBS:
                self_ally_feature_sort = np.concatenate(
                    (
                        self_ally_feature_sort,
                        np.ones(
                            shape=(
                                MAX_NUM_ALL_OBS - len(self_ally_feature_sort),
                                CORE_DIM,
                            )
                        )
                        + np.nan,
                    ),
                    axis=0,
                )

            # 合并观测数据
            OBS_ALL_AGENTS[i, :] = np.concatenate(
                (self_ally_feature_sort, a2h_feature_sort), axis=0
            )

        # 处理游戏对象观测
        self.N_Obj = len(self.key_obj)
        OBJ_UID_OFFSET = 32768

        obs_arr = RawObsArray(key="GameObj")

        for i, obj in enumerate(self.key_obj):
            obs_arr.append(-self.uid_binary[i])  # 反转UID编码
            obs_arr.append(
                [
                    obj["uId"] - OBJ_UID_OFFSET,
                    -1,
                    True,
                    obj["uId"] - OBJ_UID_OFFSET,
                ]
            )
            obs_arr.append(
                [
                    obj["location"]["x"],
                    obj["location"]["y"],
                    obj["location"]["z"],
                ]
            )
            obs_arr.append(
                [
                    obj["velocity"]["x"],
                    obj["velocity"]["y"],
                    obj["velocity"]["z"],
                    -1,
                    obj["rotation"]["yaw"],
                    0,
                ]
            )

        # 游戏对象观测后处理
        OBS_GameObj = my_view(obs_arr.get(), [len(self.key_obj), -1])[
            :MAX_OBJ_NUM_ACCEPT, :
        ]
        OBS_GameObj = repeat_at(OBS_GameObj, insert_dim=0, n_times=self.n_agents)
        OBS_ALL_AGENTS = np.concatenate((OBS_ALL_AGENTS, OBS_GameObj), axis=1)

        return OBS_ALL_AGENTS

    def init_ground(self, agent_info, pos_ro):
        """
        地面单位初始化
        Args:
            agent_info: 代理信息字典
            pos_ro: 初始旋转角度
        Returns:
            代理属性字典
        """
        # 解析代理信息
        agent_class = agent_info["type"]
        team = agent_info["team"]
        n_team_agent = agent_info["n_team_agent"]
        tid = agent_info["tid"]
        uid = agent_info["uid"]
        N_COL = 2
        if n_team_agent > 40:
            N_COL = 3
        if n_team_agent > 60:
            N_COL = 7
        x = 0 + 800 * (tid - n_team_agent // 2) // N_COL
        y = (600 * (tid % N_COL) + 2000) * (-1) ** (team + 1)
        x, y = np.matmul(
            np.array([x, y]),
            np.array(
                [[np.cos(pos_ro), -np.sin(pos_ro)], [np.sin(pos_ro), np.cos(pos_ro)]]
            ),
        )
        z = 500  # 离地高度
        yaw = 90 if team == 0 else -90  # 初始朝向

        # 基础属性设置
        agent_property = copy.deepcopy(AgentPropertyDefaults)
        agent_property.update(
            {
                "DebugAgent": False,
                "MaxMoveSpeed": 500 if agent_class == "RLA_CAR_Laser" else 700,
                "AgentScale": {"x": 0.6, "y": 0.6, "z": 0.6}
                if agent_class == "RLA_CAR_Laser"
                else {"x": 0.8, "y": 0.8, "z": 0.8},
                "DodgeProb": 0.00,
                "ExplodeDmg": 40,
                "AgentTeam": team,
                "ClassName": agent_class,
                "WeaponCD": 1,
                "PerceptionRange": 3000,
                "GuardRange": 1100 if agent_class == "RLA_CAR_Laser" else 2100,
                "FireRange": 1000 if agent_class == "RLA_CAR_Laser" else 2000,
                "RSVD1": "-Ring1=2000 -Ring2=1400 -Ring3=750"
                if agent_class == "RLA_CAR_Laser"
                else "-Ring1=2500 -Ring2=1700 -Ring3=1400",
                "RSVD2": "-InitAct=ActionSet2::Idle;StaticAlert",
                "AgentHp": np.random.randint(low=200, high=210)
                if agent_class == "RLA_CAR_Laser"
                else np.random.randint(low=95, high=100),
                "IndexInTeam": tid,
                "UID": uid,
                "Color": "(R=0,G=1,B=0,A=1)" if team == 0 else "(R=0,G=0,B=1,A=1)",
                "InitLocation": {"x": x, "y": y, "z": z},
                "InitRotator": {"pitch": 0, "roll": 0, "yaw": yaw},
            }
        )

        # 团队特殊设置
        if team == 0:
            agent_property.update(
                {
                    "FireRange": 1000 if agent_class == "RLA_CAR_Laser" else 2000,
                }
            )
        else:
            agent_property.update(
                {
                    "FireRange": 1200 if agent_class == "RLA_CAR_Laser" else 2200,
                }
            )

        return agent_property

    def init_air(self, agent_info, pos_ro):
        """
        空中单位初始化
        Args:
            agent_info: 代理信息字典
            pos_ro: 初始旋转角度
        Returns:
            代理属性字典
        """
        # 解析代理信息
        agent_class = agent_info["type"]
        team = agent_info["team"]
        n_team_agent = agent_info["n_team_agent"]
        tid = agent_info["tid"]
        uid = agent_info["uid"]
        N_COL = 2
        if n_team_agent > 40:
            N_COL = 3
        if n_team_agent > 60:
            N_COL = 5

        x = 0 + 800 * (tid - n_team_agent // 2) // N_COL
        y = 2000 * (-1) ** (team + 1)
        x, y = np.matmul(
            np.array([x, y]),
            np.array(
                [[np.cos(pos_ro), -np.sin(pos_ro)], [np.sin(pos_ro), np.cos(pos_ro)]]
            ),
        )
        z = 1200  # 飞行高度
        yaw = 90 if team == 0 else -90  # 初始朝向

        # 基础属性设置
        agent_property = copy.deepcopy(AgentPropertyDefaults)
        agent_property.update(
            {
                "DebugAgent": False,
                "MaxMoveSpeed": 1200,
                "AgentScale": {"x": 1, "y": 1, "z": 1},
                "DodgeProb": 0.5,
                "ExplodeDmg": 10,
                "AgentTeam": team,
                "ClassName": agent_class,
                "WeaponCD": 3,
                "PerceptionRange": 2500,
                "GuardRange": 1800,
                "FireRange": 1700,
                "RSVD1": "-ring1=2500 -ring2=1800 -ring3=1700",
                "RSVD2": "-InitAct=ActionSet2::Idle;StaticAlert",
                "AgentHp": 50,
                "IndexInTeam": tid,
                "UID": uid,
                "Color": "(R=0,G=1,B=0,A=1)" if team == 0 else "(R=0,G=0,B=1,A=1)",
                "InitLocation": {"x": x, "y": y, "z": z},
                "InitRotator": {"pitch": 0, "roll": 0, "yaw": yaw},
            }
        )

        return agent_property
