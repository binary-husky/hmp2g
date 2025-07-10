# 导入numpy库，用于数值计算和数组操作
import numpy as np

# # # # # # # # # # # # # # # # # # # # # # #
# # # # # Part 1, interface for RL # # # # #
# # # # # # # # # # # # # # # # # # # # # # #

# 定义动作空间的字符串列表，每个元素代表一个具体动作
# 格式："动作类型::主行为;子行为"，索引对应数字编码
dictionary_items = [
    "ActionSet2::N/A;N/A",  # 0: 无操作
    "ActionSet2::Idle;DynamicGuard",  # 1: 空闲-动态守卫
    "ActionSet2::Idle;StaticAlert",  # 2: 空闲-静态警戒
    "ActionSet2::Idle;AggressivePersue",  # 3: 空闲-主动追击
    "ActionSet2::SpecificMoving;Dir+X",  # 4: 特定移动-正X方向
    "ActionSet2::SpecificMoving;Dir+Y",  # 5: 特定移动-正Y方向
    "ActionSet2::SpecificMoving;Dir-X",  # 6: 特定移动-负X方向
    "ActionSet2::SpecificMoving;Dir-Y",  # 7: 特定移动-负Y方向
    "ActionSet2::SpecificAttacking;T1-0",  # 8: 特定攻击-目标T1-0
    "ActionSet2::SpecificAttacking;T1-1",  # 9: 特定攻击-目标T1-1
    "ActionSet2::SpecificAttacking;T1-2",  # 10: 特定攻击-目标T1-2
    "ActionSet2::SpecificAttacking;T1-3",  # 11: 特定攻击-目标T1-3
    "ActionSet2::SpecificAttacking;T1-4",  # 12: 特定攻击-目标T1-4
    "ActionSet2::SpecificAttacking;T1-5",  # 扩展攻击目标
    "ActionSet2::SpecificAttacking;T1-6",  # 扩展攻击目标
    "ActionSet2::SpecificAttacking;T0-0",  # 13: 特定攻击-目标T0-0
    "ActionSet2::SpecificAttacking;T0-1",  # 14: 特定攻击-目标T0-1
    "ActionSet2::SpecificAttacking;T0-2",  # 15: 特定攻击-目标T0-2
    "ActionSet2::SpecificAttacking;T0-3",  # 16: 特定攻击-目标T0-3
    "ActionSet2::SpecificAttacking;T0-4",  # 17: 特定攻击-目标T0-4
    "ActionSet2::SpecificAttacking;T0-5",  # 扩展攻击目标
    "ActionSet2::SpecificAttacking;T0-6",  # 扩展攻击目标
    "ActionSet2::PatrolMoving;Dir+X",  # 巡逻移动-正X方向
    "ActionSet2::PatrolMoving;Dir+Y",  # 巡逻移动-正Y方向
    "ActionSet2::PatrolMoving;Dir-X",  # 巡逻移动-负X方向
    "ActionSet2::PatrolMoving;Dir-Y",  # 巡逻移动-负Y方向
    "ActionSet2::Idle;AsFarAsPossible",  # 空闲-自动远离目标
    "ActionSet2::Idle;StayWhenTargetInRange",  # 空闲-目标进入射程时停留
    "ActionSet2::Idle;StayWhenTargetInHalfRange",  # 空闲-目标进入半射程时停留
]

# 计算动作总数
dictionary_n_actions = len(dictionary_items)

# 创建数字到动作字符串的映射字典
digit2act_dictionary = {
    i: dictionary_items[i] for i, item in enumerate(dictionary_items)
}

# 创建动作字符串到数字的映射字典
act2digit_dictionary = {
    dictionary_items[i]: i for i, item in enumerate(dictionary_items)
}

# # # # # # # # # # # # # # # # # # # # # # #
# # # # # Part 2, translate actions # # # # #
# # # # # # # # # # # # # # # # # # # # # # #

# JSON属性到本地属性的映射表，格式：(json_key, agent_key)
agent_json2local_attrs = [
    ("agentAlive", "alive"),  # 存活状态
    ("agentTeam", "team"),  # 所属队伍
    ("indexInTeam", "index"),  # 队伍内索引
    ("uId", "uid_remote"),  # 远程唯一ID
    ("maxMoveSpeed", "max_speed"),  # 最大移动速度
    ("agentLocationArr", "location"),  # 位置坐标
    ("agentRotationArr", "rotation"),  # 旋转角度
    ("agentScaleArr", "scale3"),  # 三维缩放
    ("agentVelocityArr", "velocity"),  # 速度矢量
    ("agentHp", "hp"),  # 生命值
    ("weaponCD", "weapon_cd"),  # 武器冷却时间
    ("type", "type"),  # 智能体类型
]


def encode_action_as_digits(
    main_cmd, sub_cmd, x=None, y=None, z=None, UID=None, T=None, T_index=None
):
    """将动作参数编码为数字数组"""
    # 主命令编码字典
    main_cmd_encoder = {
        "Idle": 0,  # 空闲状态
        "SpecificMoving": 1,  # 特定移动
        "PatrolMoving": 2,  # 巡逻移动
        "SpecificAttacking": 3,  # 特定攻击
        "N/A": 4,  # 无操作
    }

    # 子命令编码字典
    sub_cmd_encoder = {
        "DynamicGuard": 0,
        "StaticAlert": 1,
        "AggressivePersue": 2,
        "SpecificAttacking": 3,
        "AsFarAsPossible": 4,
        "StayWhenTargetInRange": 5,
        "StayWhenTargetInHalfRange": 6,
        "N/A": 7,
        "Dir+X": 8,
        "Dir+X+Y": 9,
        "Dir+Y": 10,
        "Dir-X+Y": 11,
        "Dir-X": 12,
        "Dir-X-Y": 13,
        "Dir-Y": 14,
        "Dir+X-Y": 15,
    }
    return np.array(
        [
            main_cmd_encoder[main_cmd],
            sub_cmd_encoder[sub_cmd],
            x if x is not None else np.inf,
            y if y is not None else np.inf,
            z if z is not None else np.inf,
            UID if UID is not None else np.inf,
            T if T is not None else np.inf,
            T_index if T_index is not None else np.inf,
        ]
    )


def decode_action_as_string(digits):
    """将数字数组解码为动作字符串"""
    # 主命令解码字典
    main_cmd_decoder = {
        0: "Idle",
        1: "SpecificMoving",
        2: "PatrolMoving",
        3: "SpecificAttacking",
        4: "N/A",
    }

    # 子命令解码字典
    sub_cmd_decoder = {
        0: "DynamicGuard",
        1: "StaticAlert",
        2: "AggressivePersue",
        3: "SpecificAttacking",
        4: "AsFarAsPossible",
        5: "StayWhenTargetInRange",
        6: "StayWhenTargetInHalfRange",
        7: "N/A",
        8: "Dir+X",
        9: "Dir+X+Y",
        10: "Dir+Y",
        11: "Dir-X+Y",
        12: "Dir-X",
        13: "Dir-X-Y",
        14: "Dir-Y",
        15: "Dir+X-Y",
    }
    main_cmd = main_cmd_decoder[digits[0]]
    sub_cmd = sub_cmd_decoder[digits[1]]
    # 解析可选参数（非inf值）
    x = digits[2] if np.isfinite(digits[2]) else None
    y = digits[3] if np.isfinite(digits[3]) else None
    z = digits[4] if np.isfinite(digits[4]) else None
    UID = digits[5] if np.isfinite(digits[5]) else None
    T = digits[6] if np.isfinite(digits[6]) else None
    T_index = digits[7] if np.isfinite(digits[7]) else None

    # 根据不同主命令生成对应动作字符串
    if main_cmd == "Idle":
        res = f"ActionSet2::Idle;{sub_cmd}"
        assert res in dictionary_items, "无效的空闲动作指令"
    elif main_cmd == "SpecificMoving":
        if sub_cmd == "N/A":
            res = f"ActionSet2::SpecificMoving;X={x} Y={y} Z={z}"  # 带坐标参数
        else:
            res = f"ActionSet2::SpecificMoving;{sub_cmd}"  # 方向移动
    elif main_cmd == "PatrolMoving":
        if sub_cmd == "N/A":
            res = "ActionSet2::PatrolMoving;X=%f Y=%f Z=%f" % (x, y, z)
        else:
            res = "ActionSet2::PatrolMoving;%s" % sub_cmd
    elif main_cmd == "SpecificAttacking":
        assert sub_cmd == "N/A", "攻击动作不允许子命令"
        if UID is not None:
            res = f"ActionSet2::SpecificAttacking;UID-{UID}"  # UID指定目标
        else:
            res = f"ActionSet2::SpecificAttacking;T{T}-{T_index}"  # 类型+索引目标
    elif main_cmd == "N/A":
        res = "ActionSet2::N/A;N/A"
    else:
        raise AssertionError("无效的动作指令")
    return res


# # # # # # # # # # # # # # # # # # # # # # #
# # # # # Part 3, agent init defaults # # # #
# # # # # # # # # # # # # # # # # # # # # # #

# 智能体默认配置参数
AgentPropertyDefaults = {
    "ClassName": "RLA_CAR",  # 智能体类别名称
    "DebugAgent": False,  # 调试模式开关
    "AgentTeam": 0,  # 默认队伍编号
    "IndexInTeam": 0,  # 队伍内索引
    "UID": 0,  # 唯一标识符
    "MaxMoveSpeed": 600,  # 最大移动速度
    "InitLocation": {"x": 0, "y": 0, "z": 0},  # 初始位置
    "InitRotation": {"x": 0, "y": 0, "z": 0},  # 初始旋转
    "AgentScale": {"x": 1, "y": 1, "z": 1},  # 初始尺寸
    "InitVelocity": {"x": 0, "y": 0, "z": 0},  # 初始速度
    "AgentHp": 100,  # 初始生命值
    "WeaponCD": 1,  # 武器冷却时间
    "IsTeamReward": True,  # 是否启用团队奖励
    "Type": "",  # 智能体类型
    "DodgeProb": 0.8,  # 闪避概率
    "ExplodeDmg": 25,  # 爆炸伤害值
    "FireRange": 1000.0,  # 射程范围
    "GuardRange": 1400.0,  # 警戒范围
    "PerceptionRange": 1500.0,  # 感知范围
    "Color": "(R=0,G=1,B=0,A=1)",  # 显示颜色
    "RSVD1": "",  # 保留字段1
    "RSVD2": "",  # 保留字段2
}

# # # # # # # # # # # # # # # # # # # # # # #
# # # # # Part 4, utility functions # # # # #
# # # # # # # # # # # # # # # # # # # # # # #


def binary_friendly(x):
    """检查数值是否可被float16精确表示"""
    y_f16 = np.array(x, dtype=np.float16)  # 转换为float16
    y_f64 = np.array(x, dtype=np.float64)  # 转换为float64
    t = y_f64 - y_f16  # 计算精度差异
    assert t.dtype == np.float64  # 验证数据类型
    return t == 0  # 返回是否无精度损失


# # # # # # # # # # # # # # # # # # # # # # #
# # # # # Part 5, legacy code # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # #

# 被弃用的单数字编码函数（用于坐标转换）
# h_map_center = (-7290.0, 6010.0)  # 地图中心坐标
# h_grid_size = 400  # 水平网格尺寸
# v_ground = 340  # 地面高度
# v_grid_size = 1000  # 垂直网格尺寸

# def _2digit(...):  # 将坐标转换为单个数字编码
# def _2coordinate(...):  # 将数字编码还原为坐标

# 事件日志示例（用于调试）
# <Agent>T0-5<UID>5<Event>Destroyed
# <Event>EndEpisode<EndReason>...

# 注释标记：
# # # # # 表示主要部分分隔
# ## 表示代码块分隔
# 注释内容使用中文说明代码功能
# 关键参数和返回值进行明确标注
# 特殊约束条件（如assert）进行说明
