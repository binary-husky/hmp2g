# -*- coding: utf-8 -*-
"""
UHMAP大规模任务配置文件
定义多智能体系统的核心配置参数，包括：
- 智能体类型定义
- 观测空间配置
- 动作空间规范
- 实体交互参数
"""

from UTIL.config_args import ChainVarSimple  # 动态变量解析工具


class SubTaskConfig:
    """
    UHMAP大规模任务配置类
    包含多智能体系统的完整配置参数
    """

    # 智能体配置列表：定义所有参与者的属性
    agent_list = [
        # 蓝队（team 0）智能体配置
        {
            "team": 0,  # 所属阵营（0-蓝队）
            "tid": 0,  # 团队内ID（0-9）
            "uid": 0,  # 全局唯一ID（0-19）
            "n_team_agent": 10,  # 所属团队总智能体数
            "type": "RLA_CAR_Laser",  # 智能体类型（带激光武器的遥控车）
            "init_fn_name": "init_ground",  # 初始化函数名称
        },
        {
            "team": 0,
            "tid": 1,
            "uid": 1,
            "n_team_agent": 10,
            "type": "RLA_CAR",
            "init_fn_name": "init_ground",
        },
        {
            "team": 0,
            "tid": 2,
            "uid": 2,
            "n_team_agent": 10,
            "type": "RLA_CAR_Laser",
            "init_fn_name": "init_ground",
        },
        {
            "team": 0,
            "tid": 3,
            "uid": 3,
            "n_team_agent": 10,
            "type": "RLA_CAR",
            "init_fn_name": "init_ground",
        },
        {
            "team": 0,
            "tid": 4,
            "uid": 4,
            "n_team_agent": 10,
            "type": "RLA_CAR_Laser",
            "init_fn_name": "init_ground",
        },
        {
            "team": 0,
            "tid": 5,
            "uid": 5,
            "n_team_agent": 10,
            "type": "RLA_CAR",
            "init_fn_name": "init_ground",
        },
        {
            "team": 0,
            "tid": 6,
            "uid": 6,
            "n_team_agent": 10,
            "type": "RLA_CAR_Laser",
            "init_fn_name": "init_ground",
        },
        {
            "team": 0,
            "tid": 7,
            "uid": 7,
            "n_team_agent": 10,
            "type": "RLA_CAR",
            "init_fn_name": "init_ground",
        },
        {
            "team": 0,
            "tid": 8,
            "uid": 8,
            "n_team_agent": 10,
            "type": "RLA_CAR_Laser",
            "init_fn_name": "init_ground",
        },
        {
            "team": 0,
            "tid": 9,
            "uid": 9,
            "n_team_agent": 10,
            "type": "RLA_CAR",
            "init_fn_name": "init_ground",
        },
        # 红队（team 1）智能体配置
        {
            "team": 1,
            "tid": 0,
            "uid": 10,
            "n_team_agent": 10,
            "type": "RLA_CAR_Laser",
            "init_fn_name": "init_ground",
        },
        {
            "team": 1,
            "tid": 1,
            "uid": 11,
            "n_team_agent": 10,
            "type": "RLA_CAR",
            "init_fn_name": "init_ground",
        },
        {
            "team": 1,
            "tid": 2,
            "uid": 12,
            "n_team_agent": 10,
            "type": "RLA_CAR_Laser",
            "init_fn_name": "init_ground",
        },
        {
            "team": 1,
            "tid": 3,
            "uid": 13,
            "n_team_agent": 10,
            "type": "RLA_CAR",
            "init_fn_name": "init_ground",
        },
        {
            "team": 1,
            "tid": 4,
            "uid": 14,
            "n_team_agent": 10,
            "type": "RLA_CAR_Laser",
            "init_fn_name": "init_ground",
        },
        {
            "team": 1,
            "tid": 5,
            "uid": 15,
            "n_team_agent": 10,
            "type": "RLA_CAR",
            "init_fn_name": "init_ground",
        },
        {
            "team": 1,
            "tid": 6,
            "uid": 16,
            "n_team_agent": 10,
            "type": "RLA_CAR_Laser",
            "init_fn_name": "init_ground",
        },
        {
            "team": 1,
            "tid": 7,
            "uid": 17,
            "n_team_agent": 10,
            "type": "RLA_CAR",
            "init_fn_name": "init_ground",
        },
        {
            "team": 1,
            "tid": 8,
            "uid": 18,
            "n_team_agent": 10,
            "type": "RLA_CAR_Laser",
            "init_fn_name": "init_ground",
        },
        {
            "team": 1,
            "tid": 9,
            "uid": 19,
            "n_team_agent": 10,
            "type": "RLA_CAR",
            "init_fn_name": "init_ground",
        },
    ]

    # 观测向量维度：定义每个智能体的观测空间大小
    obs_vec_length = (
        23  # 包含位置(2) + 速度(2) + 姿态(1) + 武器状态(2) + 传感器数据(16)
    )

    # 动作格式定义：指定动作空间的编码方式
    ActionFormat = "Multi-Digit"  # 多维离散动作空间，每位代表不同行为

    # 观测范围参数：定义传感器探测半径
    OBS_RANGE_PYTHON_SIDE = 7000  # 单位：厘米（70米半径）

    # 对手观测上限：定义可见敌方单位最大数量
    MAX_NUM_OPP_OBS = 7  # 支持最多追踪7个敌方目标

    # 友方观测上限：定义可见己方单位最大数量
    MAX_NUM_ALL_OBS = 7  # 支持最多追踪7个友方单位

    # 特殊目标观测上限：定义特殊目标可见数量
    MAX_OBJ_NUM_ACCEPT = 1  # 仅追踪最近的1个关键目标

    # 实体感知数量：定义单步可处理的实体总数
    obs_n_entity = 15  # 最大处理15个实体（7+7+1）

    # 动态实体数量计算：使用链式变量解析
    obs_n_entity_cv = ChainVarSimple(
        "$MAX_NUM_OPP_OBS$ + $MAX_NUM_ALL_OBS$ + $MAX_OBJ_NUM_ACCEPT$"  # 动态计算公式
    )
