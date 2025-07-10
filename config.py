# 该文件定义了 HMP 框架的核心全局配置类


import numpy as np

from UTIL.colorful import *
from UTIL.config_args import ChainVar

"""
    GlobalConfig: This config class will be 'injected' with new settings from JSONC.
    (E.g., override configs with ```python main.py --cfg example.jsonc```)
    (As the name indicated, ChainVars will change WITH vars it 'chained_with' during config injection)
    (please see UTIL.config_args to find out how this advanced trick works out.)

    * Explaining a very important setting option: 
        - align_episode (True/False):
            In align mode, all threads begin new episode synchronously,
            which means a env thread that ends early has to 'wait' for other threads before restarting
            If set to 'False', threads will not wait for each others, and will 'reset' immediately on 'done'

        - note (str):
            Name you experiment carefully with note setting.
            The note defines where the results of a single experiment will go. 
            for example, if note='conc', everything produced in the experiment will be save in RESULT/conc/*,
            including images, saved pytorch model, 
        
        - env_name:
            Which mission/environment/task to use, 
            See ./MISSION/env_router.py for the dictionary of available envs.

        - env_path:
            The path of selected mission. In fact, hmp do not need this setting at all,
            it exists here only to Double Check that you have chosen the correct mission env.

    * Why the Algorithm selection is not here ?!
        It is the missions (envs) that choose the algorithm(s)! Please go to mission configuration!
        Important to remember:
            - hmp selects a mission, 
            - mission selects algorithm(s).
        In fact, if you have two teams in env, 
        you can choose two different algorithms to fight each other in the same env!
            - Please goto ./MISSION/env_router.py to find out where the ScenarioConfig of your env is written,
            - Please set ```TEAM_NAMES``` to include the path of your favored algorithm(s)
"""


class GlobalConfig(object):  # ADD_TO_CONF_SYSTEM //DO NOT remove this comment//
    """
    HMP框架核心全局配置类

    功能特性：
    - 支持动态配置注入（通过JSON配置文件）
    - 使用链式变量(ChainVar)实现参数自动更新
    - 包含环境/训练/日志等核心参数

    配置优先级：命令行参数 > JSON配置 > 默认值
    """

    # 环境同步控制
    align_episode = True  # ! please try to understand this with TOP priority  True

    # 环境选择配置
    env_name = "sr_tasks->cargo"  # 指定使用的环境名称，参考./MISSION/env_router.py
    env_path = "MISSION.sr_tasks.multiagent.cargo"  # 环境路径（双重校验防止错误）
    draw_mode = "OFF"  # 可视化模式：Img/Threejs/Web/Native
    activate_logger = True  # 是否启用数据记录器（Tensorboard被弃用）
    data_logger = "auto load, do not change this var!"  # 数据记录器状态

    # 恢复训练配置
    resume_mod = False  # 是否恢复未完成的训练
    mt_act_order = "old_method"  # 恢复模式下的动作顺序
    mt_parallel = False  # 是否并行处理多线程任务

    # 随机种子设置
    seed = np.random.randint(0, 100000)  # 随机种子（numpy和pytorch）

    # 实验标识与日志路径
    note = "more_testing"  # 实验备注（决定日志路径：./RESULT/$note/*）
    logdir = "./RESULT/%s/" % note  # 日志存储目录
    # 链式变量：自动跟随note变化
    logdir_cv = ChainVar(lambda note: "./RESULT/%s/" % note, chained_with=["note"])
    recall_previous_session = False  # 是否继续之前的训练会话

    # 测试相关配置
    test_only = False  # 仅测试模式开关
    test_logger = "test_only_profile.txt"  # 测试日志路径

    # 设备选择与GPU管理
    device = "cuda"  # 设备选择（cpu/cuda/cuda:x）
    gpu_party = "off"  # GPU内存共享组（多个进程共享GPU内存）
    manual_gpu_ctl = False  # 手动GPU控制标志（自动生成）
    gpu_fraction = 1.0  # GPU内存使用比例（0-1.0）

    # 并行计算配置
    num_threads = 64  # 并行环境线程数
    fold = 1  # 进程折叠数（提升IPC效率，感谢Python GIL的"馈赠"）

    # 训练规模参数
    n_parallel_frame = int(5e6)  # 总训练帧数（每帧所有环境执行一步）
    max_n_episode = int(2e5)  # 最大训练回合数

    # 数值精度控制
    use_float64 = False  # 是否强制使用float64精度

    # 感兴趣的智能体配置
    interested_team = 0  # 需要记录奖励的团队ID
    interested_agent_num = 50  # 感兴趣的智能体数量
    interested_agent_uid = range(0, 50)  # 智能体索引范围（自动根据数量生成）
    interested_agent_uid_cv = ChainVar(
        lambda interested_agent_num: range(0, interested_agent_num),
        chained_with=["interested_agent_num"],
    )  # 链式变量：自动跟随智能体数量变化

    # 报告间隔设置
    report_reward_interval = num_threads  # 奖励报告间隔
    # 链式变量：自动跟随线程数变化
    report_reward_interval_cv = ChainVar(
        lambda num_threads: num_threads, chained_with=["num_threads"]
    )

    # 测试策略配置
    train_time_testing = True  # 是否在训练期间进行测试
    test_interval = 32 * num_threads  # 测试间隔（依赖num_threads）
    # 链式变量：自动跟随线程数变化
    test_interval_cv = ChainVar(
        lambda num_threads: 4 * num_threads, chained_with=["num_threads"]
    )
    test_epoch = 32 if num_threads <= 32 else num_threads  # 测试轮数
    # 链式变量：自动跟随线程数变化
    test_epoch_cv = ChainVar(
        lambda num_threads: 32 if num_threads <= 32 else num_threads,
        chained_with=["num_threads"],
    )

    # 任务特定配置（自动链接）
    ScenarioConfig = "This ScenarioConfig var will be automatically linked to task configuration later in ./MISSION/env_router.py"

    # 日志与备份配置
    backup_files = []  # 需要备份的文件列表
    hmap_logger = None  # 全局日志记录器
    heartbeat_on = True  # 是否启用心跳提示（命令行视觉效果）

    # 内部状态标志
    cfg_ready = False  # 配置就绪标志（自动设置为True）

    # 结果上传配置
    allow_res_upload = True  # 是否允许退出时上传结果
    upload_after_test = False  # 是否在测试完成后上传结果
    machine_info = "auto load, do not change this var!"  # 机器信息（自动生成）
    remote_server_ops = ""  # 远程服务器操作指令
