# 导入标准库模块
import importlib  # 动态导入模块
import json      # JSON数据处理
import os        # 操作系统接口
import platform  # 获取平台信息
import stat      # 文件状态常量
import subprocess# 子进程管理
import time      # 时间相关操作

# 导入第三方库
import numpy as np  # 数值计算库

# 导入项目内部模块
from config import GlobalConfig  # 全局配置
from UTIL.colorful import print蓝  # 彩色打印工具
from UTIL.config_args import ChainVar  # 配置链式变量
from UTIL.network import TcpClientP2PWithCompress, find_free_port_no_repeat, get_host_ip  # 网络工具

# 导入父级模块
from ..common.base_env import BaseEnv  # 基础环境类
from .actset_lookup import binary_friendly, dictionary_n_actions  # 动作集检查工具

def make_uhmap_env(env_id, rank):
    """
    工厂函数创建UHMAP环境实例
    Args:
        env_id: 环境ID
        rank: 进程等级
    Returns:
        UhmapEnv实例或子任务类实例
    """
    if ScenarioConfig.SubTaskSelection == "UhmapEnv":
        return UhmapEnv(rank)
    else:
        ST = ScenarioConfig.SubTaskSelection
        # 验证子任务模块存在性
        assert os.path.exists(f"./MISSION/uhmap/SubTasks/{ST}.py"), "Unknown subtask!"
        # 动态加载子任务类
        ST_CLASS = getattr(
            importlib.import_module(f".SubTasks.{ST}", package="MISSION.uhmap"), ST
        )
        return ST_CLASS(rank)

def get_subtask_conf(subtask):
    """
    获取子任务配置类
    Args:
        subtask: 子任务名称
    Returns:
        子任务配置类
    """
    ST = subtask
    # 验证配置文件存在性
    assert os.path.exists(f"./MISSION/uhmap/SubTasks/{ST}Conf.py"), "Configuration not found!"
    # 动态加载配置类
    ST_CONF_CLASS = getattr(
        importlib.import_module(f".SubTasks.{ST}Conf", package="MISSION.uhmap"),
        "SubTaskConfig",
    )
    return ST_CONF_CLASS

def usual_id_arrangment(N_AGENT_EACH_TEAM):
    """
    生成代理ID分配方案
    Args:
        N_AGENT_EACH_TEAM: 各队代理数量列表
    Returns:
        按团队划分的ID范围列表
    """
    AGENT_ID_EACH_TEAM = []
    p = 0
    for team_agent_num in N_AGENT_EACH_TEAM:
        AGENT_ID_EACH_TEAM.append(range(p, p + team_agent_num))
        p += team_agent_num
    return AGENT_ID_EACH_TEAM

class ScenarioConfig(object):
    """
    场景配置类：通过JSONC进行配置注入
    ChainVars参数会随着依赖参数变化自动更新
    """
    # <Part 1> HMP核心所需参数
    N_AGENT_EACH_TEAM = [10]  # 默认每队代理数
    AGENT_ID_EACH_TEAM = usual_id_arrangment(N_AGENT_EACH_TEAM)  # 生成ID分配
    N_TEAM = len(N_AGENT_EACH_TEAM)  # 团队数量

    # 链式参数（自动跟随N_AGENT_EACH_TEAM更新）
    AGENT_ID_EACH_TEAM_cv = ChainVar(
        lambda N_AGENT_EACH_TEAM: usual_id_arrangment(N_AGENT_EACH_TEAM),
        chained_with=["N_AGENT_EACH_TEAM"],
    )
    N_TEAM_cv = ChainVar(
        lambda N_AGENT_EACH_TEAM: len(N_AGENT_EACH_TEAM),
        chained_with=["N_AGENT_EACH_TEAM"],
    )

    # 算法选择配置
    TEAM_NAMES = ["ALGORITHM.None->None"]

    # 数据格式配置
    ActAsUnity = False   # 动作数组是否按团队维度
    RewardAsUnity = True # 奖励数组是否单值
    ObsAsUnity = False   # 观测数组是否按代理维度

    # <Part 2> 环境自身所需参数
    MaxEpisodeStep = 100  # 最大步数
    render = False        # 是否渲染
    js_render = False     # 三JS调试渲染
    TcpAddr = "127.0.0.1" # TCP地址
    UhmapPort = 21051     # UHMAP端口

    # 子任务配置
    UnrealLevel = "UhmapBreakingBad"
    SubTaskSelection = "UhmapBreakingBad"
    SubTaskConfig = get_subtask_conf(UnrealLevel)
    SubTaskConfig_cv = ChainVar(
        lambda UnrealLevel: get_subtask_conf(UnrealLevel),
        chained_with=["SubTaskSelection"],
    )

    # 端口配置
    UElink2editor = False
    AutoPortOverride = True
    AutoPortOverride_cv = ChainVar(
        lambda UElink2editor: (not UElink2editor), 
        chained_with=["UElink2editor"]
    )

    # 时间控制参数
    StepGameTime = 0.5  # 步长时间

    # 可执行文件路径
    UhmapServerExe = "F:/UHMP/Build/WindowsServer/UHMPServer.exe"
    UhmapRenderExe = ""

    # 引擎参数
    TimeDilation = 1.0  # 时间膨胀系数
    FrameRate = 25.6    # 帧率
    FrameRate_cv = ChainVar(
        lambda TimeDilation: (TimeDilation / 1 * 25.6), 
        chained_with=["TimeDilation"]
    )
    UhmapStartCmd = []

    # <Part 3> 算法所需参数
    StateProvided = False
    AvailActProvided = False
    EntityOriented = True

    ActionFormat = "ASCII"  # 动作格式

    n_actions = dictionary_n_actions  # 动作空间大小

    # 观测空间参数
    obs_vec_length = get_subtask_conf(UnrealLevel).obs_vec_length
    obs_vec_length_cv = ChainVar(
        lambda UnrealLevel: get_subtask_conf(UnrealLevel).obs_vec_length,
        chained_with=["SubTaskSelection"],
    )

    obs_n_entity = get_subtask_conf(UnrealLevel).obs_n_entity
    obs_n_entity_cv = ChainVar(
        lambda UnrealLevel: get_subtask_conf(UnrealLevel).obs_n_entity,
        chained_with=["SubTaskSelection"],
    )

    # 版本与功能开关
    UhmapVersion = "2.3"
    CanTurnOff = False
    HeteAgents = False  # 异构代理
    DemoType = "Default"  # 演示类型

class UhmapEnvParseHelper:
    def parse_response_ob_info(self, response):
        """解析响应观测信息（需子类实现）"""
        raise NotImplementedError

    def make_obs(self):
        """构建观测数据（需子类实现）"""
        raise NotImplementedError

class UhmapEnv(BaseEnv, UhmapEnvParseHelper):
    def __init__(self, rank) -> None:
        """
        初始化UHMAP环境
        Args:
            rank: 进程等级
        """
        super().__init__(rank)
        self.id = rank
        # 渲染设置（仅主进程）
        self.render = ScenarioConfig.render and (self.id == 0)
        # 代理数量统计
        self.n_agents = sum(ScenarioConfig.N_AGENT_EACH_TEAM)
        assert self.n_agents == len(ScenarioConfig.SubTaskConfig.agent_list), (
            "代理数量定义错误"
        )
        self.n_teams = ScenarioConfig.N_TEAM
        # 仿真线程和客户端初始化
        self.sim_thread = None
        self.client = None

        # 仿真生命周期管理
        self.max_simulation_life = 2048
        self.simulation_life = self.max_simulation_life

        # 线程同步机制
        traffic_light = (
            "./TEMP/uhmap_thread_0_init_ok_%s"
            % GlobalConfig.machine_info["ExpUUID"][:8]
        )
        if rank != 0:
            while not os.path.exists(traffic_light):
                time.sleep(1)  # 等待主线程初始化

        # 激活仿真
        self.activate_simulation(self.id, find_port=True)

        # 标记初始化完成
        if rank == 0:
            with open(traffic_light, mode="w+") as f:
                f.write(traffic_light)

    def __del__(self):
        """析构函数：终止仿真"""
        self.terminate_simulation()

    def activate_simulation(self, rank, find_port=True):
        """
        激活仿真环境
        Args:
            rank: 进程等级
            find_port: 是否自动寻找端口
        """
        print("thread %d initializing" % rank)
        self.sim_thread = "activiting"

        if find_port:
            # 端口配置逻辑
            self.hmp_ue_port = ScenarioConfig.UhmapPort
            if ScenarioConfig.AutoPortOverride:
                self.hmp_ue_port, release_port_fn = find_free_port_no_repeat()
            if not ScenarioConfig.UElink2editor:
                self.ue_vis_port, release_port_fn = find_free_port_no_repeat()
                print蓝("Port %d will be used by hmp, port %d will be used by UE internally" % (self.hmp_ue_port, self.ue_vis_port))
            # 可视化提示
            if (not self.render) and (not ScenarioConfig.UElink2editor):
                print蓝('To visualize on Windows, use Git Bash to run "./UHMP.exe -OpenLevel=%s:%d ..."' % (get_host_ip(), self.ue_vis_port))
            self.ip_port = (ScenarioConfig.TcpAddr, self.hmp_ue_port)

        # 主要仿真配置验证
        if not ScenarioConfig.UElink2editor:
            assert ScenarioConfig.AutoPortOverride
            print("checking ScenarioConfig args problems ...")
            assert ScenarioConfig.TimeDilation <= 128, "* TimeDilation <= 128 *"
            assert binary_friendly(1 / ScenarioConfig.FrameRate), "* A Butterfly Effect problem *"
            assert binary_friendly(ScenarioConfig.TimeDilation / 256), "* A Butterfly Effect problem *"

            # 可执行文件路径处理
            simulation_exe = ScenarioConfig.UhmapServerExe if not self.render else ScenarioConfig.UhmapRenderExe
            if platform.system() == "Linux":
                if self.render:
                    assert False, (
                        "You really want to render on Linux? If so, remove this line."
                    )
                if simulation_exe.endswith(".exe"):
                    simulation_exe = simulation_exe.replace("/Windows", "/Linux").replace(".exe", ".sh")
                simulation_exe = os.path.expanduser(simulation_exe)
            else:  # Windows
                if simulation_exe.endswith(".sh"):
                    simulation_exe = simulation_exe.replace("/Linux", "/Windows").replace(".sh", ".exe")
                if simulation_exe.startswith("/home"):
                    simulation_exe = "./TEMP" + simulation_exe

            # 自动下载二进制文件
            if not os.path.exists(simulation_exe):
                if self.rank == 0:
                    from .auto_download import download_client_binary
                    download_client_binary(
                        desired_path=simulation_exe,
                        desired_version=ScenarioConfig.UhmapVersion,
                        is_render_client=self.render,
                    )
                else:
                    while not os.path.exists(simulation_exe): time.sleep(60)

            # Linux可执行权限设置
            if platform.system() == "Linux":
                st = os.stat(simulation_exe)
                os.chmod(simulation_exe, st.st_mode | stat.S_IEXEC)

            # 启动仿真进程
            if (not self.render) and simulation_exe != "":
                # start child process
                self.sim_thread = subprocess.Popen(
                    [
                        simulation_exe,
                        # '-log',
                        "-TcpPort=%d"
                        % self.hmp_ue_port,  # port for hmp data exchanging
                        "-Port=%d" % self.ue_vis_port,  # port for remote visualizing
                        "-OpenLevel=%s" % ScenarioConfig.UnrealLevel,
                        "-TimeDilation=%.8f" % ScenarioConfig.TimeDilation,
                        "-FrameRate=%.8f" % ScenarioConfig.FrameRate,
                        "-IOInterval=%.8f" % ScenarioConfig.StepGameTime,
                        "-Seed=%d"
                        % int(
                            np.random.rand() * 1e5
                        ),  # 如果已经设定了主线程随机数种子，这里随机出来的数字则是确定的
                        "-DebugMod=False",
                        # '-LLMCSV',
                        "-ABSLOG=%s"
                        % os.path.abspath(
                            "./TEMP/uhmap/%s/%d.log"
                            % (GlobalConfig.machine_info["ExpUUID"][:8], rank)
                        ),
                        "-Version=%s" % ScenarioConfig.UhmapVersion,
                        "-LockGameDuringCom=True",
                    ],
                    stdout=subprocess.DEVNULL,
                )
                print("UHMAP (Headless) started ...")
            elif self.render and simulation_exe != "":
                self.sim_thread = subprocess.Popen(
                    [
                        simulation_exe,
                        # '-log',
                        "-TcpPort=%d"
                        % self.hmp_ue_port,  # port for hmp data exchanging
                        "-Port=%d" % self.ue_vis_port,  # port for remote visualizing
                        "-OpenLevel=%s" % ScenarioConfig.UnrealLevel,
                        "-TimeDilation=%.8f" % ScenarioConfig.TimeDilation,
                        "-FrameRate=%.8f" % ScenarioConfig.FrameRate,
                        "-IOInterval=%.8f" % ScenarioConfig.StepGameTime,
                        "-Seed=%d"
                        % int(
                            np.random.rand() * 1e5
                        ),  # 如果已经设定了主线程随机数种子，这里随机出来的数字则是确定的
                        "-DebugMod=False",
                        # '-LLMCSV',
                        "-ABSLOG=%s"
                        % os.path.abspath(
                            "./TEMP/uhmap/%s/%d.log"
                            % (GlobalConfig.machine_info["ExpUUID"][:8], rank)
                        ),
                        "-Version=%s" % ScenarioConfig.UhmapVersion,
                        "-LockGameDuringCom=True",
                        "-ResX=1280",
                        "-ResY=720",
                        "-WINDOWED",
                    ],
                    stdout=subprocess.DEVNULL,
                )
                print("UHMAP (Render) started ...")
            else:
                print("Cannot start Headless Server Or GUI Server!")
                assert False, "Cannot start Headless Server Or GUI Server!"
        else:
            print("Trying to link to unreal editor ...")
            assert not ScenarioConfig.AutoPortOverride

        time.sleep(1 + np.abs(self.id) / 100)
        self.client = TcpClientP2PWithCompress(self.ip_port)
        MAX_RETRY = 150
        for i in range(MAX_RETRY):
            try:
                self.client.manual_connect()
                print("handshake complete %d" % rank)
                break
            except:
                if i > 25:
                    print(
                        "Thread %d: Trying to connect to unreal engine. Related library not in memory, going to take some minutes. Retry %d ..."
                        % (rank, i)
                    )
                elif i > 75:
                    print(
                        "Thread %d: Waiting too long, please reduce parallel threads (num_threads), Retry %d ... | 请减小num_threads运行一次, 让动态库载入内存, 然后恢复num_threads即可"
                        % (rank, i)
                    )
                elif i >= MAX_RETRY - 1:
                    assert False, (
                        "uhmap connection timeout, please reduce parallel threads (num_threads) !"
                    )
                time.sleep(1)

        # 端口释放
        if find_port:
            if ScenarioConfig.AutoPortOverride:
                release_port_fn(self.hmp_ue_port)
            if not ScenarioConfig.UElink2editor:
                release_port_fn(self.ue_vis_port)
        self.t = 0
        print("thread %d initialize complete" % rank)

    def terminate_simulation(self):
        if (
            hasattr(self, "sim_thread")
            and (self.sim_thread is not None)
            and (self.client is not None)
        ):
            # self.sim_thread.terminate()
            # send terminate command to unreal side
            self.client.send_dgram_to_target(
                json.dumps(
                    {
                        "valid": True,
                        "DataCmd": "end_unreal_engine",
                        "TimeStepMax": ScenarioConfig.MaxEpisodeStep,
                        "TimeStep": 0,
                        "Actions": None,
                    }
                )
            )
            self.client.close()
            self.sim_thread = None
            self.client = None

    def reset(self):
        """重置环境"""
        self.simulation_life -= 1
        if self.simulation_life < 0:
            self.terminate_simulation()
            self.simulation_life = self.max_simulation_life
            self.activate_simulation(self.id, find_port=False)

    def sleep(self):
        """休眠环境"""
        self.simulation_life = -1
        self.terminate_simulation()

    def step(self, act):
        """执行一步仿真（未实现）"""
        raise NotImplementedError