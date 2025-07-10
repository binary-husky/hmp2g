# 定义环境名称到模块路径和配置类的映射关系
import_path_ref = {
    "collective_assault": (
        "MISSION.collective_assault.collective_assault_parallel_run",
        "ScenarioConfig",
    ),
    "dca_multiteam": (
        "MISSION.dca_multiteam.collective_assault_parallel_run",
        "ScenarioConfig",
    ),
    "dca": ("MISSION.dca.collective_assault_parallel_run", "ScenarioConfig"),
    "air_fight": ("MISSION.air_fight.environment.air_fight_compat", "ScenarioConfig"),
    "native_gym": ("MISSION.native_gym.native_gym_config", "ScenarioConfig"),
    "starcraft2": ("MISSION.starcraft.sc2_env_wrapper", "ScenarioConfig"),
    "sc2": ("MISSION.starcraft.sc2_env_wrapper", "ScenarioConfig"),
    "unity_game": ("MISSION.unity_game.unity_game_wrapper", "ScenarioConfig"),
    "sr_tasks->cargo": (
        "MISSION.sr_tasks.multiagent.scenarios.cargo",
        "ScenarioConfig",
    ),
    "sr_tasks->hunter_invader": (
        "MISSION.sr_tasks.multiagent.scenarios.hunter_invader",
        "ScenarioConfig",
    ),
    "sr_tasks->hunter_invader3d": (
        "MISSION.sr_tasks.multiagent.scenarios.hunter_invader3d",
        "ScenarioConfig",
    ),
    "sr_tasks->hunter_invader3d_v2": (
        "MISSION.sr_tasks.multiagent.scenarios.hunter_invader3d_v2",
        "ScenarioConfig",
    ),
    "bvr": ("MISSION.bvr_sim.init_env", "ScenarioConfig"),
    "mathgame": ("MISSION.math_game.env", "ScenarioConfig"),
    "uhmap": ("MISSION.uhmap.uhmap_env_wrapper", "ScenarioConfig"),
    "llm_trainer": ("MISSION.llm_trainer.llm_env", "ScenarioConfig"),
}

env_init_function_ref = {
    "collective_assault": (
        "MISSION.collective_assault.collective_assault_parallel_run",
        "make_collective_assault_env",
    ),
    "dca_multiteam": (
        "MISSION.dca_multiteam.collective_assault_parallel_run",
        "make_collective_assault_env",
    ),
    "dca": (
        "MISSION.dca.collective_assault_parallel_run",
        "make_collective_assault_env",
    ),
    "air_fight": (
        "MISSION.air_fight.environment.air_fight_compat",
        "make_air_fight_env",
    ),
    "native_gym": ("MISSION.native_gym.native_gym_config", "env_init_function"),
    "starcraft2": ("MISSION.starcraft.sc2_env_wrapper", "make_sc2_env"),
    "sc2": ("MISSION.starcraft.sc2_env_wrapper", "make_sc2_env"),
    "unity_game": ("MISSION.unity_game.unity_game_wrapper", "make_env"),
    "sr_tasks": ("MISSION.sr_tasks.multiagent.scenario", "sr_tasks_env"),
    "bvr": ("MISSION.bvr_sim.init_env", "make_bvr_env"),
    "mathgame": ("MISSION.math_game.env", "make_math_env"),
    "uhmap": ("MISSION.uhmap.uhmap_env_wrapper", "make_uhmap_env"),
    "llm_trainer": ("MISSION.llm_trainer.llm_env", "make_llm_env"),
}

##################################################################################################################################

# 导入全局配置模块和依赖库
import importlib
import os  # 动态导入模块和操作系统接口

from config import GlobalConfig  # 全局配置对象
from UTIL.colorful import print亮蓝


def load_ScenarioConfig():
    # 校验当前环境名称是否在映射表中
    if GlobalConfig.env_name not in import_path_ref:
        assert False, "need to find path of ScenarioConfig"  # 断言失败提示

    # 获取目标模块路径和类名
    import_path, ScenarioConfig = import_path_ref[GlobalConfig.env_name]

    # 动态导入模块并获取配置类
    GlobalConfig.ScenarioConfig = getattr(
        importlib.import_module(import_path), ScenarioConfig
    )


def make_env_function(env_name, rank):
    load_ScenarioConfig()  # 加载场景配置
    ref_env_name = env_name  # 保留原始环境名称

    # 处理带子场景的环境名称（如native_gym->xxx）
    if "native_gym" in env_name:
        assert "->" in env_name  # 必须包含分隔符
        ref_env_name, env_name = env_name.split("->")  # 拆分主环境和子环境
    elif "sr_tasks" in env_name:
        assert "->" in env_name
        ref_env_name, env_name = env_name.split("->")

    # 获取模块路径和初始化函数名
    import_path, func_name = env_init_function_ref[ref_env_name]

    # 动态导入初始化函数
    env_init_function = getattr(importlib.import_module(import_path), func_name)

    # 返回带参数的lambda函数
    return lambda: env_init_function(env_name, rank)


def make_parallel_envs(process_pool, marker=""):
    # 重新加载配置（确保最新状态）
    from MISSION.env_router import load_ScenarioConfig
    from UTIL.shm_env import SuperpoolEnv  # 并行环境管理器

    load_ScenarioConfig()

    # 构建环境参数列表
    env_args_dict_list = [
        (
            {
                "env_name": GlobalConfig.env_name,
                "proc_index": i
                if "test" not in marker
                else -(i + 1),  # 测试模式使用负索引
                "marker": marker,
            },
        )
        for i in range(GlobalConfig.num_threads)
    ]

    # 特殊环境预处理
    if GlobalConfig.env_name == "air_fight":
        # 加载必须在主线程初始化的DLL
        from MISSION.air_fight.environment.pytransform import pyarmor_runtime

        pyarmor_runtime()

    if GlobalConfig.env_name == "bvr":
        # Docker权限校验
        YOUR_ROOT_PASSWORD = "clara"  # ⚠️ 需要根据实际环境修改
        os.system("echo %s|sudo -S date" % YOUR_ROOT_PASSWORD)  # 获取sudo权限
        res = os.popen("sudo docker ps").read()
        if "CONTAINER ID" not in res:
            print亮蓝(
                "[env_router]: Error checking docker in docker, can not control host docker interface!"
            )
            raise "Error checking docker in docker, can not control host docker interface!"
        pass

    if GlobalConfig.env_name == "dca":
        # This particular env has a cython file that needs to be compiled in main process
        # that must be loaded in main process
        pass
    if GlobalConfig.env_name == "dca_multiteam":
        # This particular env has a cython file that needs to be compiled in main process
        # that must be loaded in main process
        pass
    if GlobalConfig.env_name == "uhmap":
        # This particular env has a cython file that needs to be compiled in main process
        # that must be loaded in main process
        pass

    if GlobalConfig.num_threads > 1:
        envs = SuperpoolEnv(process_pool, env_args_dict_list)
    else:
        envs = SuperpoolEnv(process_pool, env_args_dict_list)

    return envs
