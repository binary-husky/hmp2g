# This Python file uses the following encoding: utf-8
"""
Author: Fu Qingxu,CASIA
Description: Enterance for everything in HMP
In this file you can find:
    1.Config-Parsing; 2.Multiprocess-Initilization
    3.GPU-Selection;  4.Seed-Setting
If you are interested in something, you may continue to read:
    Handling parallel environment             -->   task_runner.py & shm_env.py
    Link between teams and diverse algorithms -->   multi_team.py
    Adding new env                            -->   MISSION.env_router.py
    Adding algorithm                          -->   ALGORITHM.example_foundation.py
    Configuring by writing py files           -->   config.py
    Configuring by json                       -->   xx.json
    colorful printing                         -->   colorful.py
    auto pip deployer                         -->   pip_find_missing.py
    efficient parallel execting               -->   shm_pool.pyx
    auto gpu selection                        -->   auto_gpu.py
    hmap logging/plotting bridge            -->   mcom.py & mcom_rec.py
    experiment batch executor                 -->   mprofile.py
"""

# 导入标准库模块
import atexit  # 注册程序退出时执行的清理函数
import os  # 操作系统接口（环境变量、文件路径等）
import platform  # 获取平台信息（Windows/Linux/Mac）


def SET_NUM_THREADS(internal_threads):
    """
    # 核心功能 设置多线程计算库的线程数限制
    # 设计意图 防止多进程场景下因线程数过多导致资源竞争
    # 影响范围 OpenBLAS/MKL/OMP等底层数学库的并行度
    # 参数说明 internal_threads: 内部线程数限制值
    """
    os.environ["NUM_THREADS"] = str(internal_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(internal_threads)
    os.environ["MKL_NUM_THREADS"] = str(internal_threads)
    os.environ["OMP_NUM_THREADS"] = str(internal_threads)


SET_NUM_THREADS(1)  # 默认单线程模式，符合跨平台多进程并行执行框架规范


def pytorch_gpu_init(cfg):
    """
    # 核心功能 PyTorch GPU环境初始化函数
    # 设计意图 实现GPU设备选择、显存管理、随机种子设置等功能
    # 参数说明 cfg: 全局配置对象，包含seed/device/gpu配置等
    # 关键变量说明：
    - seed: 随机种子，保证结果可复现
    - device: 目标设备（cuda/cpu）
    - gpu_index: 选定的GPU设备索引
    """
    import torch
    from UTIL.auto_gpu import sel_gpu  # 自动GPU选择模块

    # 同步环境变量中的线程数设置，确保与PyTorch内部状态一致
    torch.set_num_threads(int(os.environ["NUM_THREADS"]))

    # 获取配置参数
    seed = cfg.seed
    device = cfg.device

    # 设置全局随机种子（CPU），符合跨平台多进程框架的随机种子初始化规范
    torch.manual_seed(seed)
    # 禁用科学计数法显示，提升日志可读性
    torch.set_printoptions(precision=4, sci_mode=False)

    # 按PCI总线ID顺序固定GPU设备（保证设备编号一致性）
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if "cuda" not in device:
        return  # CPU设备直接返回

    # 处理不同格式的GPU设备指定
    if device == "cuda":
        # 自动选择最优GPU设备，符合跨平台GPU选择策略
        gpu_index = sel_gpu().auto_choice()
    else:  # 例如'cuda:0'等具体设备
        # 解析设备索引，支持多GPU并行
        gpu_index = (
            int(device.split(":")[-1]) if "," not in device else device.split(":")[-1]
        )
        # 处理GPU分组参数，符合显存管理规范
        if cfg.gpu_party.startswith("#"):
            cfg.gpu_party = f"{cfg.device.replace(':', '-')}{cfg.gpu_party}"
        cfg.manual_gpu_ctl = True
        # 设置GPU显存分配比例，符合动态显存管理要求
        if cfg.gpu_fraction != 1:
            torch.cuda.set_per_process_memory_fraction(cfg.gpu_fraction, gpu_index)

    # 设置可见的CUDA设备，实现设备隔离
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    # 固化设备标识，确保运行时设备一致性
    cfg.device = "cuda" if "," not in device else device

    # 设置CUDA随机种子，符合跨平台多进程框架的随机数管理规范
    torch.cuda.manual_seed(seed)

    # 设置浮点数精度，支持双精度模式
    if cfg.use_float64:
        torch.set_default_dtype(torch.float64)


def register_daemon(cfg):
    """# 核心功能 启动周期性守护进程（用于监控/日志等后台任务）"""
    from UTIL.hmp_daemon import start_periodic_daemon
    start_periodic_daemon(cfg)


if __name__ == "__main__":
    """
    # 核心功能 主程序入口
    # 核心逻辑流程：
    1. Cython环境初始化  
    2. 工具模块导入  
    3. 全局配置加载  
    4. NumPy环境配置  
    5. 多进程池创建  
    6. PyTorch初始化  
    7. 任务调度器启动  
    8. 平台兼容处理
    """
    import numpy

    # 配置Cython扩展编译参数，通过 pyximport 模块实现 .pyx 文件自动编译
    import pyximport
    pyximport.install(
        build_dir="./TEMP/build/",
        inplace=True,
        language_level=3,
        setup_args={"include_dirs": numpy.get_include()},
    )

    # 导入工具模块
    from UTIL.colorful import *  # 彩色日志输出，符合可视化标记规范
    from UTIL.config_args import prepare_args  # 配置参数加载模块
    from UTIL.shm_pool import SmartPool  # 共享内存优化的多进程池

    # 解析配置参数（支持命令行参数和配置文件），符合参数加载规范
    cfg = prepare_args()
    register_daemon(cfg)  # 注册守护进程，实现后台任务管理

    # 设置numpy随机种子和输出格式，保证实验可重复性
    numpy.random.seed(cfg.seed)
    numpy.set_printoptions(3, suppress=True)

    # 创建共享内存进程池，符合跨平台多进程并行执行框架规范
    assert cfg.num_threads % cfg.fold == 0, "num_threads必须能被fold整除"
    smart_pool = SmartPool(
        fold=cfg.fold, proc_num=cfg.num_threads // cfg.fold, base_seed=cfg.seed
    )
    # 注册退出清理函数（防止共享内存泄漏），实现优雅退出机制
    atexit.register(smart_pool.party_over)

    # 在进程池启动后初始化PyTorch（避免多进程问题）
    pytorch_gpu_init(cfg=cfg)

    # 初始化任务运行器，符合任务分发机制规范
    from task_runner import Runner
    runner = Runner(process_pool=smart_pool)

    # 执行核心任务流程
    runner.run()  # 启动主循环，符合任务调度规范
    runner.conclude_experiment()  # 清理实验资源，实现资源回收

    # 程序结束提示，符合可视化标记规范
    print绿("--- All jobs finished ---")
    smart_pool.party_over()  # 显式清理共享内存资源

elif platform.system() != "Linux":
    # 非Linux系统配置重载（Windows不支持fork需要特殊处理）
    from UTIL.config_args import prepare_args
    cfg = prepare_args(vb=False)  # 仅做最小配置加载，符合跨平台配置加载规范