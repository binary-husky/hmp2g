# 导入argparse模块用于解析命令行参数
import argparse

# 导入os模块用于操作系统路径操作和目录创建
import os

# 导入re模块用于正则表达式处理参数绑定字符串
import re

# 导入time模块用于时间戳生成和休眠操作
import time

# 从shutil导入文件操作函数
from shutil import copyfile, copytree, ignore_patterns, rmtree

# 导入func_timeout模块实现带超时控制的输入等待
import func_timeout

# 从colorful模块导入颜色输出函数
from .colorful import *

# 从data_struct模块导入remove_suffix工具函数
from .data_struct import remove_suffix


# 核心功能: 定义参数绑定基类，实现参数联动机制
# 设计意图: 通过函数绑定和依赖参数列表维护参数间的关系
class ChainVar(object):
    """
    # 核心功能 处理绑定超参数的基类（如线程数与测试间隔的关联）
    # 参数说明:
    - chain_func: 参数绑定函数
    - chained_with: 被绑定的参数列表
    # 设计意图 实现参数联动更新机制，保证相关参数一致性
    # 关键变量:
    - chain_func: 存储参数绑定逻辑
    - chained_with: 记录依赖参数名称
    """

    def __init__(self, chain_func, chained_with):
        # 存储参数绑定函数
        self.chain_func = chain_func
        # 存储依赖参数名称列表
        self.chained_with = chained_with


# 核心功能: 简化版参数绑定类，通过字符串模板构建绑定关系
class ChainVarSimple(object):
    """
    # 核心功能 简化的参数绑定类（通过字符串模板构建绑定关系）
    # 参数说明:
    - str: 包含绑定表达式的字符串（如"$threads$*10"）
    # 处理流程:
    1. 提取$xxx$格式的依赖变量名
    2. 将模板转换为可执行的cfg对象访问表达式
    """

    def __init__(self, str):
        # 使用正则表达式提取所有$xxx$格式的变量名作为依赖项
        self.chained_with = re.findall(re.compile(r"\$(.*?)\$"), str)
        # 将$xxx$模板替换为cfg.xxx格式的可执行表达式
        self.chain_func_str = re.sub(r"\$(.*?)\$", r"cfg.\1", str)


# 核心功能: 执行参数绑定表达式求值
def perform_varible_chain_react(cv: ChainVarSimple, cfg):
    """
    # 核心功能 执行参数绑定表达式求值
    # 关键逻辑:
    1. 使用动态执行机制计算绑定表达式
    2. 通过列表包装实现可变对象引用传递
    # 安全检查: 确保执行结果不为空
    """
    # 使用列表实现闭包可变性，突破exec作用域限制
    res = [None]
    # 动态执行转换后的表达式
    exec(f"res[0]={cv.chain_func_str}")
    # 安全检查：确保表达式执行成功
    assert res[0] is not None
    # 返回计算结果
    return res[0]


# 核心功能: 检测参数是否为绑定参数
def is_chained_key(key):
    """
    # 核心功能 检测参数是否为绑定参数
    # 返回值: (是否绑定参数, 原始参数名)
    # 命名规范检查:
    - 以'_cv'结尾: 简单绑定参数
    - 以'_CV'结尾: 复杂绑定参数
    """
    # 检测简单绑定参数
    if key.endswith("_cv"):
        # 返回去后缀的原始名
        return True, remove_suffix(key, "_cv")
    # 检测复杂绑定参数
    elif key.endswith("_CV"):
        # 返回去后缀的原始名
        return True, remove_suffix(key, "_CV")
    # 非绑定参数
    else:
        return False, key


def prepare_args(vb=True):
    """
    # 核心功能 配置参数加载主函数
    # 参数说明:
    - vb: 是否输出详细日志
    # 主要流程:
    1. 解析命令行参数
    2. 加载JSON配置文件
    3. 处理参数依赖关系
    4. 初始化日志目录
    # 安全检查:
    - 强制要求配置文件路径参数
    - 参数冲突检测
    """
    if vb:
        prepare_tmp_folder()  # 创建临时目录用于存储中间文件
    parser = argparse.ArgumentParser(description="HMP")  # 创建命令行参数解析器
    parser.add_argument(
        "-c", "--cfg", help="Path of the configuration file"
    )  # 添加配置文件路径参数
    parser.add_argument(
        "-s", "--skip", action="store_true", help="skip logdir check"
    )  # 添加跳过日志目录检查标志
    args, unknown = parser.parse_known_args()  # 解析已知参数并保留未知参数
    load_via_json = (
        hasattr(args, "cfg") and args.cfg is not None
    )  # 检查是否提供配置文件路径
    assert load_via_json  # 安全检查：必须提供配置文件路径

    # 处理跳过日志目录检查标志
    skip_logdir_check = (
        hasattr(args, "skip") and (args.skip is not None) and args.skip
    ) or (not vb)  # 当强制模式关闭或显式跳过时启用跳过检查

    # 警告未识别的参数。
    if len(unknown) > 0 and vb:
        print亮红(
            "Warning! In json setting mode, %s is ignored" % str(unknown)
        )  # 输出未处理参数警告

    # 加载JSON配置文件
    import commentjson as json  # 导入带注释支持的JSON解析库

    if vb:
        print亮绿("reading configuration at", args.cfg)  # 输出配置文件读取提示
    with open(args.cfg, encoding="utf8") as f:  # 以UTF-8编码打开配置文件
        json_data = json.load(f)  # 加载JSON配置数据

    # 处理临时算法文件夹
    if vb:
        prepare_alg_tmp_folder(json_data)  # 根据配置创建算法临时目录

    # 注入配置到全局对象
    load_config_via_json(json_data, vb)  # 将JSON配置注入到全局配置对象

    # 获取全局配置对象
    from config import GlobalConfig as cfg  # 导入全局配置类

    # 检查日志路径冲突
    note_name_overide = None
    if not skip_logdir_check:
        note_name_overide = check_experiment_log_path(
            cfg.logdir
        )  # 检查日志目录是否存在冲突
        if note_name_overide is not None:
            override_config_file(  # 覆盖配置文件中的note字段
                "config.py->GlobalConfig", {"note": note_name_overide}, vb
            )

    # 创建日志目录
    if not os.path.exists(cfg.logdir):  # 如果日志目录不存在
        os.makedirs(cfg.logdir)  # 创建日志目录

    # 备份关键文件
    if (not cfg.recall_previous_session) and vb:  # 如果不是恢复模式且处于详细模式
        copyfile(args.cfg, "%s/experiment.jsonc" % cfg.logdir)  # 备份实验配置文件
        if not os.path.exists(
            "%s/raw_exp.jsonc" % cfg.logdir
        ):  # 如果原始配置文件未备份
            copyfile(args.cfg, "%s/raw_exp.jsonc" % cfg.logdir)  # 创建原始配置文件备份
        backup_files(cfg.backup_files, cfg.logdir, args.cfg)  # 备份其他指定文件
        cfg.machine_info = register_machine_info(cfg.logdir)  # 注册机器信息

    # 标记配置加载完成
    cfg.cfg_ready = True  # 设置配置就绪标志
    return cfg  # 返回全局配置对象


def load_config_via_json(json_data, vb):
    """
    # 核心功能 从JSON数据加载配置
    # 处理流程:
    1. 遍历所有配置组
    2. 覆盖对应配置对象属性
    3. 处理参数依赖关系
    """
    # 遍历JSON配置文件中的所有配置组
    for cfg_group in json_data:  # 遍历每个配置组
        # 特殊处理全局配置组的随机种子警告
        if cfg_group == "config.py->GlobalConfig":  # 如果是全局配置组
            random_seed_warning(json_data[cfg_group])  # 发出随机种子警告
        # 调用配置覆盖函数并获取依赖项列表
        dependency = override_config_file(
            cfg_group, json_data[cfg_group], vb
        )  # 覆盖单个配置组
        # 验证依赖项是否存在
        if dependency is not None:  # 如果存在依赖项
            for dep in dependency:  # 遍历所有依赖项
                # 断言检查依赖配置项是否存在
                assert any([dep in k for k in json_data.keys()]), (  # 验证依赖项存在性
                    "Arg check failure, There is something missing!"
                )
    # 检查配置相关性（环境配置必须与环境路径匹配）
    check_config_relevence(json_data)  # 验证配置相关性
    return None  # 返回空值


def override_config_file(cfg_group, new_cfg, vb):
    """
    # 核心功能 覆盖指定配置组的属性
    # 处理步骤:
    1. 解析配置组标识
    2. 加载对应模块和类
    3. 逐项覆盖配置属性
    4. 处理绑定参数更新
    """
    import importlib  # 导入动态导入模块

    # 断言检查配置组标识必须包含'->'符号
    assert "->" in cfg_group  # 确保配置组标识符包含'->'
    # 构建配置组显示字符串
    str_pro = "------------- %s -------------" % cfg_group  # 构建配置组分隔字符串
    if vb:
        print绿(str_pro)  # 输出配置组分隔符
    # 分割文件名和类名
    file_, class_ = cfg_group.split("->")  # 分割文件路径和类名
    # 移除文件名后缀
    if ".py" in file_:  # 如果文件名包含.py后缀
        file_ = file_[:-3]  # 移除.py后缀
    # 动态加载配置模块和类
    default_configs = getattr(importlib.import_module(file_), class_)  # 动态加载配置类

    # 逐项覆盖配置属性
    for key in new_cfg:  # 遍历新配置的所有键
        if new_cfg[key] is None:  # 如果值为空
            continue  # 跳过空值
        # 调用安全设置属性函数
        my_setattr(
            conf_class=default_configs, key=key, new_value=new_cfg[key], vb=vb
        )  # 安全设置属性

    # 处理绑定参数更新
    altered_cv = secure_chained_vars(default_configs, new_cfg, vb)  # 处理绑定参数更新

    # 输出配置摘要
    if vb:  # 如果处于详细模式
        print绿("".join(["-"] * len(str_pro)))  # 输出分隔线
        # 显示配置变更状态
        arg_summary(default_configs, new_cfg, altered_cv)  # 输出配置变更摘要
        print绿("".join(["-"] * len(str_pro)), "\n\n\n")  # 输出结束分隔线

    # 返回依赖项列表
    if "TEAM_NAMES" in new_cfg:  # 如果配置包含TEAM_NAMES字段
        return [
            item.split("->")[0]  # 提取依赖项名称
            for item in new_cfg["TEAM_NAMES"]  # 遍历TEAM_NAMES列表
            if not item.startswith("TEMP")  # 过滤非临时项
        ]
    return None  # 返回空值


def secure_chained_vars(default_cfg, new_cfg, vb):
    """
    # 核心功能 安全处理绑定参数更新
    # 参数说明:
    - default_cfg: 默认配置对象
    - new_cfg: 新配置字典
    - vb: 详细输出开关标志
    # 关键逻辑:
    1. 检测绑定参数依赖关系
    2. 自动更新关联参数
    3. 输出修改记录
    """
    # 获取默认配置的字典属性
    default_cfg_dict = default_cfg.__dict__  # 获取配置对象的属性字典

    # 初始化已修改参数列表
    altered_cv = []  # 存储被自动更新的绑定参数名称

    # 遍历所有配置项
    for key in default_cfg_dict:  # 遍历配置对象的所有属性
        # 检查是否为绑定参数并获取原始名
        is_chain, o_key = is_chained_key(key)  # 判断是否为绑定参数格式
        if not is_chain:
            continue  # 跳过非绑定参数

        # 跳过已在新配置中更新的参数
        if o_key in new_cfg:  # 如果新配置中已包含该参数
            continue  # 跳过自动更新

        # 获取绑定参数对象
        chain_var = getattr(default_cfg, key)  # 获取绑定参数对象
        need_reflesh = False  # 初始化刷新标志

        # 检查依赖项是否需要更新
        for chain_by_var in chain_var.chained_with:  # 遍历所有依赖参数
            if chain_by_var in new_cfg:  # 如果依赖参数被更新
                need_reflesh = True  # 标记需要刷新

        if not need_reflesh:
            continue  # 跳过无需更新的参数

        # 计算新值（根据参数类型选择处理方式）
        if isinstance(chain_var, ChainVarSimple):  # 如果是简单绑定参数
            replace_item = perform_varible_chain_react(
                cv=chain_var, cfg=default_cfg
            )  # 执行反应式更新
        else:
            replace_item = chain_var.chain_func(  # 执行函数式更新
                *[
                    getattr(default_cfg, v) for v in chain_var.chained_with
                ]  # 获取所有依赖参数值
            )

        # 获取原始值用于对比
        original_item = getattr(default_cfg, o_key)  # 获取原始参数值
        # 输出参数更新日志（靛蓝色）
        if vb:
            print靛(  # 跨平台彩色输出
                "[config] warning, %s is chained by %s, automatic modifying:"  # 参数依赖警告
                % (o_key, str(chain_var.chained_with)),  # 显示依赖关系
                original_item,  # 原始值
                "-->",  # 变更分隔符
                replace_item,  # 新值
            )
        # 更新配置值并记录变更
        setattr(default_cfg, o_key, replace_item)  # 更新配置参数
        altered_cv.append(o_key)  # 记录变更参数
    return altered_cv  # 返回被修改的参数列表


def check_config_relevence(json_data):
    """
    # 核心功能 检查配置相关性
    # 校验规则: 环境配置必须与环境路径匹配
    # 参数说明:
    - json_data: 加载的JSON配置数据
    # 安全检查: 确保MISSION配置与环境路径一致
    """
    # 获取环境名称和路径
    env_name = json_data["config.py->GlobalConfig"]["env_name"]  # 获取环境名称
    env_path = json_data["config.py->GlobalConfig"]["env_path"]  # 获取环境路径

    # 遍历所有配置项进行校验
    for key in json_data.keys():  # 遍历所有配置组
        if "MISSION" in key:  # 如果是任务相关配置
            # 断言检查环境路径是否匹配
            assert env_path in key, "configering wrong env!"  # 安全检查：路径一致性


def random_seed_warning(json_data):
    """
    # 核心功能 检查随机种子配置
    # 安全机制: 未指定随机种子时发出警告
    # 参数说明:
    - json_data: 包含seed字段的配置数据
    """
    # 检查是否提供随机种子
    if "seed" not in json_data:  # 如果配置中未指定随机种子
        from config import GlobalConfig as cfg  # 导入全局配置

        # 输出红色警告并等待5秒
        print亮红(
            "Random seed not given, using %d" % cfg.seed
        )  # 安全检查：随机种子缺失
        time.sleep(5)  # 给予用户确认时间


def prepare_tmp_folder():
    """
    # 核心功能 准备临时文件夹
    # 创建本地和全局临时目录
    # 包含子目录: GpuLock/PortFinder
    """

    # 定义本地临时目录初始化函数
    def init_dir(dir):
        # 如果目录不存在则创建
        if not os.path.exists(dir):  # 检查目录是否存在
            os.makedirs(dir)  # 创建目录

    # 设置本地和全局临时目录路径
    local_temp_folder = "./TEMP"  # 本地临时目录路径
    global_temp_folder = os.path.expanduser("~/HmapTemp")  # 全局临时目录路径
    # 创建必要目录结构
    init_dir(local_temp_folder)  # 创建本地临时目录
    init_dir(global_temp_folder + "/GpuLock")  # 创建GPU锁目录
    init_dir(global_temp_folder + "/PortFinder")  # 创建端口查找目录


def prepare_alg_tmp_folder(json_data):
    """
    # 核心功能 准备算法临时文件夹
    # 处理流程:
    1. 解析任务配置
    2. 创建临时算法目录
    3. 复制算法文件并设置只读
    """
    try:
        # 获取任务配置键值
        mission_key = [k for k in json_data.keys() if k.startswith("MISSION")][
            0
        ]  # 查找任务配置组
        TEAM_NAMES = json_data[mission_key]["TEAM_NAMES"]  # 获取团队名称列表

        for tname in TEAM_NAMES:  # 遍历所有团队名称
            if not tname.startswith("TEMP"):  # 跳过非临时目录 
                continue

            # 解析路径并查找算法源路径
            path = tname.split("->")[0].replace(".", "/")  # 提取路径部分
            trace_success = False
            max_depth = 5  # 最大查找深度

            # 最多向上查找5层目录
            for _ in range(max_depth):  # 目录向上追溯
                parent = os.path.relpath(path + "/..")  # 获取上级目录
                if os.path.basename(parent) == "ALGORITHM":  # 如果找到算法根目录
                    src_path = os.path.relpath(  # 计算相对路径
                        path,
                        start=os.path.relpath(parent + "/.."),  # 获取源路径
                    )
                    trace_success = True  # 标记查找成功
                    break  # 退出循环
                path = parent  # 继续向上查找

            if trace_success:  # 如果成功找到源路径
                # 导入文件操作模块
                import glob
                from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWRITE  # 导入权限常量

                # 定义只读文件处理函数
                def readonly_handler(func, path, execinfo):
                    try:
                        os.chmod(path, S_IWRITE)  # 修改文件权限为可写
                        func(path)  # 执行删除操作
                    except:
                        pass  # 忽略异常
                    return

                # 删除已存在目录并复制新文件
                rmtree(path, onerror=readonly_handler)  # 删除已有目录
                print亮绿(  # 输出绿色提示
                    f"[config] Copying mirror algorithm from {src_path} to {path}"  # 显示复制信息
                )
                copytree(src_path, path)  # 复制算法文件

                # 设置所有py文件为只读
                for f in glob.glob(path + "/**/*.py", recursive=True):  # 遍历所有py文件
                    os.chmod(
                        f, S_IREAD | S_IRGRP | S_IROTH
                    )  # 设置只读权限（设计意图：防止意外修改）
    except:
        # 异常处理并等待5秒
        print亮红(
            "[config] Errors occurs when executing prepare_alg_tmp_folder"
        )  # 输出红色错误信息
        time.sleep(5)  # 延迟退出
        return  # 返回空值


def register_machine_info(logdir):
    """
    # 核心功能 注册机器信息
    # 收集信息包括:
    - 主机IP
    - 实验UUID
    - 运行路径
    - 启动时间
    - Docker容器哈希
    # 设计意图: 用于实验追踪和复现实验环境
    """
    import json
    import subprocess
    import uuid

    from .network import get_host_ip  # 导入网络工具函数获取主机IP

    # 初始化信息字典
    info = {
        "HostIP": get_host_ip(),  # 获取主机IP地址（设计意图：记录运行环境网络位置）
        "ExpUUID": uuid.uuid1().hex,  # 生成实验唯一标识（设计意图：确保实验可追溯）
        "RunPath": os.getcwd(),  # 记录当前工作路径（设计意图：定位实验执行上下文）
        "StartDateTime": time.strftime(  # 记录启动时间（设计意图：时间戳便于日志分析）
            "%Y-%m-%d %H-%M-%S", time.localtime()
        ),
    }

    try:
        # 获取Docker容器哈希（设计意图：容器环境标识）
        info["DockerContainerHash"] = subprocess.getoutput(
            r'cat /proc/self/cgroup | grep -o -e "docker/.*"| head -n 1 |sed "s/docker\\/\\(.*\\)/\\1/" |cut -c1-12'
        )
    except:
        info["DockerContainerHash"] = "None"  # 异常处理（安全检查：非容器环境标记）

    # 写入信息到JSON文件（设计意图：持久化存储机器信息）
    with open("%s/info.json" % logdir, "w+") as f:
        json.dump(info, f, indent=4)
    return info  # 返回机器信息字典


def backup_files(files, logdir, jsonfile):
    """
    # 核心功能 备份配置文件
    # 支持文件和目录备份
    # 可选远程服务器同步
    # 安全机制: 确保配置文件版本可追溯
    """
    from config import GlobalConfig as cfg

    # 远程服务器同步处理（设计意图：跨环境配置备份）
    if cfg.remote_server_ops != "":
        remote_server_ops = cfg.remote_server_ops.replace(
            "LOCALFILE", jsonfile
        ).replace(
            "REMOTEFILE",
            time.strftime("%Y_%m_%d_%H_%M_%S__", time.localtime())
            + cfg.note
            + "__"
            + os.path.basename(jsonfile),
        )
        os.popen(remote_server_ops)

    # 遍历待备份文件列表
    for file in files:
        if os.path.isfile(file):  # 文件备份处理
            print绿(
                "[config] Backup File:", file
            )  # 绿色输出（设计意图：操作状态可视化）
            bkdir = "%s/backup_files/" % logdir  # 构建备份目录路径
            if not os.path.exists(bkdir):  # 创建备份目录（安全检查：目录存在性）
                os.makedirs(bkdir)
            copyfile(file, "%s/%s" % (bkdir, os.path.basename(file)))  # 执行文件复制
        else:  # 目录备份处理
            print亮绿(
                "[config] Backup Folder:", file
            )  # 亮绿输出（设计意图：操作状态可视化）
            assert os.path.isdir(file), ("cannot find", file)  # 安全检查：路径有效性
            copytree(  # 递归复制目录
                file,
                "%s/backup_files/%s" % (logdir, os.path.basename(file)),
                dirs_exist_ok=True,  # 允许目录覆盖
                ignore=ignore_patterns("__pycache__"),  # 忽略缓存目录
            )
    return


def check_experiment_log_path(logdir):
    """
    # 核心功能 检查实验日志路径冲突
    # 安全机制:
    - 已存在路径提示覆盖警告
    - 自动暂停等待用户确认
    # 返回值: 用户输入内容或None
    """
    res = None  # 初始化用户输入结果
    if os.path.exists(logdir):  # 如果日志目录已存在
        if os.path.exists(logdir + "test_stage"):  # 如果存在测试阶段标记文件
            return None  # 跳过检查直接返回
        # 输出红色警告信息（设计意图：醒目提示风险操作）
        print亮红("Current log path:", logdir)
        print亮红("Warning! you will overwrite old logs if continue!")
        print亮红("Pause for 1 seconds before continue (or press Enter to confirm!)")
        try:
            res = askChoice()  # 获取用户输入
            if res == "":  # 如果用户直接回车
                res = None  # 重置为None
        except func_timeout.exceptions.FunctionTimedOut:  # 超时异常处理
            res = None  # 超时后自动继续
    return res  # 返回用户输入结果


@func_timeout.func_set_timeout(1)  # 设置1秒超时装饰器
def askChoice():
    return input(">>")  # 获取用户输入（设计意图：简化交互流程）


def arg_summary(config_class, modify_dict={}, altered_cv=[]):
    """
    # 核心功能 配置摘要输出
    # 参数说明:
    - config_class: 配置类对象
    - modify_dict: 用户修改项字典
    - altered_cv: 自动更新项列表
    # 输出规范:
    - 绿色：默认值
    - 靛蓝：自动更新
    - 红色：用户修改
    # 设计意图: 可视化配置变更状态
    """
    for key in config_class.__dict__:  # 遍历配置类所有属性
        if "__" in key:  # 跳过特殊属性
            continue
        is_chain, _ = is_chained_key(key)  # 检查是否为绑定参数
        if is_chain:  # 跳过绑定参数
            continue
        # 根据配置状态选择输出颜色
        if (key not in modify_dict) or (modify_dict[key] is None):  # 默认值或自动更新
            if key not in altered_cv:  # 默认值
                print绿(
                    key.center(25), "-->", str(getattr(config_class, key))
                )  # 绿色输出
            else:  # 自动更新值
                print靛(
                    key.center(25), "-->", str(getattr(config_class, key))
                )  # 靛蓝输出
        else:  # 用户修改值
            print红(key.center(25), "-->", str(getattr(config_class, key)))  # 红色输出


def my_setattr(conf_class, key, new_value, vb):
    """
    # 核心功能 安全设置配置属性
    # 参数说明:
    - conf_class: 目标配置类
    - key: 配置项名称
    - new_value: 新值
    - vb: 是否输出日志
    # 类型检查机制:
    - 浮点型/布尔型/整型/字符串/列表/字典类型验证
    - 类型转换处理
    # 安全检查:
    - 属性存在性验证
    - 类型匹配校验
    """
    assert hasattr(conf_class, key), (  # 安全检查：属性存在性
        conf_class,
        "has no such config item: **%s**" % key,
    )
    setting_name = key  # 配置项名称别名
    replace_item = new_value  # 新值赋值
    original_item = getattr(conf_class, setting_name)  # 获取原始值
    if vb:  # 详细模式输出日志
        print绿(
            "[config] override %s:" % setting_name, original_item, "-->", replace_item
        )
    # 类型检查与转换逻辑
    if isinstance(original_item, float):  # 浮点型处理
        replace_item = float(replace_item)
    elif isinstance(original_item, bool):  # 布尔型处理
        if replace_item == "True":  # 字符串转布尔
            replace_item = True
        elif replace_item == "False":
            replace_item = False
        elif isinstance(replace_item, bool):  # 原生布尔值
            replace_item = replace_item
        else:  # 非法输入抛出异常
            assert False, ("enter True or False, but have:", replace_item)
    elif isinstance(original_item, int):  # 整型处理
        assert int(replace_item) == float(replace_item), (  # 确保不是浮点数
            "warning, this var **%s** has an int default, but given a float override!"
            % key
        )
    elif isinstance(original_item, str):  # 字符串直接赋值
        replace_item = replace_item
    elif isinstance(original_item, list):  # 列表类型检查
        assert isinstance(replace_item, list)
    elif isinstance(original_item, dict):  # 字典类型检查
        assert isinstance(replace_item, dict)
    else:  # 不支持的类型抛出异常
        assert False, "not support this type"
    setattr(conf_class, setting_name, replace_item)  # 设置新值
    return  # 返回空值


def find_all_conf():
    """
    # 核心功能 查找所有配置类
    # 实现原理:
    1. 递归搜索.py文件
    2. 解析ADD_TO_CONF_SYSTEM标记
    3. 收集配置类元信息
    # 返回值: 包含文件路径、类名等信息的字典列表
    """
    import glob  # 导入文件搜索模块

    # 递归搜索所有Python文件
    py_script_list = glob.glob("./**/*.py", recursive=True)  # 获取项目内所有.py文件路径

    conf_class_gather = []  # 初始化配置类信息收集列表

    # 遍历所有Python文件
    for python_file in py_script_list:  # 处理每个Python文件
        with open(python_file, encoding="UTF-8") as f:  # 以UTF-8编码打开文件
            lines = f.readlines()  # 读取文件所有行

        # 分析文件内容
        for line in lines:  # 遍历每一行代码
            if "ADD_TO_CONF_SYSTEM" not in line:  # 跳过不含标记的行
                continue
            if "class " not in line:  # 跳过非类定义行
                continue
            # 收集匹配的配置类信息
            conf_class_gather.append(
                {"line": line, "file": python_file}
            )  # 存储类定义行和文件路径

    # 辅助函数：提取字符串中间内容
    def getBetween(str, str1, str2):  # 从字符串中提取指定标记间的内容
        strOutput = str[str.find(str1) + len(str1) : str.find(str2)]  # 截取中间部分
        return strOutput  # 返回提取结果

    # 处理收集到的配置类信息
    for target in conf_class_gather:  # 处理每个收集到的类
        # 提取类名（设计意图：解析类定义语法）
        class_name = getBetween(target["line"], "class ", "(")  # 从类定义行提取类名
        target["class_name"] = class_name  # 存储类名

        # 转换文件路径为模块路径
        target["file"] = (
            target["file"].replace("/", ".").replace("..", "")
        )  # 转换为Python模块导入格式

        import importlib  # 导入动态导入模块

        # 动态加载配置类
        target["class"] = getattr(  # 获取类对象
            importlib.import_module(target["file"].replace(".py", "")),  # 动态导入模块
            class_name,  # 获取类引用
        )
    return conf_class_gather  # 返回配置类信息列表


def make_json(conf_list):
    """
    # 核心功能 生成配置JSON摘要
    # 参数说明:
    - conf_list: 配置类信息列表（来自find_all_conf）
    # 处理流程:
    1. 收集配置项
    2. 序列化处理
    3. 文件持久化
    # 输出文件: all_conf.json
    """
    import json  # 导入JSON处理模块

    out = {}  # 初始化输出字典

    # 处理每个配置类
    for conf in conf_list:  # 遍历所有配置类
        local_conf = {}  # 创建局部配置字典
        config_class = conf["class"]  # 获取配置类引用

        # 收集配置项
        for key in config_class.__dict__:  # 遍历类属性
            if "__" in key:  # 跳过特殊属性
                continue
            is_chain, _ = is_chained_key(key)  # 检查是否为绑定参数
            if is_chain:  # 跳过绑定参数
                continue

            # 获取属性值并尝试序列化
            item_to_be_serialize = getattr(config_class, key)  # 获取配置值
            try:
                json.dumps(item_to_be_serialize)  # 测试可序列化性
            except:
                # 标记不可序列化数据（设计意图：保留数据信息同时避免序列化错误）
                item_to_be_serialize = "[cannot be json]" + str(item_to_be_serialize)
            local_conf[key] = item_to_be_serialize  # 存储处理后的值

        # 存储模块配置
        out[conf["file"]] = local_conf  # 按文件路径组织输出

    # 写入JSON文件
    with open("all_conf.json", "w") as f:  # 创建/覆盖输出文件
        json.dump(out, f, indent=4)  # 写入格式化JSON数据
        print亮紫(
            "the conf summary is successfully saved to all_conf.json"
        )  # 输出完成提示


# 主程序入口
if __name__ == "__main__":
    conf_list = find_all_conf()  # 收集配置类信息
    res_json = make_json(conf_list)  # 生成JSON摘要文件
