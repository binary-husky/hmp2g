import io  # 导入基础模块
import os
import pickle
import shutil
import traceback

import numpy as np  # 导入数值计算库
import torch

from ALGORITHM.common.rl_alg_base import RLAlgorithmBase  # 导入强化学习算法基类
from config import GlobalConfig  # 导入全局配置
from UTIL.colorful import *  # 导入彩色输出工具
from UTIL.tensor_ops import _2tensor, repeat_at  # 导入张量操作工具


class AlgorithmConfig:
    """
    # 核心功能 算法配置类
    # 设计意图: 支持从JSON配置文件动态注入配置参数
    # 安全机制: 保留默认配置防止缺失
    """

    # 强化学习核心参数
    gamma = 0.99  # 折扣因子
    tau = 0.95  # GAE参数
    train_traj_needed = 512  # 训练所需轨迹长度
    hete_n_alive_frontend = 1  # 活跃前端数量
    TakeRewardAsUnity = False  # 团队奖励模式开关
    use_normalization = True  # 归一化开关
    wait_norm_stable = True  # 归一化稳定等待
    add_prob_loss = False  # 概率损失开关
    n_focus_on = 2  # 关注智能体数量
    n_entity_placeholder = 11  # 实体占位符数量

    # 模型加载相关参数
    load_checkpoint = False  # 检查点加载开关
    load_specific_checkpoint = ""  # 特定检查点路径

    # PPO算法参数
    clip_param = 0.2  # 剪切参数
    ppo_epoch = 16  # PPO迭代轮数
    n_pieces_batch_division = 1  # 批次分割数量
    value_loss_coef = 0.1  # 价值损失系数
    entropy_coef = 0.05  # 熵系数
    max_grad_norm = 0.5  # 最大梯度范数
    lr = 1e-4  # 学习率

    # 内存优化参数
    prevent_batchsize_oom = False  # 防止OOM
    gamma_in_reward_forwarding = False  # 奖励转发中的gamma
    gamma_in_reward_forwarding_value = 0.99  # gamma具体值

    # 网络结构参数
    net_hdim = 24  # 隐藏层维度
    dual_conc = True  # 双流连接开关
    n_agent = "auto load, do not change"  # 自动加载智能体数量

    # 动态配置功能
    ConfigOnTheFly = True  # 是否启用动态配置

    # 异构网络参数
    hete_n_net_placeholder = 5  # 异构网络占位符数量
    hete_thread_align = False  # 线程对齐开关
    hete_same_prob = 0.25  # 同类型概率
    hete_lasted_n = 100  # 持续数量

    # 策略共振参数
    policy_resonance = False  # 策略共振开关

    # 可用动作参数
    use_avail_act = True  # 使用可用动作开关

    # 调试参数
    debug = False  # 调试模式开关
    ignore_test = False  # 忽略测试标志

    # 特殊功能参数
    type_agent_diff_lr = False  # 类型代理不同学习率
    hete_exclude_zero_wr = False  # 排除零权重
    policy_matrix_testing = False  # 策略矩阵测试
    test_which_cpk = 1  # 测试特定检查点
    type_sel_override = False  # 类型选择覆盖
    type_sel_override_list = []  # 类型选择覆盖列表
    allow_fast_test = True  # 允许快速测试


def str_array_to_num(str_arr):
    """
    # 核心功能 将字符串数组转换为数字标识符
    # 参数说明:
    - str_arr: 字符串列表
    # 返回值: 数字标识符列表
    # 设计意图: 为异构智能体类型分配唯一ID
    """
    out_arr = []  # 输出数组
    buffer = {}  # 类型缓冲区
    for str in str_arr:  # 遍历输入字符串
        if str not in buffer:  # 新类型处理
            buffer[str] = len(buffer)  # 分配唯一ID
        out_arr.append(buffer[str])  # 添加到输出
    return out_arr  # 返回转换结果


def itemgetter(*items):
    """
    # 核心功能 多项获取器（类似operator.itemgetter）
    # 参数说明:
    - items: 需要获取的项列表
    # 返回值: 获取器函数
    # 设计意图: 统一数据提取接口
    """

    def g(obj):
        return tuple(
            obj[item] if item in obj else None for item in items
        )  # 从对象中提取指定项

    return g  # 返回生成的获取函数


class CPU_Unpickler(pickle.Unpickler):
    """
    # 核心功能 自定义反序列化器
    # 设计意图: 支持CPU反序列化GPU模型
    # 安全机制: 重写find_class方法
    """

    def find_class(self, module, name):
        # 特殊处理torch.storage._load_from_bytes
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(
                io.BytesIO(b), map_location="cpu"
            )  # CPU反序列化
        else:
            return super().find_class(module, name)  # 默认处理方式


class ReinforceAlgorithmFoundation(RLAlgorithmBase):
    """
    # 核心功能 强化学习算法基础类
    # 继承关系: RLAlgorithmBase
    # 主要功能:
    1. 策略网络管理
    2. 异构智能体支持
    3. 动态配置更新
    4. 模型保存加载
    """

    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        """
        # 核心功能 初始化算法基础组件
        # 参数说明:
        - n_agent: 智能体数量
        - n_thread: 线程数量
        - space: 环境空间信息
        - mcv: 可视化日志器
        - team: 所属团队ID
        # 处理流程:
        1. 初始化基础组件
        2. 创建策略网络
        3. 加载检查点
        4. 初始化动态配置
        """
        from .hete_net import HeteNet  # 导入异构网络
        from .shell_env import (  # 动态导入环境包装器
            ActionConvertLegacy,
            ShellEnvWrapper,
        )

        super().__init__(n_agent, n_thread, space, mcv, team)  # 调用基类初始化
        AlgorithmConfig.n_agent = n_agent  # 更新配置中的智能体数量

        # 创建动作转换器
        self.action_converter = ActionConvertLegacy(  # 实例化动作转换器
            SELF_TEAM_ASSUME=team,  # 设置自身团队
            OPP_TEAM_ASSUME=(1 - team),  # 设置对手团队
            OPP_NUM_ASSUME=GlobalConfig.ScenarioConfig.N_AGENT_EACH_TEAM[
                1 - team
            ],  # 对手数量
        )
        n_actions = len(self.action_converter.dictionary_args)  # 获取动作空间维度

        # 创建环境包装器
        self.shell_env = ShellEnvWrapper(  # 包装环境
            n_agent,
            n_thread,
            space,
            mcv,
            self,
            AlgorithmConfig,
            GlobalConfig.ScenarioConfig,
            self.team,
        )
        # 获取观测维度
        if self.ScenarioConfig.EntityOriented:
            rawob_dim = self.ScenarioConfig.obs_vec_length  # 实体导向观测
        else:
            rawob_dim = space["obs_space"]["obs_shape"]  # 标准观测空间

        # 初始化阶段规划器
        from .stage_planner import StagePlanner  # 导入阶段规划模块

        self.stage_planner = StagePlanner(mcv=mcv)  # 创建阶段规划实例

        # 处理异构智能体类型
        agent_type_list = [
            a["type"] for a in GlobalConfig.ScenarioConfig.SubTaskConfig.agent_list
        ]  # 提取智能体类型
        self.HeteAgentType = str_array_to_num(agent_type_list)  # 转换为数字类型
        # 获取当前团队智能体类型
        hete_type = np.array(self.HeteAgentType)[
            self.ScenarioConfig.AGENT_ID_EACH_TEAM[team]
        ]  # 提取当前团队类型

        # 初始化策略网络
        self.policy = HeteNet(  # 创建异构策略网络
            rawob_dim=rawob_dim,
            n_action=n_actions,
            hete_type=hete_type,
            stage_planner=self.stage_planner,
        )
        self.policy = self.policy.to(self.device)  # 移动到指定设备

        # 初始化训练组件
        from .ppo import PPO  # 导入PPO算法
        from .trajectory import BatchTrajManager  # 导入轨迹管理器

        self.trainer = PPO(
            self.policy, ppo_config=AlgorithmConfig, mcv=mcv
        )  # 创建PPO训练器
        # 创建轨迹管理器
        self.traj_manager = BatchTrajManager(
            n_env=n_thread,
            traj_limit=int(GlobalConfig.ScenarioConfig.MaxEpisodeStep),  # 轨迹长度限制
            trainer_hook=self.trainer.train_on_traj,  # 训练回调
        )
        self.stage_planner.trainer = self.trainer  # 注册训练器到阶段规划器

        # 验证奖励类型
        self.check_reward_type(AlgorithmConfig)  # 检查奖励类型一致性

        # 加载检查点
        self.load_model(AlgorithmConfig)  # 加载预训练模型

        # 动态配置功能
        if AlgorithmConfig.ConfigOnTheFly:  # 如果启用动态配置
            self._create_config_fly()  # 创建动态配置文件

        # 策略矩阵测试初始化
        if AlgorithmConfig.policy_matrix_testing:  # 如果启用策略矩阵测试
            # 初始化测试统计容器
            self.threads_test_reward_sum = np.zeros(
                shape=(n_thread,), dtype=float
            )  # 线程奖励统计
            self.recent_test_rewards = []  # 近期奖励
            self.recent_test_wins = []  # 近期胜利
            self._unfi_frag_matrix_ = None  # 矩阵片段
            self.recent_test_hete_gp_summary = []  # 异构组摘要
            self.current_hete_gp_summary = None  # 当前异构组摘要

            # 创建可视化记录器
            from VISUALIZE.mcom import mcom  # 导入可视化模块

            self.mcv_matrix = mcom(  # 创建矩阵记录器
                path="%s/logger/matrix/" % GlobalConfig.logdir,  # 日志路径
                image_path="%s/matrix.jpg" % GlobalConfig.logdir,  # 图像路径
                draw_mode="Img",  # 绘图模式
                tag="[ppo.py]",
            )  # 标签
            self.mcv_matrix.rec_init(color="r")  # 初始化红色记录器

    def action_making(self, StateRecall, test_mode):
        """
        # 核心功能 生成动作决策
        # 参数说明:
        - StateRecall: 状态回忆对象
        - test_mode: 是否测试模式
        # 返回值: 动作和更新后的状态
        """
        assert "_hook_" not in StateRecall  # 确保钩子已清除

        # 提取核心观测数据
        (
            obs,
            threads_active_flag,
            avail_act,
            hete_pick,
            hete_type,
            gp_sel_summary,
            eprsn,
        ) = itemgetter(
            "obs",
            "threads_active_flag",
            "avail_act",
            "_hete_pick_",
            "_hete_type_",
            "_gp_pick_",
            "_EpRsn_",
        )(StateRecall)  # 从状态中提取必要数据

        # 观测数据有效性检查
        assert obs is not None, "Make sure obs is ok"  # 确保观测非空
        assert len(obs) == sum(threads_active_flag), "check batch size"  # 批量大小检查

        # 可用动作检查
        if AlgorithmConfig.use_avail_act:
            assert avail_act is not None  # 确保可用动作存在

        # 扩展策略共振标志
        eprsn = repeat_at(eprsn, -1, self.n_agent)  # 扩展到智能体维度
        thread_index = np.arange(self.n_thread)[threads_active_flag]  # 获取活跃线程索引

        # 生成动作（无梯度）
        with torch.no_grad():
            action, value, action_log_prob = self.policy.act(  # 执行策略网络
                obs=obs,
                test_mode=test_mode,
                avail_act=avail_act,
                hete_pick=hete_pick,
                hete_type=hete_type,
                gp_sel_summary=gp_sel_summary,
                thread_index=thread_index,
                eprsn=eprsn,
            )

        # 构建轨迹片段
        traj_framefrag = {
            "_SKIP_": ~threads_active_flag,  # 跳过标志
            "value": value,  # 价值估计
            "hete_pick": hete_pick,  # 异构选择
            "hete_type": hete_type,  # 异构类型
            "gp_sel_summary": gp_sel_summary,  # 组策略摘要
            "avail_act": avail_act,  # 可用动作
            "actionLogProb": action_log_prob,  # 动作对数概率
            "obs": obs,  # 观测数据
            "action": action,  # 动作输出
        }
        if avail_act is not None:
            traj_framefrag.update({"avail_act": avail_act})  # 添加可用动作

        # 注册回调钩子
        if not test_mode:  # 训练模式
            StateRecall["_hook_"] = self.commit_traj_frag(
                traj_framefrag, req_hook=True
            )  # 提交轨迹片段
        else:  # 测试模式
            if test_mode and AlgorithmConfig.policy_matrix_testing:
                StateRecall["_hook_"] = self.matrix_callback_special(
                    traj_framefrag
                )  # 特殊回调处理
        return action.copy(), StateRecall  # 返回动作和更新后的状态

    def matrix_callback_special(self, framefrag):
        """
        # 核心功能 特殊矩阵回调
        # 参数说明:
        - framefrag: 轨迹片段
        # 返回值: 回调函数
        """
        assert self._unfi_frag_matrix_ is None  # 确保无旧片段
        self._unfi_frag_matrix_ = framefrag  # 存储轨迹片段
        return self.matrix_callback_special_callback  # 返回回调函数

    def matrix_callback_special_callback(self, new_frag):
        """
        # 核心功能 特殊矩阵回调处理器
        # 参数说明:
        - new_frag: 新的轨迹片段
        # 处理流程:
        1. 更新奖励统计
        2. 处理完成标志
        3. 记录测试结果
        """
        fi_frag = self._unfi_frag_matrix_  # 获取原始片段
        self._unfi_frag_matrix_ = None  # 清除片段

        # 更新奖励统计
        reward = new_frag["reward"].copy()  # 复制奖励数据
        done = new_frag["done"].copy()  # 复制完成标志
        self.threads_test_reward_sum += reward * ~fi_frag["_SKIP_"]  # 累加有效奖励

        # 更新异构组摘要
        if not any(fi_frag["_SKIP_"]):  # 如果没有跳过项
            self.current_hete_gp_summary = fi_frag["gp_sel_summary"]  # 保存当前摘要

        # 处理完成事件
        if done.all():  # 如果全部完成
            # 更新近期奖励统计
            self.recent_test_rewards.extend(
                self.threads_test_reward_sum
            )  # 扩展奖励记录
            # 更新近期胜利统计
            self.recent_test_wins.extend(
                [q["team_ranking"][self.team] == 0 for q in new_frag["info"]]
            )  # 0表示第一名
            # 保存异构组摘要
            self.recent_test_hete_gp_summary.extend(
                self.current_hete_gp_summary
            )  # 扩展摘要记录
            self.threads_test_reward_sum *= 0  # 重置奖励统计
            self.current_hete_gp_summary = None  # 清除当前摘要
        return None  # 返回空值

    def interact_with_env(self, StateRecall):
        """
        # 核心功能 与环境交互接口
        # 设计意图: 标准MARL接口实现
        """
        return self.shell_env.interact_with_env(StateRecall)  # 调用环境包装器

    def interact_with_env_genuine(self, StateRecall):
        """
        # 核心功能 真实环境交互
        # 设计意图: 在环境准备完成后调用
        """
        return self.action_making(StateRecall, StateRecall["Test-Flag"])  # 生成动作

    def train(self):
        """
        # 核心功能 训练接口
        # 处理流程:
        1. 检查是否可执行训练
        2. 清理轨迹池
        3. 动态配置更新
        4. 阶段规划器更新
        """
        if self.traj_manager.can_exec_training():  # 如果可执行训练
            if self.stage_planner.can_exec_trainning():  # 如果允许训练
                self.traj_manager.train_and_clear_traj_pool()  # 训练并清理轨迹
            else:
                self.traj_manager.clear_traj_pool()  # 仅清理轨迹

            # 动态配置处理
            if AlgorithmConfig.ConfigOnTheFly:
                self._config_on_fly()  # 更新动态配置

            self.stage_planner.update_plan()  # 更新阶段计划

    def on_notify(self, message, **kargs):
        """
        # 核心功能 接收通知回调
        # 参数说明:
        - message: 通知消息
        - kargs: 扩展参数
        # 处理流程:
        1. 保存模型
        2. 注册检查点
        3. 矩阵测试可视化
        """
        win_rate = kargs["win_rate"]  # 提取胜率
        mean_reward = kargs["mean_reward"]  # 提取平均奖励

        # 保存模型
        path = self.save_model(  # 获取模型路径
            update_cnt=self.traj_manager.update_cnt,  # 更新计数
            info=str(kargs),  # 检查点信息
        )
        # 注册检查点
        self.policy.register_ckp(win_rate, path, mean_reward)  # 注册新检查点

        # 策略矩阵测试可视化
        if AlgorithmConfig.policy_matrix_testing:  # 如果启用测试
            from UTIL.data_struct import UniqueList  # 导入唯一列表工具

            # 转换摘要为列表
            recent_test_hete_gp_summary_ls = [
                q.tolist() for q in self.recent_test_hete_gp_summary
            ]  # 转为列表
            ulist = UniqueList(recent_test_hete_gp_summary_ls)  # 创建唯一列表

            # 可视化数据记录
            for u in ulist:  # 遍历唯一摘要
                feature = (
                    self.policy.ph_to_feature[u].squeeze().cpu().numpy().tolist()
                )  # 提取特征
                feature = "[%.2f,%.2f,%.2f]" % tuple(feature)  # 格式化特征
                mask = [u == uu for uu in recent_test_hete_gp_summary_ls]  # 构建掩码
                # 计算统计指标
                r = np.array(self.recent_test_rewards)[mask].mean()  # 计算奖励均值
                wr = np.array(self.recent_test_wins)[mask].mean()  # 计算胜率均值

                # 记录测试数据
                self.mcv_matrix.rec(self.policy.ckpg_input_cnt, "time")  # 记录时间
                self.mcv_matrix.rec(r, "r of=%s" % feature)  # 记录奖励
                self.mcv_matrix.rec(wr, "w of=%s" % feature)  # 记录胜率
                self.mcv_matrix.rec(sum(mask), "n of=%s" % feature)  # 记录样本数

            self.mcv_matrix.rec_show()  # 显示可视化结果

            self.recent_test_rewards = []  # 重置奖励统计
            self.recent_test_wins = []  # 重置胜率统计
            self.recent_test_hete_gp_summary = []  # 重置摘要统计

    def save_model(self, update_cnt, info=None):
        """
        # 核心功能 保存模型
        # 触发条件:
        1. 定期更新
        2. 收到通知
        3. 检测到人工指令
        """
        # 创建检查点目录
        if not os.path.exists("%s/history_cpt/" % GlobalConfig.logdir):  # 目录不存在
            os.makedirs("%s/history_cpt/" % GlobalConfig.logdir)  # 创建目录

        # 保存主模型
        pt_path = "%s/model.pt" % GlobalConfig.logdir  # 主模型路径
        print绿("saving model to %s" % pt_path)  # 绿色提示
        torch.save(
            {  # 保存模型
                "policy": self.policy.state_dict(),  # 策略参数
                "optimizer": self.trainer.optimizer.state_dict(),  # 优化器参数
            },
            pt_path,
        )

        # 保存历史模型
        info = (
            str(update_cnt) if info is None else "".join([str(update_cnt), "_", info])
        )  # 构建信息
        pt_path2 = "%s/history_cpt/model_%s.pt" % (
            GlobalConfig.logdir,
            info,
        )  # 历史路径
        shutil.copyfile(pt_path, pt_path2)  # 复制模型文件

        # 保存检查点信息
        with open(
            "%s/history_cpt/ckpg_info.pkl" % GlobalConfig.logdir, "wb"
        ) as f:  # 打开文件
            pickle.dump(
                (  # 序列化保存
                    self.policy.ckpg_info,  # 检查点信息
                    self.policy.ckpg_input_cnt,  # 输入计数
                    [
                        (n.feature, n.static, n.ready_to_go)
                        for n in self.policy._nets_flat_placeholder_
                    ],
                ),  # 网络状态
                f,
            )
        print绿("save_model fin")  # 提示保存完成
        return pt_path2  # 返回路径

    def find_ckp(self, feature):
        """
        # 核心功能 查找检查点
        # 参数说明:
        - feature: 特征标识
        # 返回值: 检查点字典
        """
        import glob  # 导入文件搜索模块

        list_ckp = glob.glob(
            "%s/history_cpt/*.pt" % GlobalConfig.logdir
        )  # 获取所有检查点
        ckp_dir = [ckp for ckp in list_ckp if str(feature[0]) in ckp][
            0
        ]  # 匹配特征的检查点
        cuda_n = "cpu" if "cpu" in self.device else self.device  # 设备适配
        cpt = torch.load(ckp_dir, map_location=cuda_n)  # 加载检查点
        # 提取网络参数
        return {
            k.replace("_nets_flat_placeholder_.0.", ""): v
            for k, v in cpt["policy"].items()
            if "_nets_flat_placeholder_.0." in k
        }  # 过滤并重命名参数

    def load_model(self, AlgorithmConfig):
        """
        # 核心功能 加载模型
        # 参数说明:
        - AlgorithmConfig: 算法配置对象
        """
        if AlgorithmConfig.load_checkpoint:  # 如果启用加载
            manual_dir = AlgorithmConfig.load_specific_checkpoint  # 获取手动路径
            # 构建检查点路径
            ckpt_dir = (
                "%s/model.pt" % GlobalConfig.logdir
                if manual_dir == ""
                else "%s/%s" % (GlobalConfig.logdir, manual_dir)
            )
            cuda_n = "cpu" if "cpu" in self.device else self.device  # 设备适配
            strict = True  # 严格模式
            # Windows路径安全检查
            if not platform.system() == "Linux":
                assert ":" not in ckpt_dir, "Windows OS不允许路径含冒号"
            # 加载检查点
            cpt = torch.load(ckpt_dir, map_location=cuda_n)  # 加载模型
            self.policy.load_state_dict(cpt["policy"], strict=strict)  # 加载策略参数
            # 加载优化器状态
            self.trainer.optimizer.load_state_dict(cpt["optimizer"])  # 加载优化器
            print黄("loaded checkpoint:", ckpt_dir)  # 黄色提示

            # 加载检查点信息
            if os.path.exists(
                "%s/history_cpt/ckpg_info.pkl" % GlobalConfig.logdir
            ):  # 如果存在
                with open(
                    "%s/history_cpt/ckpg_info.pkl" % GlobalConfig.logdir, "rb"
                ) as f:  # 打开文件
                    # 使用自定义反序列化
                    self.policy.ckpg_info, self.policy.ckpg_input_cnt, n_flags = (
                        CPU_Unpickler(f).load()
                    )  # 加载信息
                # 恢复网络状态
                for n, flags in zip(
                    self.policy._nets_flat_placeholder_, n_flags
                ):  # 遍历网络
                    n.feature = flags[0]  # 恢复特征
                    n.static = flags[1]  # 恢复静态标志
                    n.ready_to_go = flags[2]  # 恢复就绪状态
                    if n.feature != 1:  # 如果非默认特征
                        n.load_state_dict(
                            self.find_ckp(n.feature), strict=True
                        )  # 加载对应检查点
                # 更新特征映射
                self.policy.ph_to_feature = _2tensor(
                    np.array([n.feature for n in self.policy._nets_flat_placeholder_])
                )  # 构建特征张量
                print黄("loaded ckpg_info")  # 黄色提示
            else:
                print("Warning, past policy missing !!")  # 警告信息

    def process_framedata(self, traj_framedata):
        """
        # 核心功能 处理轨迹数据
        # 处理流程:
        1. 清理无关字段
        2. 数据格式转换
        3. 暂停线程过滤
        4. 提交轨迹数据
        """
        # 移除非数组字段
        items_to_pop = ["info", "Latest-Obs"]  # 待移除字段
        for k in items_to_pop:  # 遍历字段
            if k in traj_framedata:  # 如果存在
                traj_framedata.pop(k)  # 移除字段

        # 处理团队奖励模式
        if self.ScenarioConfig.RewardAsUnity:  # 如果团队奖励
            traj_framedata["reward"] = repeat_at(
                traj_framedata["reward"], insert_dim=-1, n_times=self.n_agent
            )  # 扩展奖励维度

        # 重命名完成标志
        traj_framedata["_DONE_"] = traj_framedata.pop("done")  # 重命名done字段
        # 处理终端观测
        traj_framedata["_TOBS_"] = (
            traj_framedata.pop("Terminal-Obs-Echo")
            if "Terminal-Obs-Echo" in traj_framedata
            else None
        )  # 移除终端观测

        # 过滤暂停线程
        traj_framedata = self.mask_paused_env(traj_framedata)  # 执行过滤
        # 提交轨迹数据
        self.traj_manager.feed_traj_framedata(traj_framedata)  # 提交到管理器

    def mask_paused_env(self, frag):
        """
        # 核心功能 过滤暂停线程
        # 参数说明:
        - frag: 轨迹片段
        # 返回值: 过滤后的片段
        """
        running = ~frag["_SKIP_"]  # 获取运行线程掩码
        if running.all():  # 如果全部运行
            return frag  # 返回原片段

        # 过滤非运行线程
        for key in frag:  # 遍历所有键
            # 处理数组型数据
            if (
                not key.startswith("_")
                and hasattr(frag[key], "__len__")
                and len(frag[key]) == self.n_thread
            ):
                frag[key] = frag[key][running]  # 应用掩码
        return frag  # 返回过滤后数据

    def _create_config_fly(self):
        """
        # 核心功能 创建动态配置文件
        # 安全检查: 硥文件不存在时创建
        """
        logdir = GlobalConfig.logdir  # 获取日志目录
        self.input_file_dir = "%s/cmd_io.txt" % logdir  # 构建输入文件路径
        if not os.path.exists(self.input_file_dir):  # 文件不存在
            # 创建并初始化文件
            with open(self.input_file_dir, "w+", encoding="utf8") as f:
                f.writelines(["# Write cmd at next line: ", "\n"])  # 写入初始内容

    def _config_on_fly(self):
        """
        # 核心功能 动态配置更新
        # 处理流程:
        1. 读取配置文件
        2. 执行配置指令
        3. 更新文件状态
        """
        if not os.path.exists(self.input_file_dir):
            return  # 文件不存在直接返回

        # 读取配置文件
        with open(self.input_file_dir, "r", encoding="utf8") as f:
            cmdlines = f.readlines()  # 读取所有行

        cmdlines_writeback = []  # 准备写回内容
        any_change = False  # 变化标志

        # 处理每条配置指令
        for cmdline in cmdlines:  # 遍历所有指令
            if cmdline.startswith("#") or cmdline in ["\n", " \n"]:  # 注释或空行
                cmdlines_writeback.append(cmdline)  # 直接写回
            else:
                any_change = True  # 标记变化
                try:
                    print亮绿(
                        "[foundation.py] ------- executing: %s ------" % cmdline
                    )  # 绿色提示
                    exec(cmdline)  # 执行指令
                    cmdlines_writeback.append(
                        "# [execute successfully]\t" + cmdline
                    )  # 记录成功
                except:
                    print红(traceback.format_exc())  # 输出异常信息
                    cmdlines_writeback.append(
                        "# [execute failed]\t" + cmdline
                    )  # 记录失败

        # 写回更新后的配置
        if any_change:  # 如果有变更
            with open(self.input_file_dir, "w+", encoding="utf8") as f:
                f.writelines(cmdlines_writeback)  # 写回更新后的配置
