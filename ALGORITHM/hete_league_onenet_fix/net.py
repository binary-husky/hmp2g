import torch
import torch.nn as nn  # 导入神经网络模块
from torch.distributions.categorical import Categorical  # 导入分类分布

from ALGORITHM.common.attention import SimpleAttention  # 导入注意力机制
from ALGORITHM.common.hyper_net import HyperNet  # 导入超网络模块
from ALGORITHM.common.net_manifest import weights_init  # 导入权重初始化
from ALGORITHM.common.norm import DynamicNormFix  # 导入动态归一化
from UTIL.tensor_ops import (
    my_view,  # 导入张量操作工具
    pt_inf,  # 导入张量操作工具
)

from .ccategorical import CCategorical  # 导入自定义分类分布
from .foundation import AlgorithmConfig  # 导入算法配置

"""
网络初始化模块
"""


class Net(nn.Module):
    """
    # 核心功能 策略网络基础类（Actor）
    # 网络结构:
    1. 共享编码层（obs_encoder + attention）
    2. 策略头部（policy_head）
    3. 支持策略共振（Policy Resonance）
    """

    def __init__(self, rawob_dim, n_action, **kwargs):
        """
        # 核心功能 初始化策略网络
        # 参数说明:
        - rawob_dim: 观测数据维度
        - n_action: 动作空间大小
        - **kwargs: 扩展参数（包含stage_planner）
        # 网络组件:
        1. 共享编码层：obs_encoder, attention_layer
        2. 策略头部：hyper_net, policy_head
        """
        super().__init__()  # 初始化基类

        # 更新计数器（不可训练参数）
        self.update_cnt = nn.Parameter(
            torch.zeros(1, requires_grad=False, dtype=torch.long), requires_grad=False
        )

        # 从配置加载参数
        self.use_normalization = AlgorithmConfig.use_normalization  # 归一化开关
        self.use_policy_resonance = AlgorithmConfig.policy_resonance  # 策略共振开关
        self.n_action = n_action  # 动作空间维度

        # 策略共振初始化
        if self.use_policy_resonance:
            self.ccategorical = CCategorical(kwargs["stage_planner"])  # 创建共振分类器
            # 获取共振激活状态检测函数
            self.is_resonance_active = lambda: kwargs[
                "stage_planner"
            ].is_resonance_active()

        h_dim = AlgorithmConfig.net_hdim  # 隐藏层维度

        # 观测数据归一化层
        if self.use_normalization:
            self._batch_norm = DynamicNormFix(  # 动态归一化
                rawob_dim,  # 输入维度
                only_for_last_dim=True,  # 仅归一化最后维度
                exclude_one_hot=True,  # 排除one-hot编码
                exclude_nan=True,
            )  # 排除NaN值

        # 实体占位符数量
        n_entity = AlgorithmConfig.n_entity_placeholder

        # 共享编码层（Actor-Critic共享部分）
        self.obs_encoder = nn.Sequential(  # 观测编码器
            nn.Linear(rawob_dim, h_dim),  # 线性变换
            nn.ReLU(inplace=True),  # 激活函数
            nn.Linear(h_dim, h_dim),  # 第二层线性变换
        )
        self.attention_layer = SimpleAttention(h_dim=h_dim)  # 注意力机制层

        # 策略头部（Actor专用）
        _size = n_entity * h_dim  # 计算输入大小
        # 超网络初始化
        self.hyper_net = HyperNet(embed_dim=h_dim, hyper_input_dim=6, x_input_dim=_size)
        # 策略输出头
        self.policy_head = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),  # 隐藏层
            nn.Linear(h_dim, self.n_action),
        )  # 动作输出层

        self.is_recurrent = False  # 非循环网络
        self.apply(weights_init)  # 应用权重初始化
        return  # 返回空值

    def act(self, *args, **kargs):
        """
        # 核心功能 生成动作决策
        # 设计意图: 封装实际执行方法
        """
        return self._act(*args, **kargs)  # 调用实际动作生成方法

    def evaluate_actions(self, *args, **kargs):
        """
        # 核心功能 评估给定动作
        # 设计意图: 支持PPO中的动作评估
        """
        return self._act(*args, **kargs, eval_mode=True)  # 调用评估模式

    def _act(
        self,
        obs=None,
        test_mode=None,
        eval_mode=False,
        eval_actions=None,
        avail_act=None,
        agent_ids=None,
        eprsn=None,
        obs_hfeature=None,
    ):
        """
        # 核心功能 动作生成主方法
        # 处理流程:
        1. 检查网络就绪状态
        2. 数据预处理（归一化、NaN处理）
        3. 网络前向传播
        4. 动作选择
        # 关键参数:
        - obs: 观测数据
        - test_mode: 测试模式标志
        - avail_act: 可用动作掩码
        - obs_hfeature: 观测特征输入
        """
        assert self.ready_to_go  # 确保网络已就绪

        # 检测死亡智能体（存在NaN）
        mask_dead = torch.isnan(obs).any(-1)  # 检测NaN值

        if self.static:
            assert self.gp >= 1  # 静态模式下组编号必须有效

        # 准备评估动作
        eval_act = eval_actions if eval_mode else None  # 评估模式处理
        others = {}  # 其他输出容器

        # 观测归一化处理
        if self.use_normalization:
            if torch.isnan(obs).all():
                pass  # 全部无效跳过
            else:  # 正常观测
                # 执行归一化（冻结模式处理）
                obs = self._batch_norm(
                    obs, freeze=(eval_mode or test_mode or self.static)
                )
            obs_hfeature_norm = obs_hfeature  # 保存归一化特征

        # 替换死亡智能体观测值为0
        mask_dead = torch.isnan(obs).any(-1)  # 重新计算死亡掩码
        obs = torch.nan_to_num_(obs, 0)  # 将NaN替换为0

        # 共享编码层处理
        baec = self.obs_encoder(obs)  # 观测编码
        # 注意力机制应用（处理mask）
        baec = self.attention_layer(k=baec, q=baec, v=baec, mask=mask_dead)

        # 筟网络处理
        at_bac = my_view(baec, [0, 0, -1])  # 张量视图转换
        # 超网络应用（输入特征拼接）
        at_bac_hn = self.hyper_net(at_bac, hyper_x=obs_hfeature_norm)

        # 策略头生成logits
        logits = self.policy_head(at_bac_hn)  # 生成动作logits

        # 动作选择器选择
        logit2act = (
            self._logit2act_rsn
            if self.use_policy_resonance and self.is_resonance_active()
            else self._logit2act
        )  # 根据配置选择动作选择方法

        # 执行动作选择
        act, actLogProbs, distEntropy, probs = logit2act(
            logits,
            eval_mode=eval_mode,
            greedy=(test_mode or self.static),  # 是否贪婪选择
            eval_actions=eval_act,  # 评估模式动作
            avail_act=avail_act,  # 可用动作掩码
            eprsn=eprsn,
        )  # 策略共振参数

        # 返回值处理
        if not eval_mode:
            return act, "vph", actLogProbs  # 训练模式返回
        else:
            return "vph", actLogProbs, distEntropy, probs, others  # 评估模式返回

    def _logit2act_rsn(
        self,
        logits_agent_cluster,
        eval_mode,
        greedy,
        eval_actions=None,
        avail_act=None,
        eprsn=None,
    ):
        """
        # 核心功能 带策略共振的动作选择
        # 参数说明:
        - logits_agent_cluster: 动作logits
        - greedy: 是否贪婪选择
        # 返回值: 动作、log概率、熵、概率分布
        """
        # 可用动作过滤
        if avail_act is not None:
            logits_agent_cluster = torch.where(
                avail_act > 0, logits_agent_cluster, -pt_inf()
            )  # 过滤不可用动作

        # 注入策略共振信息
        act_dist = self.ccategorical.feed_logits(logits_agent_cluster)  # 生成动作分布

        # 动作采样逻辑
        if not greedy:  # 非贪婪模式
            act = (
                self.ccategorical.sample(act_dist, eprsn)
                if not eval_mode
                else eval_actions
            )  # 采样或使用给定动作
        else:  # 贪婪模式
            act = torch.argmax(act_dist.probs, axis=2)  # 选择最大概率动作

        # 计算动作log概率
        actLogProbs = self._get_act_log_probs(act_dist, act)  # 获取log概率
        # 计算分布熵
        distEntropy = act_dist.entropy().mean(-1) if eval_mode else None  # 分布熵
        return act, actLogProbs, distEntropy, act_dist.probs  # 返回结果

    def _logit2act(
        self,
        logits_agent_cluster,
        eval_mode,
        greedy,
        eval_actions=None,
        avail_act=None,
        **kwargs,
    ):
        """
        # 核心功能 普通分类动作选择
        # 参数说明:
        - logits_agent_cluster: 动作logits
        - greedy: 是否贪婪模式
        """
        # 可用动作过滤
        if avail_act is not None:
            logits_agent_cluster = torch.where(
                avail_act > 0, logits_agent_cluster, -pt_inf()
            )  # 过滤不可用动作

        # 创建标准分类分布
        act_dist = Categorical(logits=logits_agent_cluster)  # 生成动作分布

        # 动作采样逻辑
        if not greedy:  # 非贪婪模式
            act = (
                act_dist.sample() if not eval_mode else eval_actions
            )  # 采样或使用给定动作
        else:  # 贪婪模式
            act = torch.argmax(act_dist.probs, axis=2)  # 选择最大概率动作

        # 计算log概率
        actLogProbs = self._get_act_log_probs(act_dist, act)  # 获取log概率
        # 计算分布熵
        distEntropy = act_dist.entropy().mean(-1) if eval_mode else None  # 分布熵
        return act, actLogProbs, distEntropy, act_dist.probs  # 返回结果

    @staticmethod
    def _get_act_log_probs(distribution, action):
        """
        # 核心功能 获取动作log概率
        # 参数说明:
        - distribution: 动作分布
        - action: 选择的动作
        # 返回值: log概率
        """
        return distribution.log_prob(action.squeeze(-1)).unsqueeze(
            -1
        )  # 计算log概率并调整形状


class NetCentralCritic(nn.Module):
    def __init__(self, rawob_dim, n_action, **kwargs):
        """
        # 核心功能 集中式评论家网络（Critic）
        # 网络结构:
        1. 共享编码层（obs_encoder + attention）
        2. 价值估计分支（ct_encoder + ct_attention + get_value）
        # 参数说明:
        - rawob_dim: 原始观测数据维度
        - n_action: 动作空间大小
        - **kwargs: 扩展参数（包含stage_planner）
        """
        super().__init__()  # 初始化基类

        # 更新计数器（只读）
        self.update_cnt = nn.Parameter(
            torch.zeros(1, requires_grad=False, dtype=torch.long), requires_grad=False
        )  # 用于跟踪模型更新次数

        # 从配置加载参数
        self.use_normalization = AlgorithmConfig.use_normalization  # 归一化开关
        self.use_policy_resonance = AlgorithmConfig.policy_resonance  # 策略共振开关
        self.n_action = n_action  # 动作空间维度

        # 策略共振初始化
        if self.use_policy_resonance:
            # 创建共振分类器
            self.ccategorical = CCategorical(kwargs["stage_planner"])
            # 获取共振激活状态检测函数
            self.is_resonance_active = lambda: kwargs[
                "stage_planner"
            ].is_resonance_active()

        h_dim = AlgorithmConfig.net_hdim  # 隐藏层维度

        # 观测数据归一化层
        if self.use_normalization:
            self._batch_norm = DynamicNormFix(  # 动态归一化
                rawob_dim,  # 输入维度
                only_for_last_dim=True,  # 仅归一化最后维度
                exclude_one_hot=True,  # 排除one-hot编码
                exclude_nan=True,  # 排除NaN值
            )

        # 实体占位符数量
        n_entity = AlgorithmConfig.n_entity_placeholder

        # 共享编码层（Actor-Critic共享部分）
        self.obs_encoder = nn.Sequential(  # 观测编码器
            nn.Linear(rawob_dim, h_dim),  # 线性变换
            nn.ReLU(inplace=True),  # 激活函数
            nn.Linear(h_dim, h_dim),  # 第二层线性变换
        )
        self.attention_layer = SimpleAttention(h_dim=h_dim)  # 注意力机制层

        # 价值估计分支（Critic专用）
        _size = n_entity * h_dim  # 计算输入大小
        # 超网络初始化
        self.hyper_net = HyperNet(  # 超网络
            embed_dim=h_dim,  # 嵌入维度
            hyper_input_dim=6,  # 超输入维度
            x_input_dim=_size,
        )  # 主输入维度

        # 价值编码层
        self.ct_encoder = nn.Sequential(  # 价值编码器
            nn.Linear(h_dim, h_dim),  # 线性变换
            nn.ReLU(inplace=True),  # 激活函数
            nn.Linear(h_dim, h_dim),  # 第二层线性变换
        )

        # 价值注意力层
        self.ct_attention_layer = SimpleAttention(h_dim=h_dim)  # 注意力机制层

        # 价值输出头
        self.get_value = nn.Sequential(  # 价值估计头
            nn.Linear(h_dim, h_dim),  # 线性变换
            nn.ReLU(inplace=True),  # 激活函数
            nn.Linear(h_dim, 1),  # 输出价值（单值）
        )

        self.is_recurrent = False  # 非循环网络
        self.apply(weights_init)  # 应用权重初始化
        return  # 返回空值

    def estimate_state(
        self,
        obs=None,
        test_mode=None,
        eval_mode=False,
        eval_actions=None,
        avail_act=None,
        agent_ids=None,
        eprsn=None,
        obs_hfeature_critic=None,
    ):
        """
        # 核心功能 状态价值估计主方法
        # 处理流程:
        1. 数据预处理（归一化、NaN处理）
        2. 共享编码层处理
        3. 价值分支处理
        4. 生成价值估计
        """
        # 观测归一化处理
        if self.use_normalization:
            if torch.isnan(obs).all():
                pass  # 全部无效跳过
            else:  # 正常观测
                # 执行归一化（冻结模式处理）
                obs = self._batch_norm(
                    obs,
                    freeze=(eval_mode or test_mode),  # 冻结评估/测试模式
                )
            obs_hfeature_norm = obs_hfeature_critic  # 保存归一化特征

        # 替换死亡智能体观测值为0
        mask_dead = torch.isnan(obs).any(-1)  # 检测死亡智能体
        obs = torch.nan_to_num_(obs, 0)  # 将NaN替换为0（设计意图：处理无效观测）

        # 共享编码层处理
        baec = self.obs_encoder(obs)  # 观测编码
        # 注意力机制应用（处理mask）
        baec = self.attention_layer(k=baec, q=baec, v=baec, mask=mask_dead)

        # 价值估计分支处理
        ct_bac = my_view(baec, [0, 0, -1])  # 张量视图转换
        # 超网络应用（输入特征拼接）
        ct_bac_hn = self.hyper_net(ct_bac, hyper_x=obs_hfeature_norm)

        # 价值编码层
        ct_bac = self.ct_encoder(ct_bac_hn)  # 编码价值特征

        # 价值注意力机制
        ct_bac = self.ct_attention_layer(k=ct_bac, q=ct_bac, v=ct_bac)  # 注意力处理

        # 价值输出
        value = self.get_value(ct_bac)  # 生成最终价值估计
        return value  # 返回价值估计结果
