# 导入必要的库
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from UTIL.tensor_ops import (
    Args2tensor,
    Args2tensor_Return2numpy,
    pt_inf,  # 自定义张量操作工具函数
)

"""
    网络初始化模块
    该模块包含多种将logits转换为动作的策略
"""


class Logit2Act(nn.Module):
    """
    动作选择策略网络

    提供多种基于分类分布的动作采样方法，支持：
    - 可学习的分布采样（ccategorical）
    - 基础分类采样
    - 可用动作掩码处理
    - 熵计算
    - 不同模式下的动作选择（训练/评估）
    """

    def __init__(self, *args, **kwargs):
        """构造函数"""
        super().__init__()
        # 注意：ccategorical属性可能在父类/子类中初始化
        # 它是一个可学习的分类分布生成器

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
        基于可学习分布的动作选择

        参数:
            logits_agent_cluster: [batch, agent, action] 原始动作分数
            eval_mode: 是否为评估模式
            greedy: 是否使用贪婪策略
            eval_actions: 评估时使用的固定动作
            avail_act: [batch, agent, action] 可用动作掩码
            eprsn: 采样参数（如温度）

        返回:
            act: 选择的动作
            actLogProbs: 动作对数概率
            distEntropy: 分布熵
            probs: 动作概率分布
        """
        # 应用可用动作掩码
        if avail_act is not None:
            logits_agent_cluster = torch.where(
                avail_act > 0, logits_agent_cluster, -pt_inf()
            )

        # 使用可学习分布生成器
        act_dist = self.ccategorical.feed_logits(logits_agent_cluster)

        # 动作选择逻辑
        if not greedy:
            # 训练模式使用采样，评估模式使用固定动作
            act = (
                self.ccategorical.sample(act_dist, eprsn)
                if not eval_mode
                else eval_actions
            )
        else:
            # 贪婪策略选择最大概率动作
            act = torch.argmax(act_dist.probs, axis=2)

        # 计算对数概率（策略梯度关键路径）
        actLogProbs = self._get_act_log_probs(act_dist, act)

        # 计算熵（用于探索）
        distEntropy = act_dist.entropy().mean(-1) if eval_mode else None

        return act, actLogProbs, distEntropy, act_dist.probs

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
        基础分类采样策略
        （与_rsn的区别在于使用默认Categorical分布）
        """
        # 处理可用动作掩码
        if avail_act is not None:
            logits_agent_cluster = torch.where(
                avail_act > 0, logits_agent_cluster, -pt_inf()
            )

        # 创建标准分类分布
        act_dist = Categorical(logits=logits_agent_cluster)

        # 动作选择逻辑
        if not greedy:
            act = act_dist.sample() if not eval_mode else eval_actions
        else:
            act = torch.argmax(act_dist.probs, axis=2)

        # 计算对数概率
        actLogProbs = self._get_act_log_probs(act_dist, act)

        # 计算熵
        distEntropy = act_dist.entropy().mean(-1) if eval_mode else None

        return act, actLogProbs, distEntropy, act_dist.probs

    def _logit2act_entropy_split(
        self,
        logits_agent_cluster,
        eval_mode,
        greedy,
        eval_actions=None,
        avail_act=None,
        **kwargs,
    ):
        """
        独立计算每个agent的熵的版本
        （与基础版本的区别在于不取平均）
        """
        if avail_act is not None:
            logits_agent_cluster = torch.where(
                avail_act > 0, logits_agent_cluster, -pt_inf()
            )
        act_dist = Categorical(logits=logits_agent_cluster)

        if not greedy:
            act = act_dist.sample() if not eval_mode else eval_actions
        else:
            act = torch.argmax(act_dist.probs, axis=2)

        actLogProbs = self._get_act_log_probs(act_dist, act)

        # 返回原始熵值（非平均）
        distEntropy = act_dist.entropy() if eval_mode else None

        return act, actLogProbs, distEntropy, act_dist.probs

    def _logit2act_rsn_entropy_split(
        self,
        logits_agent_cluster,
        eval_mode,
        greedy,
        eval_actions=None,
        avail_act=None,
        eprsn=None,
    ):
        """
        可学习分布 + 独立熵计算版本
        """
        if avail_act is not None:
            logits_agent_cluster = torch.where(
                avail_act > 0, logits_agent_cluster, -pt_inf()
            )
        act_dist = self.ccategorical.feed_logits(logits_agent_cluster)

        if not greedy:
            act = (
                self.ccategorical.sample(act_dist, eprsn)
                if not eval_mode
                else eval_actions
            )
        else:
            act = torch.argmax(act_dist.probs, axis=2)

        actLogProbs = self._get_act_log_probs(act_dist, act)
        distEntropy = act_dist.entropy() if eval_mode else None

        return act, actLogProbs, distEntropy, act_dist.probs

    @staticmethod
    def _get_act_log_probs(distribution, action):
        """
        计算动作对数概率的静态方法

        参数:
            distribution: 概率分布对象
            action: [batch, agent, 1] 选择的动作

        返回:
            [batch, agent, 1] 对数概率
        """
        # squeeze去掉最后一个维度，计算后再恢复
        return distribution.log_prob(action.squeeze(-1)).unsqueeze(-1)

    @Args2tensor_Return2numpy
    def act(self, *args, **kargs):
        """
        主动作选择接口
        将输入转换为tensor，输出转为numpy数组

        调用实际动作选择方法（需在子类实现）
        """
        return self._act(*args, **kargs)

    @Args2tensor
    def evaluate_actions(self, *args, **kargs):
        """
        动作评估接口
        将输入转换为tensor，保留输出为tensor

        用于评估模式下的动作价值计算
        """
        return self._act(*args, **kargs, eval_mode=True)
