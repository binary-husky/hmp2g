# -*- coding: utf-8 -*-
# 基础库导入
import traceback

import numpy as np  # 数值计算
import torch
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 神经网络函数
import torch.optim as optim  # 优化器

from config import GlobalConfig as cfg  # 全局配置

# 数据采样工具
# 自定义模块
from UTIL.colorful import *  # 彩色控制台输出
from UTIL.gpu_share import GpuShareUnit  # GPU共享内存单元
from UTIL.tensor_ops import _2tensor  # 张量转换工具
from VISUALIZE.mcom import mcom  # 可视化工具

from .ppo_sampler import TrajPoolSampler  # 轨迹池采样器


class PPO:
    """PPO算法核心类，实现近端策略优化算法"""

    def __init__(self, policy_and_critic, ppo_config, mcv=None):
        """
        初始化PPO算法参数
        Args:
            policy_and_critic: 策略网络和价值网络对象
            ppo_config: PPO超参数配置对象
            mcv: 监控变量记录器（可选）
        """
        # 存储策略网络和价值网络
        self.policy_and_critic = policy_and_critic

        # PPO超参数配置
        self.clip_param = ppo_config.clip_param  # clip参数ε
        self.ppo_epoch = ppo_config.ppo_epoch  # 每次更新的迭代次数
        self.use_avail_act = ppo_config.ppo_epoch  # 是否使用可用动作
        self.n_pieces_batch_division = ppo_config.n_pieces_batch_division  # 批次分割数
        self.value_loss_coef = ppo_config.value_loss_coef  # 价值函数损失系数
        self.entropy_coef = ppo_config.entropy_coef  # 熵系数
        self.max_grad_norm = ppo_config.max_grad_norm  # 最大梯度范数（用于裁剪）
        self.add_prob_loss = ppo_config.add_prob_loss  # 是否添加概率损失
        self.prevent_batchsize_oom = ppo_config.prevent_batchsize_oom  # OOM防护开关
        self.lr = ppo_config.lr  # 学习率

        # 获取所有参数（带名称）
        self.all_parameter = list(policy_and_critic.named_parameters())
        # 提取参数张量
        self.parameter = [p for p_name, p in self.all_parameter]

        # 不同类型参数设置不同学习率
        if ppo_config.type_agent_diff_lr:
            others_parameters = [
                v for k, v in self.all_parameter if "_nets_flat" not in k
            ]
            # 构建学习率分组
            adam_lr_list = [
                {"params": list(n.parameters()), "lr": self.lr * n.lr_div}
                for n in policy_and_critic._nets_flat_placeholder_
            ] + [
                {
                    "params": list(policy_and_critic._critic_central.parameters()),
                    "lr": self.lr,
                }
            ]

            # 验证参数数量一致性
            assert sum([len(list(d["params"])) for d in adam_lr_list]) == len(
                self.all_parameter
            )
            self.optimizer = optim.Adam(adam_lr_list, lr=self.lr)
        else:
            # 使用统一学习率
            self.optimizer = optim.Adam(self.parameter, lr=self.lr)

        # 状态变量初始化
        self.g_update_delayer = 0  # 更新延迟计数器
        self.g_initial_value_loss = 0  # 初始价值损失
        self.mcv = mcv  # 监控记录器
        self.ppo_update_cnt = 0  # 更新计数
        self.batch_size_reminder = True  # 批量大小提醒
        self.trivial_dict = {}  # 临时指标字典

        # 强制批次分割数为1（当前实现限制）
        assert self.n_pieces_batch_division == 1

        # GPU共享内存单元（用于多进程通信）
        self.gpu_share_unit = GpuShareUnit(cfg.device, gpu_party=cfg.gpu_party)

        # 可视化记录器
        self.mcv2 = mcom(
            path="%s/logger/ppo/" % cfg.logdir,
            image_path="%s/detail_reward.jpg" % cfg.logdir,
            rapid_flush=True,
            draw_mode=cfg.draw_mode,
            tag="[ppo.py]",
        )
        self.mcv2.rec_init(color="g")  # 初始化记录器

    def freeze_body(self):
        """冻结网络主体参数（当前被禁用）"""
        self.freeze_body = True
        # 分离AT策略头参数
        self.at_parameter = [
            p for p_name, p in self.all_parameter if "AT_policy_head" in p_name
        ]
        self.at_optimizer = optim.Adam(self.at_parameter, lr=self.lr)
        # 分离CT相关参数
        self.ct_parameter = [p for p_name, p in self.all_parameter if "CT_" in p_name]
        self.ct_optimizer = optim.Adam(self.ct_parameter, lr=self.lr * 10.0)
        print("change train object")

    def train_on_traj(self, traj_pool, task):
        """训练主循环（带OOM防护）"""
        while True:
            try:
                with self.gpu_share_unit:  # GPU共享上下文
                    self.train_on_traj_(traj_pool, task)
                break  # 成功执行则退出循环
            except RuntimeError:  # 捕获运行时错误
                print(traceback.format_exc())  # 打印错误追踪
                if self.prevent_batchsize_oom:
                    # 调整最大样本数
                    if TrajPoolSampler.MaxSampleNum[-1] < 0:
                        TrajPoolSampler.MaxSampleNum.pop(-1)
                    assert TrajPoolSampler.MaxSampleNum[-1] > 0
                    TrajPoolSampler.MaxSampleNum[-1] = -1
                    print亮红("Insufficient gpu memory, using previous sample size !")
                else:
                    assert False  # 重新抛出错误
            torch.cuda.empty_cache()  # 清理缓存

    def log_reward_rich(self, traj_pool, mcv2):
        """记录丰富的奖励信息"""
        tags = {}
        for traj in traj_pool:
            # 计算轨迹总奖励
            traj.reward_sum = sum(traj.reward[:, 0])
            gp_list = traj.gp_sel_summary[0]

            # 分类奖励标签
            if (gp_list == 0).all():
                tag = "frontend"
                if tag not in tags:
                    tags[tag] = []
                tags[tag].append(traj.reward_sum)
            else:
                gp = max(gp_list)
                wr = self.policy_and_critic.ckpg_info[gp - 1]["win_rate"]
                tp = np.argmax(gp_list)
                tag = "tp:%d wr:%.2f" % (tp, wr)
                if tag not in tags:
                    tags[tag] = []
                tags[tag].append(traj.reward_sum)

        # 按标签排序并记录
        tags = dict(sorted(tags.items()))
        for k in tags:
            mcv2.rec(np.array(tags[k]).mean(), k)
        mcv2.rec_show()

    def train_on_traj_(self, traj_pool, task):
        """实际训练过程"""
        # 记录奖励信息
        self.log_reward_rich(traj_pool, self.mcv2)
        ppo_valid_percent_list = []

        # 创建轨迹采样器
        sampler = TrajPoolSampler(
            n_div=1,
            traj_pool=traj_pool,
            flag=task,
            prevent_batchsize_oom=self.prevent_batchsize_oom,
            mcv=self.mcv,
        )

        for e in range(self.ppo_epoch):
            # 获取数据迭代器
            sample_iter = sampler.reset_and_get_iter()
            self.optimizer.zero_grad()  # 梯度清零

            # 获取批次数据
            sample = next(sample_iter)

            # 构建计算图并更新网络
            loss_final, others = self.establish_pytorch_graph(task, sample, e)
            loss_final = loss_final * 0.5  # 缩放损失

            # 记录显存占用
            if e == 0:
                print(
                    "[PPO.py] Memory Allocated %.2f GB"
                    % (torch.cuda.memory_allocated() / 1073741824)
                )

            loss_final.backward()  # 反向传播

            # 日志记录
            ppo_valid_percent_list.append(others.pop("PPO valid percent").item())
            self.log_trivial(dictionary=others)
            others = None

            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.parameter, self.max_grad_norm)
            self.optimizer.step()  # 参数更新

            # 早期终止条件
            if ppo_valid_percent_list[-1] < 0.70:
                print亮黄("策略变化过大，提前终止迭代")
                break

        # 打印验证百分比
        print亮黄(np.array(ppo_valid_percent_list))
        self.log_trivial_finalize()  # 最终日志处理

        # 检查参数是否更新
        net_updated = [
            any([p.grad is not None for p in t.parameters()])
            for t in (self.policy_and_critic._nets_flat_placeholder_)
        ]

        self.optimizer.zero_grad(set_to_none=True)  # 彻底清零
        self.ppo_update_cnt += 1  # 更新计数

        # 更新网络计数器
        for updated, net in zip(
            net_updated, self.policy_and_critic._nets_flat_placeholder_
        ):
            if updated:
                net.update_cnt.data[0] = self.ppo_update_cnt

        self.policy_and_critic.on_update(self.ppo_update_cnt)  # 调用更新钩子
        torch.cuda.empty_cache()  # 清理缓存
        return self.ppo_update_cnt  # 返回更新次数

    def freeze_body(self):
        """冻结网络主体（被禁用的方法）"""
        assert False, "function forbidden"  # 禁止调用

    def log_trivial(self, dictionary):
        """记录临时指标"""
        for key in dictionary:
            # 自动初始化列表
            if key not in self.trivial_dict:
                self.trivial_dict[key] = []
            # 提取数值
            item = (
                dictionary[key].item()
                if hasattr(dictionary[key], "item")
                else dictionary[key]
            )
            self.trivial_dict[key].append(item)

    def log_trivial_finalize(self, print=True):
        """最终日志处理"""
        # 转换为numpy数组
        for key in self.trivial_dict:
            self.trivial_dict[key] = np.array(self.trivial_dict[key])

        # 构建打印信息
        print_buf = ["[ppo.py] "]
        for key in self.trivial_dict:
            # 计算平均值
            self.trivial_dict[key] = self.trivial_dict[key].mean()
            # 构建字符串
            print_buf.append(" %s:%.3f, " % (key, self.trivial_dict[key]))
            # 记录到监控器
            if self.mcv is not None:
                self.mcv.rec(self.trivial_dict[key], key)
        # 打印或记录
        if print:
            print紫("".join(print_buf))
        if self.mcv is not None:
            self.mcv.rec_show()
        self.trivial_dict = {}  # 清空字典

    def establish_pytorch_graph(self, flag, sample, n):
        """构建PyTorch计算图"""
        # 转换为张量
        obs = _2tensor(sample["obs"])
        advantage = _2tensor(sample["advantage"])
        action = _2tensor(sample["action"])
        oldPi_actionLogProb = _2tensor(sample["actionLogProb"])
        real_value = _2tensor(sample["return"])
        hete_pick = _2tensor(sample["hete_pick"])
        hete_type = _2tensor(sample["hete_type"])
        gp_sel_summary = _2tensor(sample["gp_sel_summary"])
        avail_act = _2tensor(sample["avail_act"]) if "avail_act" in sample else None

        batch_agent_size = advantage.shape[0] * advantage.shape[1]  # 计算总样本数

        assert flag == "train"  # 确保训练模式

        # 评估动作
        newPi_value, newPi_actionLogProb, entropy, probs, others = (
            self.policy_and_critic.evaluate_actions(
                obs=obs,
                eval_actions=action,
                test_mode=False,
                avail_act=avail_act,
                hete_pick=hete_pick,
                hete_type=hete_type,
                gp_sel_summary=gp_sel_summary,
            )
        )

        entropy_loss = entropy.mean()  # 计算熵损失

        # 概率损失（可选）
        n_actions = probs.shape[-1]
        if self.add_prob_loss:
            assert n_actions <= 15  # 动作空间限制
        penalty_prob_line = (1 / n_actions) * 0.12
        probs_loss = (
            penalty_prob_line - torch.clamp(probs, min=0, max=penalty_prob_line)
        ).mean()
        if not self.add_prob_loss:
            probs_loss = torch.zeros_like(probs_loss)

        # PPO核心公式
        E = newPi_actionLogProb - oldPi_actionLogProb  # 计算比率
        E_clip = torch.zeros_like(E)
        # 根据优势函数正负进行clip
        E_clip = torch.where(
            advantage > 0, torch.clamp(E, max=np.log(1.0 + self.clip_param)), E_clip
        )
        E_clip = torch.where(
            advantage < 0,
            torch.clamp(E, min=np.log(1.0 - self.clip_param), max=np.log(5)),
            E_clip,
        )
        ratio = torch.exp(E_clip)
        policy_loss = -(ratio * advantage).mean()  # 策略损失

        # 价值函数损失
        value_loss = 0.5 * F.mse_loss(real_value, newPi_value)

        # 组合总损失
        AT_net_loss = policy_loss - entropy_loss * self.entropy_coef  # 策略网络损失
        CT_net_loss = value_loss * 1.0  # 价值网络损失
        loss_final = AT_net_loss + CT_net_loss  # 总损失

        # 计算有效更新比例
        ppo_valid_percent = (E_clip == E).int().sum() / batch_agent_size

        # 非零掩码下的绝对损失
        nz_mask = real_value != 0
        value_loss_abs = (real_value[nz_mask] - newPi_value[nz_mask]).abs().mean()

        # 返回指标字典
        others = {
            "Value loss Abs": value_loss_abs,
            "PPO valid percent": ppo_valid_percent,
            "CT_net_loss": CT_net_loss,
            "AT_net_loss": AT_net_loss,
        }

        return loss_final, others
