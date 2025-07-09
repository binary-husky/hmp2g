# 导入PyTorch及数学库
import math

# PyTorch神经网络模块
# 神经网络函数模块（如激活函数）
# 优化器模块
# 数值计算库
import numpy as np

# 随机数生成器
# 数据采样工具
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

# 自定义颜色输出模块（用于控制台高亮）
from UTIL.colorful import *

# 自定义张量操作工具（张量转换、哈希、维度扩展）
# 全局配置参数
# GPU共享内存单元（用于进程间通信）


class TrajPoolSampler:
    """轨迹池采样器，用于PPO算法中的数据分批采样"""

    def __init__(self, n_div, traj_pool, flag, prevent_batchsize_oom=False, mcv=None):
        """
        初始化轨迹池采样器
        Args:
            n_div: 数据分批次数
            traj_pool: 轨迹池数据（包含多个轨迹对象）
            flag: 模式标识（仅支持'train'）
            prevent_batchsize_oom: OOM保护开关
            mcv: 监控变量记录器
        """
        # 存储分批次数
        self.n_pieces_batch_division = n_div
        # 存储OOM保护标志
        self.prevent_batchsize_oom = prevent_batchsize_oom
        # 存储监控记录器
        self.mcv = mcv

        # 当启用OOM保护时强制单一分批
        if self.prevent_batchsize_oom:
            assert self.n_pieces_batch_division == 1, "分批次数必须为1"

        # 初始化批量数量和容器状态
        self.num_batch = None  # 批次数目
        self.container = {}  # 数据存储容器
        self.warned = False  # 警告标记

        # 仅支持训练模式
        assert flag == "train", "仅支持训练模式"

        # 定义需要采集的数据字段及其重命名
        req_dict = [
            "hete_type",
            "gp_sel_summary",
            "avail_act",
            "obs",
            "action",
            "actionLogProb",
            "return",
            "reward",
            "hete_pick",
            "value",
        ]
        req_dict_rename = [
            "hete_type",
            "gp_sel_summary",
            "avail_act",
            "obs",
            "action",
            "actionLogProb",
            "return",
            "reward",
            "hete_pick",
            "state_value",
        ]
        return_rename = "return"
        value_rename = "state_value"
        advantage_rename = "advantage"

        # 处理带子键的字段（如obs>xxxx）
        for key_index, key in enumerate(req_dict):
            key_name = req_dict[key_index]
            key_rename = req_dict_rename[key_index]
            # 如果主键不存在，查找带子键的字段
            if not hasattr(traj_pool[0], key_name):
                real_key_list = [
                    real_key
                    for real_key in traj_pool[0].__dict__
                    if (key_name + ">" in real_key)
                ]
                assert len(real_key_list) > 0, (
                    "check variable provided!",
                    key,
                    key_index,
                )
                for real_key in real_key_list:
                    mainkey, subkey = real_key.split(">")
                    req_dict.append(real_key)
                    req_dict_rename.append(key_rename + ">" + subkey)

        # 检查数据一致性（初始批大小）
        self.big_batch_size = -1  # 初始化批大小

        # 加载轨迹数据到容器
        for key_index, key in enumerate(req_dict):
            key_name = req_dict[key_index]
            key_rename = req_dict_rename[key_index]
            # 跳过不存在的字段
            if not hasattr(traj_pool[0], key_name):
                continue

            # 合并所有轨迹的同类型数据
            set_item = np.concatenate(
                [getattr(traj, key_name) for traj in traj_pool], axis=0
            )

            # 验证数据维度一致性
            if not (
                self.big_batch_size == set_item.shape[0] or (self.big_batch_size < 0)
            ):
                print("数据维度不一致错误")
            assert self.big_batch_size == set_item.shape[0] or (
                self.big_batch_size < 0
            ), (key, key_index)
            self.big_batch_size = set_item.shape[0]  # 记录总样本数
            self.container[key_rename] = set_item  # 存储数据（指针赋值）

        # 计算并归一化优势值
        self.container[advantage_rename] = (
            self.container[return_rename] - self.container[value_rename]
        )
        self.container[advantage_rename] = (
            self.container[advantage_rename] - self.container[advantage_rename].mean()
        ) / (self.container[advantage_rename].std() + 1e-5)  # 标准化处理

        # 计算小批量尺寸
        self.mini_batch_size = math.ceil(
            self.big_batch_size / self.n_pieces_batch_division
        )

    def __len__(self):
        # 返回总批次数
        return self.n_pieces_batch_division

    def determine_max_n_sample(self):
        """动态调整最大样本数（防OOM）"""
        assert self.prevent_batchsize_oom
        # 首次初始化MaxSampleNum列表
        if not hasattr(TrajPoolSampler, "MaxSampleNum"):
            TrajPoolSampler.MaxSampleNum = [
                int(self.big_batch_size * (i + 1) / 50) for i in range(50)
            ]
            max_n_sample = self.big_batch_size
        # 正常情况更新最大样本数
        elif TrajPoolSampler.MaxSampleNum[-1] > 0:
            if self.big_batch_size > TrajPoolSampler.MaxSampleNum[-1]:
                TrajPoolSampler.MaxSampleNum.append(self.big_batch_size)
            max_n_sample = self.big_batch_size
        # OOM后使用历史安全值
        else:
            assert TrajPoolSampler.MaxSampleNum[-2] > 0
            max_n_sample = TrajPoolSampler.MaxSampleNum[-2]
        return max_n_sample

    def reset_and_get_iter(self):
        """生成数据迭代器"""
        if not self.prevent_batchsize_oom:
            # 正常采样模式
            self.sampler = BatchSampler(
                SubsetRandomSampler(range(self.big_batch_size)),
                self.mini_batch_size,
                drop_last=False,
            )
        else:
            # 防OOM模式
            max_n_sample = self.determine_max_n_sample()
            n_sample = min(self.big_batch_size, max_n_sample)

            # 首次提醒信息
            if not hasattr(self, "reminded"):
                self.reminded = True
                drop_percent = (
                    (self.big_batch_size - n_sample) / self.big_batch_size * 100
                )
                if self.mcv is not None:
                    self.mcv.rec(drop_percent, "drop percent")  # 记录丢弃比例
                # 输出警告信息
                if drop_percent > 20:
                    print_ = print亮红  # 红色警告
                    print_("丢弃%.1f%%样本.." % drop_percent)
                    assert False, "GPU内存溢出！"
                else:
                    print_ = print
                    print_("丢弃%.1f%%样本.." % drop_percent)
            # 创建采样器
            self.sampler = BatchSampler(
                SubsetRandomSampler(range(n_sample)), n_sample, drop_last=False
            )

        # 生成批次数据
        for indices in self.sampler:
            selected = {}
            # 提取索引对应数据
            for key in self.container:
                selected[key] = self.container[key][indices]
            # 重组嵌套字典结构
            for key in [key for key in selected if ">" in key]:
                mainkey, subkey = key.split(">")
                if mainkey not in selected:
                    selected[mainkey] = {}
                selected[mainkey][subkey] = selected[key]
                del selected[key]
            yield selected  # 返回生成器
