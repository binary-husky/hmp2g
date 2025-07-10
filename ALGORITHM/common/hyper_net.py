# 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F

from UTIL.tensor_ops import my_view  # 自定义张量操作工具函数


class HyperNet(nn.Module):
    """
    超网络实现，用于动态生成神经网络权重

    参数:
        x_input_dim: 输入特征维度
        embed_dim: 嵌入层维度
        hyper_input_dim: 超网络输入维度
        use_resnet: 是否使用残差连接 (默认False)
    """

    def __init__(self, **kwargs):
        super(HyperNet, self).__init__()
        # 初始化参数保存到实例变量
        self.x_input_dim = kwargs["x_input_dim"]
        self.embed_dim = kwargs["embed_dim"]
        self.hyper_input_dim = kwargs["hyper_input_dim"]
        self.use_resnet = kwargs.get("use_resnet", False)

        # 构建第一层权重生成网络 (输入 -> 隐藏层 -> 权重矩阵)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.hyper_input_dim, self.embed_dim),  # 输入投影
            nn.ReLU(inplace=True),  # 激活函数
            nn.Linear(
                self.embed_dim, self.x_input_dim * self.embed_dim
            ),  # 输出权重矩阵
        )

        # 构建第一层偏置生成网络
        self.hyper_b1 = nn.Sequential(
            nn.Linear(self.hyper_input_dim, self.embed_dim)  # 直接输出偏置向量
        )

        # 构建第二层权重生成网络
        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.hyper_input_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim * self.embed_dim),
        )

        # 构建第二层偏置生成网络
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.hyper_input_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, 1),  # 输出单值偏置
        )

    def forward(self, x, hyper_x):
        """
        前向传播

        参数:
            x: 主网络输入 (shape: [thread/batch, agent, core])
            hyper_x: 超网络输入 (shape: [thread/batch, core])

        返回:
            输出张量 (shape: [thread/batch, agent, 1])
        """
        assert hyper_x.dim() == 3  # 确保超网络输入维度正确

        # 生成第一层权重并重塑形状
        w1 = my_view(self.hyper_w1(hyper_x), [0, 0, self.x_input_dim, self.embed_dim])
        # 生成第一层偏置并增加维度
        b1 = self.hyper_b1(hyper_x).unsqueeze(-2)

        # 生成第二层权重并重塑形状
        w2 = my_view(self.hyper_w2(hyper_x), [0, 0, self.embed_dim, self.embed_dim])
        # 生成第二层偏置并增加维度
        b2 = self.hyper_b2(hyper_x).unsqueeze(-2)

        # 输入张量增加维度以进行矩阵乘法
        x_ = x.unsqueeze(-2)
        # 第一层计算：线性变换 + ReLU激活
        hidden = F.relu(torch.matmul(x_, w1) + b1)
        # 第二层计算：线性变换
        y = torch.matmul(hidden, w2) + b2

        # 可选残差连接
        if self.use_resnet:
            return y.squeeze(-2) + x
        else:
            return y.squeeze(-2)


class MyHyperNet(nn.Module):
    """
    可扩展的超网络实现，支持多层动态权重生成

    参数:
        x_in_dim: 输入特征维度
        hyber_in_dim: 超网络输入维度
        layer_out_dims: 各层输出维度列表
        hyber_hid_dim: 超网络隐藏层维度
    """

    def __init__(self, x_in_dim, hyber_in_dim, layer_out_dims, hyber_hid_dim):
        super(MyHyperNet, self).__init__()
        # 初始化参数保存
        self.x_in_dim = x_in_dim
        self.layer_out_dims = layer_out_dims
        self.hyber_in_dim = hyber_in_dim
        self.hyber_hid_dim = hyber_hid_dim
        self.n_layer = len(layer_out_dims)  # 网络层数

        # 构建各层维度映射表 [(输入维,输出维), ...]
        self.layer_dim_dict = [(x_in_dim, layer_out_dims[0])] + [
            (d_in, d_out)
            for d_in, d_out in zip(layer_out_dims[:-1], layer_out_dims[1:])
        ]

        # 构建各层权重生成网络
        self.weight_each_layer = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hyber_in_dim, self.hyber_hid_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.hyber_hid_dim, d_in * d_out),
                )
                for d_in, d_out in self.layer_dim_dict
            ]
        )

        # 构建各层偏置生成网络
        self.bias_each_layer = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hyber_in_dim, self.hyber_hid_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.hyber_hid_dim, d_out),
                )
                for d_in, d_out in self.layer_dim_dict
            ]
        )

    def forward(self, x, hyper_x):
        """
        前向传播

        参数:
            x: 主网络输入 (shape: [thread/batch, agent, core])
            hyper_x: 超网络输入 (shape: [thread/batch, core])

        返回:
            输出张量 (shape: [thread/batch, agent, 1])
        """
        assert hyper_x.dim() == 3  # 确保超网络输入维度正确
        x = x.unsqueeze(-2)  # 增加维度以适配矩阵乘法

        # 逐层处理
        for i in range(self.n_layer):
            d_in, d_out = self.layer_dim_dict[i]
            # 生成当前层权重并重塑
            w = my_view(self.weight_each_layer[i](hyper_x), [0, 0, d_in, d_out])
            # 生成当前层偏置并增加维度
            b = self.bias_each_layer[i](hyper_x).unsqueeze(-2)

            # 线性变换
            x = torch.matmul(x, w) + b

            # 非最后一层使用ReLU激活
            if i != self.n_layer - 1:
                x = F.relu(x, inplace=True)

        return x.squeeze(-2)  # 移除额外维度返回结果
