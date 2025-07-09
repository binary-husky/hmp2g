# 导入数学库用于三角函数计算
import math

# 导入彩色输出工具
from UTIL.colorful import *

# 从基础配置模块导入算法配置
from .foundation import AlgorithmConfig


# 策略共振配置类
class PolicyRsnConfig:
    resonance_start_at_update = 10  # 启动共振的更新次数阈值
    yita_min_prob = 0.15  # 共振概率下限（需≥1/动作数量）
    yita_max = 0.75  # 共振概率上限
    yita_inc_per_update = 0.0075  # 每次更新增量（500次达到0.75）
    freeze_critic = False  # 是否冻结critic网络

    yita_shift_method = "-sin"  # 变化模式（可选：'-cos','-sin','slow-inc'）
    yita_shift_cycle = 1000  # 周期性变化的周期长度


# 阶段计划器类
class StagePlanner:
    def __init__(self, mcv) -> None:
        # 初始化策略共振状态
        if AlgorithmConfig.policy_resonance:
            self.resonance_active = False  # 共振激活标志
            self.yita = 0  # 共振概率值
            self.yita_min_prob = PolicyRsnConfig.yita_min_prob  # 最小概率

        self.freeze_body = False  # 网络主体冻结标志
        self.update_cnt = 0  # 更新计数器
        self.mcv = mcv  # 监控控制器
        self.trainer = None  # 训练器引用

        # 初始化归一化等待计数器
        if AlgorithmConfig.wait_norm_stable:
            self.wait_norm_stable_cnt = 2  # 需要等待2次更新
        else:
            self.wait_norm_stable_cnt = 0  # 不需要等待
        return

    # 检查共振是否激活
    def is_resonance_active(self):
        return self.resonance_active

    # 检查网络是否冻结
    def is_body_freeze(self):
        return self.freeze_body

    # 获取当前共振概率
    def get_yita(self):
        return self.yita

    # 获取最小概率阈值
    def get_yita_min_prob(self):
        return PolicyRsnConfig.yita_min_prob

    # 检查是否可以执行训练
    def can_exec_trainning(self):
        if self.wait_norm_stable_cnt > 0:
            print亮绿(
                "waiting initial normalization stable, skip training!"
            )  # 彩色输出
            self.wait_norm_stable_cnt -= 1  # 减少等待计数
            return False  # 暂不训练
        else:
            return True  # 可以训练

    # 更新计划
    def update_plan(self):
        self.update_cnt += 1  # 更新计数器递增
        if AlgorithmConfig.policy_resonance:
            if self.resonance_active:
                self.when_pr_active()  # 处理共振激活状态
            elif not self.resonance_active:
                self.when_pr_inactive()  # 处理共振未激活状态
        return

    # 激活策略共振
    def activate_pr(self):
        self.resonance_active = True  # 激活标志
        self.freeze_body = True  # 冻结网络
        if PolicyRsnConfig.freeze_critic:  # 如果需要冻结critic
            self.trainer.freeze_body()  # 冻结训练器主体

    # 共振未激活时的处理
    def when_pr_inactive(self):
        assert not self.resonance_active  # 断言当前未激活
        # 检查是否达到启动阈值
        if PolicyRsnConfig.resonance_start_at_update >= 0:
            if self.update_cnt > PolicyRsnConfig.resonance_start_at_update:
                self.activate_pr()  # 达到阈值后激活共振

        # 日志记录
        pr = 1 if self.resonance_active else 0
        self.mcv.rec(pr, "resonance")  # 记录共振状态
        self.mcv.rec(self.yita, "self.yita")  # 记录概率值

    # 共振激活时的处理
    def when_pr_active(self):
        assert self.resonance_active  # 断言当前已激活
        self._update_yita()  # 更新共振概率

        # 日志记录
        pr = 1 if self.resonance_active else 0
        self.mcv.rec(pr, "resonance")  # 记录共振状态
        self.mcv.rec(self.yita, "self.yita")  # 记录概率值

    # 更新yita值的私有方法
    def _update_yita(self):
        """
        根据配置方法更新yita值
        调用频率：每次update_plan()时
        """
        # 余弦波模式
        if PolicyRsnConfig.yita_shift_method == "-cos":
            self.yita = PolicyRsnConfig.yita_max
            # 计算负余弦值并缩放
            t = (
                -math.cos(
                    2 * math.pi / PolicyRsnConfig.yita_shift_cycle * self.update_cnt
                )
                * PolicyRsnConfig.yita_max
            )
            if t <= 0:
                self.yita = 0  # 负值置零
            else:
                self.yita = t  # 有效值赋入
            print亮绿("yita update:", self.yita)  # 输出日志

        # 正弦波模式
        elif PolicyRsnConfig.yita_shift_method == "-sin":
            self.yita = PolicyRsnConfig.yita_max
            # 计算负正弦值并缩放
            t = (
                -math.sin(
                    2 * math.pi / PolicyRsnConfig.yita_shift_cycle * self.update_cnt
                )
                * PolicyRsnConfig.yita_max
            )
            if t <= 0:
                self.yita = 0
            else:
                self.yita = t
            print亮绿("yita update:", self.yita)

        # 缓慢递增模式
        elif PolicyRsnConfig.yita_shift_method == "slow-inc":
            self.yita += PolicyRsnConfig.yita_inc_per_update  # 累加增量
            # 达到上限后停止
            if self.yita > PolicyRsnConfig.yita_max:
                self.yita = PolicyRsnConfig.yita_max
            print亮绿("yita update:", self.yita)

        # 无效模式断言
        else:
            assert False  # 遇到未知模式报错
