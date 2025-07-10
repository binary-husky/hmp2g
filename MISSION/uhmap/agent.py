# 导入numpy库用于数组操作
import numpy as np

# 从同级目录下的actset_lookup模块导入agent_json2local_attrs映射表
from .actset_lookup import agent_json2local_attrs


# 定义Agent类，继承自object基类
class Agent(object):
    # 初始化方法，接收团队名称、团队ID、唯一标识符参数
    def __init__(self, team, team_id, uid) -> None:
        # 存储团队名称
        self.team = team
        # 存储团队ID
        self.team_id = team_id
        # 存储唯一标识符
        self.uid = uid
        # 将全局属性映射表绑定到实例
        self.attrs = agent_json2local_attrs
        # 根据属性映射初始化所有属性为None
        for attr_json, attr_agent in self.attrs:
            setattr(self, attr_agent, None)
        # 初始化3D位置数组（x,y,z），用NaN表示未初始化
        self.pos3d = np.array([np.nan, np.nan, np.nan])
        # 初始化2D位置数组（x,y），同样用NaN填充
        self.pos2d = np.array([np.nan, np.nan])

    # 更新代理属性的方法，接收包含新数据的字典
    def update_agent_attrs(self, dictionary):
        # 如果agent存活状态为False
        if not dictionary["agentAlive"]:
            self.alive = False  # 标记为非存活
        else:
            assert dictionary["valid"]  # 断言数据有效性
            # 遍历属性映射表更新对应属性值
            for attr_json, attr_agent in self.attrs:
                setattr(self, attr_agent, dictionary[attr_json])
            # 断言本地UID与远程数据一致
            assert self.uid == self.uid_remote
            # 更新3D位置数据（需确保self.location存在）
            self.pos3d = np.array(self.location)
            # 从3D位置生成2D位置
            self.pos2d = self.pos3d[:2]
            # 更新3D速度数据（需确保self.velocity存在）
            self.vel3d = np.array(self.velocity)
            # 从3D速度生成2D速度（此处可能应为self.velocity[:2]）
            self.vel2d = self.pos3d[:2]
            # 更新3D缩放数据（需确保self.scale3存在）
            self.scale3d = np.array(self.scale3)
            # 提取第一个缩放值作为通用缩放
            self.scale = self.scale3[0]
            # 提取旋转值的第一个元素作为偏航角
            self.yaw = self.rotation[0]
            # 标记为存活状态
            self.alive = True
            # 更新团队归属信息
            self.team = dictionary["agentTeam"]
