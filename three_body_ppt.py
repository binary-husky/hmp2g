from UTILS.tensor_ops import repeat_at, delta_matrix, dir2rad
from VISUALIZE.mcom import mcom
import numpy as np
import time
PI = np.pi; cos = np.cos; sin = np.sin
color = ['red','green','blue']
class 三体运动仿真():
    def __init__(self) -> None: # 初始条件 initial position
        self.位置 = 7*np.array([    
            [cos(0),    sin(0), 0],           [cos(2*PI/3),    sin(2*PI/3),  0],             [cos(-2*PI/3),      sin(-2*PI/3), 0],  ])  # the position of entities
        self.速度 = 0.3*np.array([ 
            [cos(0+PI/2), sin(+PI/2), 0],     [cos(2*PI/3+PI/2),    sin(2*PI/3+PI/2),  0],   [cos(-2*PI/3+PI/2),    sin(-2*PI/3+PI/2), 0],  ])  # the vel of entities
        self.质量 = np.array( [     1,               1,              1      ] )  # mass
        self.n体 = len(self.位置); assert len(self.位置) == len(self.质量)
        self.step = 0; self.仿真间隔 = 0.001;self.G = 4

    def physic_sim(self):
        质量m = repeat_at(self.质量, insert_dim=-1, n_times=len(self.质量))
        mass2mass = 质量m.T * 质量m
        delta = delta_matrix(self.位置)
        dis = np.linalg.norm(delta, axis=-1, keepdims=True)
        delta_unit = delta / (dis + 1e-16)
        gravity = (self.G*mass2mass)/(dis.squeeze(-1)**2 + 1e-10)
        gravity = delta_unit *  repeat_at(gravity, insert_dim=-1, n_times=3)
        for i in range(self.n体): gravity[i,i]=0
        gravity_acc = gravity.sum(1)/ repeat_at(self.质量, insert_dim=-1, n_times=3)
        self.新速度 = self.速度 + self.仿真间隔*gravity_acc 
        self.位置 = self.位置 + self.仿真间隔*(self.速度+self.新速度)/2
        self.速度 = self.新速度
        self.step += 1

    def sim_and_render(self):
        self.physic_sim()
        if self.step%50==0: self.render()

    def render(self):
        if not hasattr(self, '可视化桥'):
            self.可视化桥 = mcom(ip='127.0.0.1', port=12084, path='RECYCLE/v2d_logger/', digit=8, rapid_flush=False, draw_mode='Threejs')
            self.可视化桥.初始化3D()        # init 3d interface
            self.可视化桥.设置样式('grid'); self.可视化桥.设置样式('gray')
            self.可视化桥.形状之旋转缩放和平移('ball',  0, 0, 0,   1, 1, 1,   0, 0, 0)
        for index in range(self.n体):
            self.可视化桥.发送几何体('ball|%d|%s|%.3f'%(index, color[index],  0.1),
                self.位置[index, 0], self.位置[index, 1], self.位置[index, 2],
                ro_x=0, ro_y=0, ro_z=0, label_color='Black', opacity=1,
                label='(%.2f, %.2f, %.2f)'%(self.位置[index, 0],self.位置[index, 1],self.位置[index, 2]))
        self.可视化桥.结束关键帧()
        
x = 三体运动仿真()
for i in range(100000): x.sim_and_render()
time.sleep(1000)