from UTILS.tensor_ops import dir2rad
import numpy as np
pos = np.array([
    [0,0,0],
    [1,0,0],
    [1,1,0],
    [0,1,0],
])
class ppt_test():
    def __init__(self) -> None:
        pass

    def render(self, i):
        if not hasattr(self, 'threejs_bridge'):
            from VISUALIZE.mcom import mcom
            self.threejs_bridge = mcom(ip='127.0.0.1', port=12084, path='RECYCLE/v2d_logger/', digit=8, rapid_flush=False, draw_mode='Threejs')
            self.threejs_bridge.初始化3D()
            # self.threejs_bridge.set_style('star')
            self.threejs_bridge.set_style('grid')
            # self.threejs_bridge.set_style('grid3d')
            self.threejs_bridge.set_style('gray')
            self.threejs_bridge.geometry_rotate_scale_translate('box',   0, 0,       0,       3, 2, 1,         0, 0, 0)
            self.threejs_bridge.geometry_rotate_scale_translate('cone',  0, np.pi/2, 0,       1.2, 0.9, 0.9,   1.5,0,0.5) # x -> y -> z
            # for available geometries, see https://threejs.org/docs/index.html?q=Geometry
            # self.threejs_bridge.advanced_geometry_rotate_scale_translate('oct', 'OctahedronGeometry(1,0)',   0, 0, 0,  1,1,1, 0,0,0) # x -> y -> z
            # self.threejs_bridge.advanced_geometry_rotate_scale_translate('oct2', 'OctahedronGeometry(1,0)',   0, 0, 0,  1,1,1, 0,0,0) # x -> y -> z
            self.threejs_bridge.advanced_geometry_rotate_scale_translate('Tor', 'TorusGeometry(10,3,16,100 )',   0, 0, 0,  1,1,1, 0,0,0) # x -> y -> z
            self.threejs_bridge.advanced_geometry_rotate_scale_translate('Tor2', 'TorusGeometry(10,3,16,100 )',   0, 0, 0,  1,1,2, 0,0,0) # x -> y -> z
            
            self.timex = 0

        size = 0.1
        color = 'purple'
        i__ = i%4
        self.threejs_bridge.v2dx(
            'Tor|%d|%s|%.3f'%(0, color, size),  # 填入 ‘形状|几何体之ID标识|颜色|大小’即可
            pos[i__,0], pos[i__,1], 0,          # 三维位置，3/6dof
            ro_x=0, ro_y=0, ro_z=0, # 欧拉旋转变换，3/6dof
            opacity=1,              # 透明度，1为不透明
            label='',               # 显示标签，空白不显示
            label_color='white',    # 标签颜色
            track_n_frame=3,        # 是否显示轨迹（0代表否），轨迹由最新的track_n_frame次位置连接而成
            track_tension=0.1,      # 轨迹曲线的平滑度，0为不平滑，推荐不平滑
            track_color='green',    # 轨迹的颜色显示，输入js颜色名或者hex值均可
            )
        self.threejs_bridge.v2dx(
            'Tor2|%d|%s|%.3f'%(1, color, size),  # 填入 ‘形状|几何体之ID标识|颜色|大小’即可
            pos[i__,0], pos[i__,1], 0,          # 三维位置，3/6dof
            ro_x=0, ro_y=0, ro_z=0, # 欧拉旋转变换，3/6dof
            opacity=1,              # 透明度，1为不透明
            label='',               # 显示标签，空白不显示
            label_color='white',    # 标签颜色
            track_n_frame=3,        # 是否显示轨迹（0代表否），轨迹由最新的track_n_frame次位置连接而成
            track_tension=0.1,      # 轨迹曲线的平滑度，0为不平滑，推荐不平滑
            track_color='green',    # 轨迹的颜色显示，输入js颜色名或者hex值均可
            )

 
        self.threejs_bridge.v2d_show()
    
        self.timex += 1
        condition = (self.timex==12)


x = ppt_test()
for i in range(100):
    x.render(i)


import time
time.sleep(1000)