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
            self.threejs_bridge.set_style('OrthographicCamera')
            self.threejs_bridge.geometry_rotate_scale_translate('box',   0, 0,       0,       3, 2, 1,         0, 0, 0)
            self.threejs_bridge.geometry_rotate_scale_translate('cone',  0, np.pi/2, 0,       1.2, 0.9, 0.9,   1.5,0,0.5) # x -> y -> z
            self.timex = 0

        size = 0.1
        color = 'red'
        i__ = i%4
        self.threejs_bridge.v2dx(
            'box|%d|%s|%.3f'%(0, color, size),
            pos[i__,0], pos[i__,1], 0,
            ro_x=0, ro_y=0, ro_z=0,
            label='', label_color='white', opacity=1, track_n_frame=3, 
            track_tension=0.1,
            track_color='black')


 
        self.threejs_bridge.v2d_show()
    
        self.timex += 1
        condition = (self.timex==12)


x = ppt_test()
for i in range(100):
    x.render(i)


import time
time.sleep(1000)