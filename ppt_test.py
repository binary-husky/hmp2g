from UTILS.tensor_ops import dir2rad
import numpy as np
class ppt_test():
    def __init__(self) -> None:
        pass

    def render(self, i):
        if not hasattr(self, 'threejs_bridge'):
            from VISUALIZE.mcom import mcom
            self.threejs_bridge = mcom(ip='127.0.0.1', port=12084, path='RECYCLE/v2d_logger/', digit=8, rapid_flush=False, draw_mode='Threejs')
            self.threejs_bridge.v2d_init()
            # self.threejs_bridge.set_style('star')
            self.threejs_bridge.set_style('grid')
            # self.threejs_bridge.set_style('grid3d')
            # self.threejs_bridge.set_style('gray')
            self.threejs_bridge.geometry_rotate_scale_translate('box',   0, 0,       0,       3, 2, 1,         0, 0, 0)
            self.threejs_bridge.geometry_rotate_scale_translate('cone',  0, np.pi/2, 0,       1.2, 0.9, 0.9,   1.5,0,0.5) # x -> y -> z
            self.timex = 0

        for index in range(5):
            size = 0.025 if i<5 else 0.01
            color = 'red' if i<5 else 'blue'
            self.threejs_bridge.v2dx(
                'cone|%d|%s|%.3f'%(index, color, size),
                index/100+i/1, i/1, 0,
                ro_x=0, ro_y=0, ro_z=0,
                label='%d-%d'%(index,i), label_color='white', attack_range=0, opacity=1)


 
        self.threejs_bridge.v2d_show()
    
        self.timex += 1
        condition = (self.timex==12)


x = ppt_test()
for i in range(10):
    x.render(i)


import time
time.sleep(1000)