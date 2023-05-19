import os, fnmatch, matplotlib, time, copy, json
import numpy as np
from functools import lru_cache
from config import GlobalConfig
# 设置matplotlib正常显示中文和负号
# matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
# matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
ClassicPlotFigIndex = 1
AdvancePlotFigIndex = 2
class rec_family(object):
    def __init__(self, colorC=None, draw_mode='Native', image_path=None, figsize=None, smooth_level=None, rec_exclude=[], **kwargs):
        # the list of vars' name (with order), string
        self.name_list = []
        # the list of vars' value sequence (with order), float
        self.line_list = []
        # the list of vars' time sequence (with order), float
        self.time_list = []
        # the list of line plotting handles
        self.classic_plot_handle = []
        self.advance_plot_handle = []
        # subplot list
        self.classic_subplots = {}
        self.advance_subplots = {}
        # working figure handle
        self.classic_fig_handle = None
        self.default_fig_spp = None
        self.advance_fig_handle = None
        self.re_plot_timer = time.time()
        self.re_plot_time_lim = 60
        # recent time
        self.current_time = None
        self.time_index = None
        self.smooth_level = smooth_level
        self.figsize_given = figsize
        self.colorC = 'k' if colorC is None else colorC
        self.Working_path = 'Testing-beta'
        self.image_num = -1
        self.draw_mode = draw_mode
        self.rec_exclude = rec_exclude
        self.vis_95percent = True
        self.enable_percentile_clamp = True
        logdir = GlobalConfig.logdir
        self.plt = None
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        if self.draw_mode == 'Web':
            import matplotlib.pyplot as plt, mpld3
            self.html_to_write = '%s/html.html'%logdir
            self.plt = plt; self.mpld3 = mpld3
        elif self.draw_mode =='Native':
            import matplotlib.pyplot as plt
            plt.ion()
            self.plt = plt
        elif self.draw_mode =='Img':
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            self.plt = plt
            self.img_to_write = '%s/rec.jpg'%logdir
            self.img_to_write2 = '%s/rec.jpeg'%logdir
            if image_path is not None:
                self.img_to_write = image_path
                self.img_to_write2 = image_path+'.jpeg'
        else:
            assert False

    def rec_get(self):
        assert self.draw_mode =='Img'
        return json.dumps({
            'name_list': self.name_list,
            'line_list': self.line_list,
            'time_list': self.time_list,
        })

    def rec_init(self, colorC=None):
        if colorC is not None: self.colorC = colorC
        return
    
    @lru_cache(500)
    def match_exclude(self, name):
        for n in self.rec_exclude:
            if fnmatch.fnmatch(name, n): return True
        return False

    @lru_cache(500)
    def get_index(self, name):
        return self.name_list.index(name)

    def rec(self, var, name):
        if self.match_exclude(name):
            # if var is backlisted
            return
        if name in self.name_list:
            # if var is already known, skip
            pass
        else:
            # if var is new, prepare lists
            self.name_list.append(name)
            self.line_list.append([])  #新建一个列表
            self.time_list.append([])
            self.classic_plot_handle.append(None)
            self.advance_plot_handle.append(None)
        # get the index of the var
        index = self.get_index(name)
        if name=='time': 
            # special var: time
            self.current_time = var
            if self.time_index is None:
                self.time_index = index
                self.handle_all_missing_time()
            else:
                assert self.time_index == index
        else:
            # normal vars: if time is available, add it
            if self.time_index is not None:
                if len(self.line_list[index]) != len(self.time_list[index]):
                    self.handle_missing_time(self.line_list[index], self.time_list[index])
                self.time_list[index].append(self.current_time)
        # finally, add var value
        self.line_list[index].append(var)

    def handle_all_missing_time(self):
        for name in self.name_list:
            if name=='time': continue
            index = self.get_index(name)
            if len(self.line_list[index]) != len(self.time_list[index]):
                self.handle_missing_time(self.line_list[index], self.time_list[index])

    def handle_missing_time(self, line_arr, time_arr):
        assert len(line_arr) > len(time_arr)
        for i in range(len(line_arr) - len(time_arr)):
            time_arr.append(self.current_time - i - 1)
    
    def get_figure_size(self, image_num, baseline = 10):
        if self.figsize_given is None:
            expand_ratio = max((image_num - baseline)/4, 1)
            return (12*expand_ratio, 6*expand_ratio)
        else:
            return self.figsize_given
            
    def reflesh_figure(self, draw_advance_fig):
        if True:
            self.classic_subplots = {}
            if self.classic_fig_handle is not None: 
                self.classic_fig_handle.clf()
                # self.classic_fig_handle.close()
                self.classic_fig_handle = None
            for q, handle in enumerate(self.classic_plot_handle): 
                self.classic_plot_handle[q] = None

        if draw_advance_fig: 
            self.advance_subplots = {}
            if self.advance_fig_handle is not None: 
                self.advance_fig_handle.clf()
                # self.advance_fig_handle.close()
                self.advance_fig_handle = None
            for q, handle in enumerate(self.advance_plot_handle): 
                self.advance_plot_handle[q] = None

    def rec_show(self):
        # the number of total classic_subplots | 一共有多少条曲线
        image_num = len(self.line_list)
        # enable advanced plot | 是否启动高级曲线绘制
        draw_advance_fig = False
        # enable advanced plot when we have 'of=' in any value key
        for name in self.name_list:
            if 'of=' in name: draw_advance_fig = True
        # check whether the time var exists | 检查是否有时间轴，若有，做出修改
        time_explicit = ('time' in self.name_list)
        if time_explicit:
            assert self.time_index == self.get_index('time')
            img_num_to_show = image_num - 1
        else:
            img_num_to_show = image_num

        # capture the change of image_num
        if self.image_num!=image_num:
            self.reflesh_figure(draw_advance_fig=draw_advance_fig)
        self.image_num = image_num

        # replot figure when time is up
        if (time.time() - self.re_plot_timer) > self.re_plot_time_lim:
            self.re_plot_timer = time.time()
            self.reflesh_figure(draw_advance_fig=draw_advance_fig)

        # draw classic figure
        if True:
            rows = self.get_proper_row_num(img_num_to_show)
            cols = int(np.ceil(image_num/rows)) #根据行数求列数
            self.plot_classic(image_num, rows, time_explicit, self.time_index, cols)
        # draw advanced figure
        if draw_advance_fig:
            self.plot_advanced()

        # now end, output images
        if self.draw_mode == 'Web':
            content = self.mpld3.fig_to_html(self.classic_fig_handle)
            with open(self.html_to_write, 'w+') as f: f.write(content)
            return
        elif self.draw_mode == 'Native':
            self.plt.pause(0.01)
            return
        elif self.draw_mode == 'Img':
            if self.classic_fig_handle is not None: 
                self.classic_fig_handle.tight_layout()
                self.classic_fig_handle.savefig(self.img_to_write)
            if self.advance_fig_handle is not None: 
                self.advance_fig_handle.tight_layout()
                self.advance_fig_handle.savefig(self.img_to_write2)
            return

    def get_proper_row_num(self, img_num_to_show):
        rows = 1
        if img_num_to_show >= 3: rows = 2
        if img_num_to_show > 8: rows = 3
        if img_num_to_show > 12: rows = 4
        return rows

    def smooth(self, data, sm_lv=1):
        if len(data) < sm_lv:
            raise RuntimeError("You should not enable smoothing")
        if sm_lv > 1:
            y = np.ones(sm_lv)*1.0/sm_lv
            d = np.convolve(y, data, 'same')#"same")
        else:
            d = data
        return np.array(d)

    def recreate_fig_handle(self, index, num_group):
        handle = self.plt.figure(index, figsize=self.get_figure_size(num_group, baseline=6), dpi=100)
        # solve a bug inside matplotlib
        if self.default_fig_spp is None:
            self.default_fig_spp = copy.deepcopy(handle.subplotpars)
        else:
            handle.subplotpars = copy.deepcopy(self.default_fig_spp)
        # native
        if self.draw_mode == 'Native': 
            handle.canvas.set_window_title(f'Working-{index}')
            self.plt.show()
        return handle
    
    def plot_advanced(self):
        group_name = []
        group_member = []
        time_explicit = ('time' in self.name_list)
        image_num = len(self.line_list)
        for index in range(image_num):
            if 'of=' not in self.name_list[index]:
                continue # 没有的直接跳过
            g_name_ = self.name_list[index].split('of=')[0] # 找出组别
            if g_name_ in group_name:
                i = group_name.index(g_name_)
                group_member[i].append(index)
            else:
                group_name.append(g_name_)
                group_member.append([index])
        
        num_group = len(group_name)
        image_num_multi = num_group
        rows = self.get_proper_row_num(image_num_multi)
        cols = int(np.ceil(image_num_multi/rows)) #根据行数求列数

        if self.advance_fig_handle is None: 
            self.advance_fig_handle = self.recreate_fig_handle(index=AdvancePlotFigIndex, num_group=num_group)
        
        for i in range(num_group):
            subplot_index = i+1
            subplot_name = '%d,%d,%d'%(rows,cols,subplot_index)
            if subplot_name in self.advance_subplots: 
                target_subplot = self.advance_subplots[subplot_name]
            else:
                target_subplot = self.advance_fig_handle.add_subplot(rows,cols,subplot_index)
                self.advance_subplots[subplot_name] = target_subplot

            tar_true_name=group_name[i]
            num_member = len(group_member[i])

            _xdata_min_ = np.inf
            _xdata_max_ = -np.inf
            _ydata_min_ = np.inf
            _ydata_max_ = -np.inf

            for j in range(num_member):
                index = group_member[i][j]

                _ydata_ = np.array(self.line_list[index], dtype=np.double)
                if self.smooth_level is not None:
                    _ydata_ = self.smooth(_ydata_, sm_lv=self.smooth_level)
                # 如果有时间数据，把x轴绑定时间
                if time_explicit:
                    _xdata_ = np.array(self.time_list[index], dtype=np.double)
                else:
                    _xdata_ = np.arange(len(self.line_list[index]), dtype=np.double)

                limx1 = _xdata_.min() 
                if limx1 < _xdata_min_: _xdata_min_ = limx1
                limx2 = _xdata_.max()
                if limx2 > _xdata_max_: _xdata_max_ = limx2
                limy1 = _ydata_.min()
                if limy1 < _ydata_min_: _ydata_min_ = limy1
                limy2 = _ydata_.max()
                if limy2 > _ydata_max_: _ydata_max_ = limy2

                name_tmp = self.name_list[index]
                name_tmp = name_tmp.replace('=',' ')
                if (self.advance_plot_handle[index] is None):
                    # 第一次绘制
                    if time_explicit:
                        self.advance_plot_handle[index], =  target_subplot.plot(_xdata_, _ydata_, lw=1,label=name_tmp)
                    else:
                        self.advance_plot_handle[index], =  target_subplot.plot(_ydata_, lw=1, label=name_tmp)
                else:
                    # 非第一次，则只需要更新数据即可
                    self.advance_plot_handle[index].set_data((_xdata_, _ydata_))

            #标题
            target_subplot.set_title(tar_true_name)
            target_subplot.set_xlabel('time')
            target_subplot.set_ylabel(tar_true_name)
            self.change_target_figure_lim(target_subplot, _xdata_min_, _xdata_max_, _ydata_min_, _ydata_max_)
            target_subplot.grid(visible=True)
            # target_subplot.ticklabel_format(useOffset=False)
            target_subplot.legend(loc='best')


    def plot_classic(self, image_num, rows, time_explicit, time_index, cols):
        time_var_met = False
        if self.classic_fig_handle is None: 
            self.classic_fig_handle = self.recreate_fig_handle(index=ClassicPlotFigIndex, num_group=image_num)

        for index in range(image_num):
            if time_explicit:
                if time_index == index:
                    time_var_met = True 
                    continue # skip time var
            # 有时间轴时，因为不绘制时间，所以少算一个subplot
            subplot_index = index if time_var_met else index+1
            subplot_name = '%d,%d,%d'%(rows,cols,subplot_index)
            if subplot_name in self.classic_subplots: 
                target_subplot = self.classic_subplots[subplot_name]
            else:
                target_subplot = self.classic_fig_handle.add_subplot(rows,cols,subplot_index)
                self.classic_subplots[subplot_name] = target_subplot

            _ydata_ = np.array(self.line_list[index], dtype=np.double)
            # 如果有时间数据，把x轴绑定时间
            if time_explicit:
                _xdata_ = np.array(self.time_list[index], dtype=np.double)
            else:
                _xdata_ = np.arange(len(self.line_list[index]), dtype=np.double)

            if (self.classic_plot_handle[index] is None):
                # 第一次绘制
                if time_explicit:
                    self.classic_plot_handle[index], =  target_subplot.plot(_xdata_, _ydata_, lw=1,c=self.colorC)
                else:
                    self.classic_plot_handle[index], =  target_subplot.plot(_ydata_, lw=1, c=self.colorC)
                        
            else:
                # 后续绘制，更新数据
                self.classic_plot_handle[index].set_data((_xdata_, _ydata_))

            if 'of=' in self.name_list[index]:
                #把等号替换成空格
                name_tmp = self.name_list[index]
                name_tmp = name_tmp.replace('=',' ')
                target_subplot.set_title(name_tmp)
                target_subplot.set_xlabel('time')
                target_subplot.set_ylabel(name_tmp)
                # target_subplot.ticklabel_format(useOffset=False)
                target_subplot.grid(visible=True)
            else:
                target_subplot.set_title(self.name_list[index])
                target_subplot.set_xlabel('time')
                target_subplot.set_ylabel(self.name_list[index])
                # target_subplot.ticklabel_format(useOffset=False)
                target_subplot.grid(visible=True)

            _xdata_min_ = _xdata_.min() #target_subplot.dataLim.xmin
            _xdata_max_ = _xdata_.max() #target_subplot.dataLim.xmax
            _ydata_min_ = _ydata_.min() #min(self.line_list[index])
            _ydata_max_ = _ydata_.max() #max(self.line_list[index])

            if self.enable_percentile_clamp and len(_ydata_)>220 and self.vis_95percent:
                _ydata_min_ = np.percentile(_ydata_, 3, interpolation='midpoint') # 3%
                _ydata_max_ = np.percentile(_ydata_, 97, interpolation='midpoint') # 97%

            self.change_target_figure_lim(target_subplot, _xdata_min_, _xdata_max_, _ydata_min_, _ydata_max_)

    def change_target_figure_lim(self, target_subplot, limx1, limx2, limy1, limy2):
        if limy1!=limy2:
            meany = limy1/2 + limy2/2
            limy1 = (limy1 - meany)*1.2+meany
            limy2 = (limy2 - meany)*1.2+meany
            target_subplot.set_ylim(limy1,limy2)

        if limx1 != limx2:
            meanx = limx1/2 + limx2/2
            limx1 = (limx1 - meanx)*1.1+meanx
            limx2 = (limx2 - meanx)*1.1+meanx
            target_subplot.set_xlim(limx1,limx2)

    def rec_disable_percentile_clamp(self):
        self.enable_percentile_clamp = False

    def rec_enable_percentile_clamp(self):
        self.enable_percentile_clamp = True