import os, copy, atexit, time, gzip, threading, setproctitle, traceback, json
import numpy as np
from multiprocessing import Process
from UTIL.colorful import *
from UTIL.network import get_host_ip, find_free_port
from .mcom_def import fn_names, align_names, find_where_to_log


class mcom():
    """
        2D/3D visualizer interface
        The Design Principle: Under No Circumstance should this program interrupt the main program!
        args:
            draw_mode: ('Web', 'Native', 'Img', 'Threejs')
            rapid_flush: flush data instantly. set 'False' if you'd like your SSD to survive longer
            digit: the precision of float number. Choose from -1 (auto), 4, 8, 16
            tag: give a name for debugging when multiple mcom object is used
            resume_mod: resume previous session
            resume_file: resume from which file
            image_path: if draw_mode=='Img', where to save image
            figsize: if draw_mode=='Img', determine the size of the figure, default is (12, 6)
            rec_exclude: if draw_mode=='Img', blacklist some vars
    """
    def __init__(self, path=None, digit=-1, rapid_flush=True, draw_mode="Img", tag='default', resume_mod=False, **kargs):
        self.draw_mode = draw_mode
        self.rapid_flush = rapid_flush
        self.path = path
        self.digit = digit
        self.tag = tag
        self.resume_mod = resume_mod
        self.kargs = kargs
        if self.kargs is None: self.kargs = {}
        
        self.flow_cnt = 0

        if draw_mode in ['Web', 'Native', 'Img', 'Threejs']:
            self.draw_process = True
            self.init_draw_subprocess()
            if draw_mode in ['Web', 'Native', 'Img']:
                self.init_2d_kernel()
        else:
            print亮红('[mcom.py]: Draw process off! No plot will be done')
            self.draw_process = False

        atexit.register(lambda: self.__del__())

    def init_draw_subprocess(self):
        port = find_free_port()
        print红('[mcom.py]: draw process active!')
        self.draw_tcp_port = ('localhost', port)
        self.kargs.update({
            'draw_mode': self.draw_mode,
            'draw_udp_port': self.draw_tcp_port,
            'port': self.draw_tcp_port,
            'backup_file': self.path + '/backup.dp.gz'
        })
        DP = DrawProcess if self.draw_mode != 'Threejs' else DrawProcessThreejs
        self.draw_proc = DP(**self.kargs)
        self.draw_proc.start()
        from UTIL.network import QueueOnTcpClient
        self.draw_tcp_client = QueueOnTcpClient('localhost:%d'%port)



    def init_2d_kernel(self):
        if self.resume_mod:
            if "resume_file" in self.kargs:
                # if resume_file is specified, use it
                self.starting_file = self.kargs["resume_file"]
            else:
                # otherwise find previous log path
                _, _, self.current_buffer_index = find_where_to_log(self.path)
                self.starting_file = self.path + '/mcom_buffer_%d____starting_session.txt' % (self.current_buffer_index-1)
            # open the previous file, transfer previous data
            self.file_handle = open(self.starting_file, 'r', encoding = "utf-8")
            for line in self.file_handle.readlines(): self.draw_tcp_client.send_str(line)
            self.file_handle.close()
            print蓝('previous data transfered')
            # open this file again with append mode
            self.file_handle = open(self.starting_file, 'a+', encoding = "utf-8")

        else:
            _, _, self.current_buffer_index = find_where_to_log(self.path)
            self.starting_file = self.path + '/mcom_buffer_%d____starting_session.txt' % (self.current_buffer_index)
            print蓝('[mcom.py]: log file at:' + self.starting_file)
            self.file_handle = open(self.starting_file, 'w+', encoding = "utf-8")


    # on the end of the program
    def __del__(self):
        if hasattr(self,'_deleted_'): return    # avoid exit twice
        else: self._deleted_ = True     # avoid exit twice
        # print红('[mcom.py]: mcom exiting! tag: %s'%self.tag)
        if hasattr(self, 'file_handle') and self.file_handle is not None:
            end_file_flag = ('><EndTaskFlag\n')
            self.file_handle.write(end_file_flag)
            self.file_handle.close()
        if hasattr(self, 'port') and self.port is not None:
            self.disconnect()
        if hasattr(self, 'draw_proc') and self.draw_proc is not None:
            try:
                self.draw_proc.terminate()
                self.draw_proc.join()
            except:
                pass
        # print蓝('[mcom.py]: mcom exited! tag: %s'%self.tag)


    def disconnect(self):
        # self.draw_udp_client.close()
        self.draw_tcp_client.close()


    def recall(self, starting_file):
        with open(starting_file,'rb') as f:
            lines = f.readlines()
        r = None
        for l in lines:
            if 'rec_show' in str(l, encoding='utf8'): 
                r = copy.deepcopy(l)
                continue
            self.draw_tcp_client.send_str(l)
        if r is not None:
            self.draw_tcp_client.send_str(r)
        return None

    '''
        mcom core function: send out/write str
    '''
    def send(self, data):
        # step 1: send directive to draw process
        if self.draw_process: 
            self.draw_tcp_client.send_str(data)


        # step 2: add to file
        if self.draw_mode=='Threejs': return
        self.file_handle.write(data)
        if self.rapid_flush: 
            self.file_handle.flush()
        elif self.flow_cnt>500:
            self.file_handle.flush()
            self.flow_cnt = 0
        else:
            self.flow_cnt += 1
        return


    def rec_init(self, color='k'):
        str_tmp = '>>rec_init(\'%s\')\n' % color
        self.send(str_tmp)

    def rec_show(self):
        self.send('>>rec_show\n')

    def rec_end(self):
        self.send('>>rec_end\n')

    def rec_save(self):
        self.send('>>rec_save\n')

    def rec_get(self):
        self.send('>>rec_get\n')
        res = self.draw_tcp_client.wait_reply()
        return json.loads(res)

    def rec_end_hold(self):
        self.send('>>rec_end_hold\n')

    def rec_clear(self, name):
        str_tmp = '>>rec_clear("%s")\n' % (name)
        self.send(str_tmp)

    def rec(self, value, name):
        value = float(value)
        if self.digit == -1:
            str_tmp = '>>rec(%.16g,"%s")\n' % (value, name)
        elif self.digit == 16:
            str_tmp = '>>rec(%.16e,"%s")\n' % (value, name)
        elif self.digit == 8:
            str_tmp = '>>rec(%.8e,"%s")\n' % (value, name)
        elif self.digit == 4:
            str_tmp = '>>rec(%.4e,"%s")\n' % (value, name)
        self.send(str_tmp)

    def other_cmd(self, func_name, *args, **kargs):
        strlist = ['>>', func_name, '(']
        for _i_ in range(len(args)):
            if isinstance(args[_i_], np.ndarray):
                strlist = self._process_ndarray(args[_i_], strlist)
            else:
                strlist = self._process_scalar(args[_i_], strlist)
        if len(kargs)>0:
            for _key_ in kargs:
                if isinstance(kargs[_key_], np.ndarray):
                    strlist = self._process_ndarray(kargs[_key_], strlist, _key_)
                else:
                    strlist = self._process_scalar(kargs[_key_], strlist, _key_)
        if strlist[len(strlist) - 1] == "(": strlist.append(")\n")
        else: strlist[len(strlist) - 1] = ")\n" # 把逗号换成后括号
        self.send(''.join(strlist))

    def _process_scalar(self, arg, strlist,key=None):
        if key is not None: strlist += '%s='%key
        if isinstance(arg, int):
            strlist.append("%d" % arg)
            strlist.append(",")
        elif isinstance(arg, float):
            if self.digit == -1:    strlist.append("%.16g" % arg)
            elif self.digit == 16:  strlist.append("%.16e" % arg)
            elif self.digit == 8:   strlist.append("%.8e" % arg)
            elif self.digit == 4:   strlist.append("%.4e" % arg)
            strlist.append(",")
        elif isinstance(arg, str):
            assert '$' not in arg
            strlist.extend(["\'", arg.replace('\n', '$'), "\'", ","])
        elif isinstance(arg, list):
            strlist.append(str(arg))
            strlist.append(",")
        elif hasattr(arg, 'dtype') and np.issubdtype(arg.dtype, np.integer):
            strlist.append("%d" % arg)
            strlist.append(",")
        elif hasattr(arg, 'dtype') and np.issubdtype(arg.dtype, np.floating):
            if self.digit == -1:    strlist.append("%.16g" % arg)
            elif self.digit == 16:  strlist.append("%.16e" % arg)
            elif self.digit == 8:   strlist.append("%.8e" % arg)
            elif self.digit == 4:   strlist.append("%.4e" % arg)
            strlist.append(",")
        else:
            print('unknown input type | 输入的参数类型不能处理', arg.__class__)
        return strlist

    def _process_ndarray(self, args, strlist, key=None):
        if args.ndim == 1:
            if key is not None: strlist += '%s='%key
            d = len(args)
            sub_list = ["["] + ["%.3e,"%t if (i+1)!=d else "%.3e"%t for i, t in enumerate(args)] + ["]"]
            strlist += sub_list
            strlist.append(",")
        else:
            print红('[mcom]: input dimension > 1, unable to process | 输入数组的维度大于2维')
        return strlist

    for fn_name in fn_names:
        build_exec_cmd = 'def %s(self,*args,**kargs):\n self.other_cmd("%s", *args,**kargs)\n'%(fn_name, fn_name)
        exec(build_exec_cmd)

    for align, fn_name in align_names:
        build_exec_cmd = '%s = %s\n'%(align, fn_name)
        exec(build_exec_cmd)












class DrawProcessThreejs(Process):
    def __init__(self, draw_udp_port, draw_mode, **kargs):
        super(DrawProcessThreejs, self).__init__()
        from UTIL.network import QueueOnTcpServer
        self.draw_mode = draw_mode
        self.draw_udp_port = draw_udp_port
        self.tcp_connection = QueueOnTcpServer(self.draw_udp_port)
        self.buffer_list = []
        self.backup_file = kargs['backup_file']
        self.allow_backup = False if self.backup_file is None else True
        if self.allow_backup:
            if os.path.exists(self.backup_file):
                print亮红('[mcom.py]: warning, purge previous 3D visual data!')
                try: os.remove(self.backup_file)
                except: pass
            self.tflush_buffer = []
        self.client_tokens = {}

    def flush_backup(self):
        while True:
            time.sleep(20)
            if not os.path.exists(os.path.dirname(self.backup_file)):
                os.makedirs(os.path.dirname(self.backup_file))
            # print('Flush backup')
            with gzip.open(self.backup_file, 'at') as f:
                f.writelines(self.tflush_buffer)
            self.tflush_buffer = []
            # print('Flush backup done')

    def init_threejs(self):
        t = threading.Thread(target=self.run_flask, args=(find_free_port(),))
        t.daemon = True
        t.start()

        if self.allow_backup:
            self.tflush = threading.Thread(target=self.flush_backup)
            self.tflush.daemon = True
            self.tflush.start()

    def run(self):
        setproctitle.setproctitle('ThreejsVisualWorker')
        
        self.init_threejs()
        try:
            from queue import Empty
            queue = self.tcp_connection.get_queue()
            self.tcp_connection.wait_connection() # after this, the queue begin to work
            while True:
                buff_list = []
                buff_list.extend(queue.get(block=True))
                for _ in range(queue.qsize()): buff_list.extend(queue.get(block=True))
                self.run_handler(buff_list)
        except KeyboardInterrupt:
            self.__del__()
        self.__del__()

    def __del__(self):
        return
        
    def run_handler(self, new_buff_list):
        self.buffer_list.extend(new_buff_list)
        self.tflush_buffer.extend(new_buff_list)

        # too many, delete with fifo
        if len(self.buffer_list) > 1e9: 
            # 当存储的指令超过十亿后，开始删除旧的
            del self.buffer_list[:len(new_buff_list)]

    def run_flask(self, port):
        from flask import Flask, request, send_from_directory
        from waitress import serve
        from mimetypes import add_type
        add_type('application/javascript', '.js')
        add_type('text/css', '.css')

        app = Flask(__name__)
        dirname = os.path.dirname(__file__) + '/threejsmod'
        import zlib

        self.init_cmd_captured = False
        init_cmd_list = []
        def init_cmd_capture_fn(tosend):
            for strx in tosend:
                if '>>v2d_show()\n'==strx:
                    self.init_cmd_captured = True
                init_cmd_list.append(strx)    
                if self.init_cmd_captured:
                    break
            return
            
        @app.route("/up", methods=["POST"])
        def up():

            # 本次正常情况下，需要发送的数据
            # dont send too much in one POST, might overload the network traffic
            if len(self.buffer_list)>35000:
                tosend = self.buffer_list[:30000]
                self.buffer_list = self.buffer_list[30000:]
            else:
                tosend = self.buffer_list
                self.buffer_list = []

            # 处理断线重连的情况，断线重连时，会出现新的token
            token = request.data.decode('utf8')
            if token not in self.client_tokens:
                print('[mcom.py] Establishing new connection, token:', token)
                self.client_tokens[token] = 'connected'
                if (len(self.client_tokens)==0) or (not self.init_cmd_captured):  
                    # 尚未捕获初始化命令，或者第一次client 
                    buf = "".join(tosend)
                else:
                    print('[mcom.py] If there are other tabs, please close them now.')
                    buf = "".join(init_cmd_list + tosend)
            else:
                # 正常连接
                buf = "".join(tosend)

            # 尝试捕获并保存初始化部分的命令
            if not self.init_cmd_captured:
                init_cmd_capture_fn(tosend)

            # use zlib to compress output command, worked out like magic
            buf = bytes(buf, encoding='utf8')   
            zlib_compress = zlib.compressobj()
            buf = zlib_compress.compress(buf) + zlib_compress.flush(zlib.Z_FINISH)
            return buf

        @app.route("/<path:path>")
        def static_dirx(path):
            if path=='favicon.ico': 
                return send_from_directory("%s/"%dirname, 'files/HMP.ico')
            return send_from_directory("%s/"%dirname, path)

        @app.route("/")
        def main_app():
            with open('%s/examples/abc.html'%dirname, 'r', encoding = "utf-8") as f:
                buf = f.read()
            return buf

        print('\n--------------------------------')
        print('JS visualizer online: http://%s:%d'%(get_host_ip(), port))
        print('JS visualizer online (localhost): http://localhost:%d'%(port))
        print('--------------------------------')
        # app.run(host='0.0.0.0', port=port)
        serve(app, threads=8, ipv4=True, ipv6=True, listen='*:%d'%port)


class DrawProcess(Process):
    def __init__(self, draw_udp_port, draw_mode, **kargs):
        from UTIL.network import QueueOnTcpServer
        super(DrawProcess, self).__init__()
        self.draw_mode = draw_mode
        self.draw_udp_port = draw_udp_port
        self.tcp_connection = QueueOnTcpServer(self.draw_udp_port)
        self.kwargs = kargs

        return

    def init_matplot_lib(self):
        if self.draw_mode in ['Web', 'Img']:
            import matplotlib
            matplotlib.use('Agg') # set the backend before importing pyplot
            import matplotlib.pyplot as plt
            self.gui_reflesh = lambda: time.sleep(1) # plt.pause(0.1)
        elif self.draw_mode == 'Native':
            import matplotlib
            # matplotlib.use('Agg') # set the backend before importing pyplot
            matplotlib.use('Qt5Agg')
            import matplotlib.pyplot as plt
            self.gui_reflesh = lambda: plt.pause(0.2)
        elif self.draw_mode == 'Threejs':
            assert False
        else:
            assert False

        from config import GlobalConfig
        logdir = GlobalConfig.logdir
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        if self.draw_mode == 'Web':
            self.avail_port = find_free_port()
            my_http = MyHttp('%s/html.html'%logdir, self.avail_port)
            my_http.daemon = True
            my_http.start()
    
        self.libs_family = {
            "rec_disable_percentile_clamp": 'rec', "rec_enable_percentile_clamp": 'rec',
            'rec_init': 'rec', 'rec': 'rec', 'rec_show': 'rec',
            'v2d_init': 'v2d', 'v2dx':'v2d', 'v2d_show': 'v2d', 'v2d_pop':'v2d',
            'v2d_line_object':'v2d', 'v2d_clear':'v2d', 'v2d_add_terrain': 'v2d',
        }
        self.libs_init_fns = {
            'rec': self.rec_init_fn,
            'v2d': self.v2d_init_fn,
        }

    def run(self):
        setproctitle.setproctitle('HmapPlotProcess')
        self.init_matplot_lib()
        try:
            # self.tcp_connection.set_handler(self.run_handler)
            from queue import Empty
            queue = self.tcp_connection.get_queue()
            # self.tcp_connection.set_handler(self.run_handler)
            self.tcp_connection.wait_connection() # after this, the queue begin to work
            while True:
                try: 
                    buff_list = []
                    buff_list.extend(queue.get(timeout=0.1))
                    for _ in range(queue.qsize()): buff_list.extend(queue.get(timeout=0.1))
                    self.run_handler(buff_list)
                except Empty: self.gui_reflesh()

        except KeyboardInterrupt:
            self.__del__()
        self.__del__()

    def run_handler(self, buff_list):
        while True:
            if len(buff_list) == 0: break
            buff = buff_list.pop(0)
            if (buff=='>>rec_show\n') and ('>>rec_show\n' in buff_list): 
                continue # skip
            elif (buff=='>>rec_get\n'): 
                result = self.rec.rec_get()
                self.tcp_connection.tcpServerP2P.reply_last_client(result)
            else:
                try:
                    self.process_cmd(buff)
                except:
                    traceback.print_exc()
                    print亮红(f'[mcom.py] We have encountered error processing command: {buff}')

        #     # print('成功处理指令:', buff)

    def __del__(self):
        self.tcp_connection.close()



    def process_cmd(self, cmd_str):
        if '>>' in cmd_str:
            cmd_str_ = cmd_str[2:].strip('\n')
            if ')' not in cmd_str_:
                cmd_str_ = cmd_str_+'()'
            prefix = self.get_cmd_lib(cmd_str_)
            if prefix is not None: 
                eval(f'{prefix}.{cmd_str_}')

    def get_cmd_lib(self, cmd):
        cmd_key = None
        func_name = cmd.split('(')[0]
        if func_name not in self.libs_family:
            print蓝('绘图函数不能处理：', cmd)
            return None
        family_name = self.libs_family[func_name]
        if self.libs_init_fns[family_name] is not None:
            self.libs_init_fns[family_name]()
            self.libs_init_fns[family_name] = None
        return 'self.%s'%family_name

    def rec_init_fn(self):
        from VISUALIZE.mcom_rec import rec_family
        self.rec = rec_family('r', 
            self.draw_mode, 
            **self.kwargs
        )

    def v2d_init_fn(self):
        from VISUALIZE.mcom_v2d import v2d_family
        self.v2d = v2d_family(self.draw_mode)




class MyHttp(Process):
    def __init__(self, path_to_html, avail_port):
        super(MyHttp, self).__init__()
        self.path_to_html = path_to_html
        self.avail_port = avail_port

    def run(self):
        from flask import Flask
        app = Flask(__name__)
        @app.route("/")
        def hello():
            try:
                with open(self.path_to_html,'r') as f:
                    html = f.read()
            except:
                html = "no plot yet please wait"
            return html
        app.run(port=self.avail_port)
