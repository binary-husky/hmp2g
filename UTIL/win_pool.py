"""
Author: Fu Qingxu, CASIA
Description: 高效并行执行工具
相比仅支持Linux的shm_pool性能略低
但支持Windows和Linux双平台
"""

# 导入依赖库
import multiprocessing
import platform
import time
from multiprocessing import Pipe  # 进程间通信管道
from sys import stdout  # 标准输出

import numpy as np
import psutil

from config import GlobalConfig  # 全局配置

from .hmp_daemon import kill_process_and_its_children  # 进程终止工具


# 定义彩色打印函数
def print_red(*kw, **kargs):
    print("\033[1;31m", *kw, "\033[0m", **kargs)


def print_green(*kw, **kargs):
    print("\033[1;32m", *kw, "\033[0m", **kargs)


# 非终端环境回退到普通打印
if not stdout.isatty():
    print_green = print_red = print


def child_process_load_config(machine_info):
    """
    子进程配置加载函数（仅Windows需要）
    用于在Windows环境下加载JSON配置或命令行参数
    """
    from UTIL.config_args import prepare_args

    prepare_args(vb=False)
    # 同步主进程的machine_info配置
    GlobalConfig.machine_info = machine_info


class SuperProc(multiprocessing.Process):
    """
    多进程工作单元类
    继承自Process，实现自定义进程行为
    """

    def __init__(self, pipe, pipeHelp, index, base_seed, machine_info):
        super(SuperProc, self).__init__()
        # 主通信管道
        self.p = pipe
        # 辅助管道（用于配置加载）
        self.pH = pipeHelp
        # 本地随机种子（进程索引+基础种子）
        self.local_seed = index + base_seed
        # 进程索引
        self.index = index
        # 机器配置信息
        self.machine_info = machine_info

    def automatic_generation(self, name, gen_fn, *arg):
        """动态生成对象方法"""
        setattr(self, name, gen_fn(*arg))

    def automatic_execution(self, name, dowhat, *arg):
        """动态执行对象方法"""
        return getattr(getattr(self, name), dowhat)(*arg)

    def add_targets(self, new_target_args):
        """添加目标对象到进程"""
        for new_target_arg in new_target_args:
            name, gen_fn, arg = new_target_arg
            if arg is None:
                self.automatic_generation(name, gen_fn)
            elif isinstance(arg, tuple):
                self.automatic_generation(name, gen_fn, *arg)
            else:
                self.automatic_generation(name, gen_fn, arg)

    def execute_target(self, recv_args):
        """执行目标对象的方法"""
        res_list = [None] * len(recv_args)
        for i, recv_arg in enumerate(recv_args):
            name, dowhat, arg = recv_arg
            if arg is None:
                res = self.automatic_execution(name, dowhat)
            elif isinstance(arg, tuple):
                res = self.automatic_execution(name, dowhat, *arg)
            else:
                res = self.automatic_execution(name, dowhat, arg)
            res_list[i] = res
        return res_list

    def run(self):
        """进程主循环"""
        import numpy

        numpy.random.seed(self.local_seed)
        # Windows需要重新加载配置（Linux使用fork）
        if not platform.system() == "Linux":
            child_process_load_config(self.machine_info)
        print("[win_pool]: process worker %d started" % self.index)
        try:
            while True:
                # 主通信管道接收任务
                recv_args = self.p.recv()
                # 非列表参数切换到辅助通道
                if not isinstance(recv_args, list):
                    if recv_args == 0:
                        # 接收目标参数
                        self.add_targets(self.pH.recv())
                    elif recv_args == -1:
                        print("Parallel worker exit")
                        break  # 终止信号
                    else:
                        assert False
                    continue
                # 执行任务并返回结果
                result = self.execute_target(recv_args)
                self.p.send(result)
        except KeyboardInterrupt:
            self.__del__()
        self.__del__()

    def __del__(self):
        """析构函数，清理资源"""
        self.p.close()
        self.pH.close()
        # 终止进程及其子进程
        kill_process_and_its_children(psutil.Process())


class SmartPool(object):
    """
    智能进程池管理器
    管理多个SuperProc进程的创建和通信
    """

    def __init__(self, proc_num, fold, base_seed=None):
        # 进程数量
        self.proc_num = proc_num
        # 任务折叠数（每个进程承载的实例数）
        self.task_fold = fold
        # 创建通信管道
        self.thisSide, self.thatSide = zip(*[Pipe() for _ in range(proc_num)])
        self.thisSideHelp, self.thatSideHelp = zip(*[Pipe() for _ in range(proc_num)])
        # 随机种子生成
        self.base_seed = int(np.random.rand() * 1e5) if base_seed is None else base_seed
        print("[win_pool]: SmartPool base rand seed", self.base_seed)
        # 创建进程池
        self.proc_pool = [
            SuperProc(
                pipe=p,
                pipeHelp=pH,
                index=cnt,
                base_seed=self.base_seed,
                machine_info=GlobalConfig.machine_info,
            )
            for p, pH, cnt in zip(self.thatSide, self.thatSideHelp, range(proc_num))
        ]
        # 启动所有进程
        for proc in self.proc_pool:
            proc.daemon = False
            proc.start()
            time.sleep(0.001)
        # 关闭子端管道
        for i in range(proc_num):
            self.thatSide[i].close()
            self.thatSideHelp[i].close()

    def add_target(self, name, lam, args_list=None):
        """
        添加目标对象到进程池
        name: 对象名称
        lam: 构造函数
        args_list: 参数列表
        """
        lam_list = None
        if isinstance(lam, list):
            lam_list = lam
        for j in range(self.proc_num):
            tuple_list_to_be_send = []
            for i in range(self.task_fold):
                name_fold = name + str(i)
                args = None if args_list is None else args_list[i + j * self.task_fold]
                if lam_list is not None:
                    lam = lam_list[i + j * self.task_fold]
                tuple_list_to_be_send.append((name_fold, lam, args))
            # 切换到辅助通道发送参数
            self.thisSide[j].send(0)
            self.thisSideHelp[j].send(tuple_list_to_be_send)

    def exec_target(self, name, dowhat, args_list=None, index_list=None):
        """
        在进程池中执行目标方法
        name: 目标名称
        dowhat: 方法名
        args_list: 参数列表
        index_list: 执行索引列表
        """
        if index_list is None:
            # 执行所有实例
            for j in range(self.proc_num):
                tuple_list_to_be_send = []
                for i in range(self.task_fold):
                    name_fold = name + str(i)
                    args = (
                        None if args_list is None else args_list[i + j * self.task_fold]
                    )
                    tuple_list_to_be_send.append((name_fold, dowhat, args))
                self.thisSide[j].send(tuple_list_to_be_send)
            # 收集结果
            res_sort = []
            for j in range(self.proc_num):
                res_sort.extend(self.thisSide[j].recv())
            return res_sort
        else:
            # 执行指定索引的实例
            tuple_List_List = [
                [None for _ in range(self.task_fold)] for _ in range(self.proc_num)
            ]
            do_task_flag = [False for _ in range(self.proc_num)]
            do_task_fold = [[] for _ in range(self.proc_num)]
            result_recv_List_List = [
                [None for _ in range(self.task_fold)] for _ in range(self.proc_num)
            ]

            # 任务分配
            for i, index in enumerate(index_list):
                which_proc = index // self.task_fold
                which_fold = index % self.task_fold
                name_fold = name + str(which_fold)
                args = None if args_list is None else args_list[i]
                tuple_List_List[which_proc][which_fold] = (name_fold, dowhat, args)
                do_task_flag[which_proc] = True

            # 发送任务
            for which_proc in range(self.proc_num):
                tuple_send_buffer = []
                for which_fold, item in enumerate(tuple_List_List[which_proc]):
                    if item is None:
                        continue
                    tuple_send_buffer.append(item)
                    do_task_fold[which_proc].append(which_fold)
                if do_task_flag[which_proc]:
                    assert len(tuple_send_buffer) > 0
                    self.thisSide[which_proc].send(tuple_send_buffer)

            # 接收结果
            for which_proc in range(self.proc_num):
                if not do_task_flag[which_proc]:
                    continue
                recv_tmp = self.thisSide[which_proc].recv()
                for index, recv_item in enumerate(recv_tmp):
                    which_fold = do_task_fold[which_proc][index]
                    result_recv_List_List[which_proc][which_fold] = recv_item

            # 结果排序
            res_sort = [None] * len(index_list)
            for i, index in enumerate(index_list):
                which_proc = index // self.task_fold
                which_fold = index % self.task_fold
                res_sort[i] = result_recv_List_List[which_proc][which_fold]
            return res_sort

    def party_over(self):
        """结束进程池"""
        self.__del__()

    def __del__(self):
        """析构函数，资源清理"""
        if hasattr(self, "terminated"):
            return

        print("[win_pool]: executing superpool del")
        print("[win_pool]: Sending exit command to workers ...")
        try:
            # 发送退出信号
            for i in range(self.proc_num):
                self.thisSide[i].send(-1)
            self.terminated = True
        except:
            pass

        print("[win_pool]: Closing pipe ...")
        # 关闭所有管道
        for i in range(self.proc_num):
            try:
                self.thisSide[i].close()
                self.thisSideHelp[i].close()
            except:
                pass

        # 等待清理
        N_SEC_WAIT = 2
        for i in range(N_SEC_WAIT):
            print_red(
                "[win_pool]: terminate in %d, please wait for a clean up"
                % (N_SEC_WAIT - i)
            )
            time.sleep(1)

        # 终止所有子进程
        for proc in self.proc_pool:
            try:
                kill_process_and_its_children(proc)
            except Exception:
                pass

        print_green("[shm_pool]: __del__ finish")
        self.terminated = True
