import subprocess
import threading
import copy, os
import time
import json
from UTIL.network import get_host_ip
from UTIL.colorful import *
def get_info(script_path):
    info = {
        'HostIP': get_host_ip(),
        'RunPath': os.getcwd(),
        'ScriptPath': os.path.abspath(script_path),
        'StartDateTime': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }
    try:
        info['DockerContainerHash'] = subprocess.getoutput(r'cat /proc/self/cgroup | grep -o -e "docker/.*"| head -n 1 |sed "s/docker\\/\\(.*\\)/\\1/" |cut -c1-12')
    except: 
        info['DockerContainerHash'] = 'None'
    return info

def run_batch_exp(sum_note, n_run, n_run_mode, base_conf, conf_override, script_path, skip_confirm=False, master_folder='MultiServerMission', auto_rl=False, debug=False, logger=None):
    arg_base = ['python', 'main.py']
    time_mark_only = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    time_mark = time_mark_only + '-' + sum_note
    log_dir = '%s/'%time_mark
    exp_log_dir = log_dir+'exp_log'
    if not os.path.exists('PROFILE/%s'%exp_log_dir):
        os.makedirs('PROFILE/%s'%exp_log_dir)
    exp_json_dir = log_dir+'exp_json'
    if not os.path.exists('PROFILE/%s'%exp_json_dir):
        os.makedirs('PROFILE/%s'%exp_json_dir)

    conf_list = []
    new_json_paths = []
    for i in range(n_run):
        conf = copy.deepcopy(base_conf)
        new_json_path = 'PROFILE/%s/run-%d.json'%(exp_json_dir, i+1)
        for key in conf_override:
            try:
                assert n_run == len(conf_override[key]), ('检查！n_run是否对应', key)
            except:
                pass
            tree_path, item = key.split('-->')
            conf[tree_path][item] = conf_override[key][i]
        with open(new_json_path,'w') as f:
            json.dump(conf, f, indent=4)
        # print(conf)
        conf_list.append(conf)
        new_json_paths.append(new_json_path)

    if not auto_rl:
        print红('\n')
        print红('\n')
        print红('\n')

    printX = [
        print亮红, print亮绿, print亮黄, print亮蓝, print亮紫, print亮靛, 
        print红,   print绿,   print黄,   print蓝,   print紫,   print靛,
        print亮红, print亮绿, print亮黄, print亮蓝, print亮紫, print亮靛, 
        print红,   print绿,   print黄,   print蓝,   print紫,   print靛,
        print亮红, print亮绿, print亮黄, print亮蓝, print亮紫, print亮靛, 
        print红,   print绿,   print黄,   print蓝,   print紫,   print靛,
        print亮红, print亮绿, print亮黄, print亮蓝, print亮紫, print亮靛, 
        print红,   print绿,   print黄,   print蓝,   print紫,   print靛,
        print亮红, print亮绿, print亮黄, print亮蓝, print亮紫, print亮靛, 
        print红,   print绿,   print黄,   print蓝,   print紫,   print靛,
    ]
    
    conf_base_ = conf_list[0]
    for k_ in conf_base_:
        conf_base = conf_base_[k_]
        for key in conf_base:
            different = False
            for i in range(len(conf_list)):
                if conf_base[key]!=conf_list[i][k_][key]:
                    different = True
                    break
            # 
            if different:
                for i in range(len(conf_list)):
                    printX[i](key, conf_list[i][k_][key])
            else:
                print(key, conf_base[key])



    final_arg_list = []

    for ith_run in range(n_run):
        final_arg = copy.deepcopy(arg_base)
        final_arg.append('--cfg')
        final_arg.append(new_json_paths[ith_run])
        final_arg_list.append(final_arg)
        print('')

    def local_worker_std_console(ith_run):
        printX[ith_run%len(printX)](final_arg_list[ith_run])
        subprocess.run(final_arg_list[ith_run])

    def local_worker(ith_run):
        log_path = open('PROFILE/%s/run-%d.log'%(exp_log_dir, ith_run+1), 'w+')
        printX[ith_run%len(printX)](final_arg_list[ith_run])
        subprocess.run(final_arg_list[ith_run], stdout=log_path, stderr=log_path)

    def remote_worker(ith_run):
        # step 1: transfer all files
        from UTIL.exp_helper import get_ssh_sftp
        
        addr = n_run_mode[ith_run]['addr']
        if 'exe_here' in addr: 
            _, addr = addr.split('=>')
            usr = n_run_mode[ith_run]['usr']
            pwd = n_run_mode[ith_run]['pwd']
            ssh, sftp = get_ssh_sftp(addr, usr, pwd)
            src_path = os.getcwd()
        else:
            # assert False
            usr = n_run_mode[ith_run]['usr']
            pwd = n_run_mode[ith_run]['pwd']
            ssh, sftp = get_ssh_sftp(addr, usr, pwd)
            sftp.mkdir('/home/%s/%s'%(usr, master_folder), ignore_existing=True)
            sftp.mkdir('/home/%s/%s/%s'%(usr, master_folder, time_mark), ignore_existing=True)
            src_path = '/home/%s/%s/%s/src'%(usr, master_folder, time_mark)
            try:
                sftp.mkdir(src_path, ignore_existing=False)
                if auto_rl:
                    ignore_list=['__pycache__','TEMP','ZHECKPOINT', '.git', 'threejsmod', 'md_imgs', 'ZDOCS', 'AUTORL']
                else:
                    ignore_list=['__pycache__','TEMP','ZHECKPOINT']
                sftp.put_dir('./', src_path, ignore_list=ignore_list)
                sftp.close()
                print紫('upload complete')
            except:
                sftp.close()
                print紫('do not need upload')

        print('byobu attach -t %s'%time_mark_only)
        addr_ip, addr_port = addr.split(':')


        if logger is not None and ith_run==0:
            logger.info("Attach cmd: ssh %s@%s -p %s -t \"byobu attach -t %s\""%(usr, addr_ip, addr_port, time_mark_only))


        print亮蓝("Attach cmd: ssh %s@%s -p %s -t \"byobu attach -t %s\""%(usr, addr_ip, addr_port, time_mark_only))
        
        stdin, stdout, stderr = ssh.exec_command(command='byobu new-session -d -s %s'%time_mark_only, timeout=1)
        print亮紫('byobu new-session -d -s %s'%time_mark_only)
        time.sleep(1)

        byobu_win_name = '%s--run-%d'%(time_mark_only, ith_run)
        byobu_win_name = byobu_win_name
        stdin, stdout, stderr = ssh.exec_command(command='byobu new-window -t %s'%time_mark_only, timeout=1)
        # print亮紫('byobu new-window -t %s'%time_mark_only)
        time.sleep(1)

        cmd = 'cd  ' + src_path
        stdin, stdout, stderr = ssh.exec_command(command='byobu send-keys -t %s "%s" C-m'%(time_mark_only, cmd), timeout=1)
        # print亮紫('byobu send-keys "%s" C-m'%cmd)
        time.sleep(1)

        
        cmd = ' '.join(['echo',  str(get_info(script_path)) ,'>>', './private_remote_execution.log'])
        stdin, stdout, stderr = ssh.exec_command(command='byobu send-keys -t %s "%s" C-m'%(time_mark_only, cmd), timeout=1)
        # print亮紫('byobu send-keys "%s" C-m'%cmd)
        time.sleep(1)


        cmd = ' '.join(final_arg_list[ith_run])
        stdin, stdout, stderr = ssh.exec_command(command='byobu send-keys -t %s "%s" C-m'%(time_mark_only, cmd), timeout=1)
        print亮紫('byobu send-keys "%s" C-m'%cmd)
        time.sleep(1)




        print亮蓝("command send is done!")
        time.sleep(2)

        # 杀死
        # stdin, stdout, stderr = ssh.exec_command(command='byobu kill-session -t %s'%byobu_win_name, timeout=1)
        note = conf_list[ith_run]['config.py->GlobalConfig']['note']
        future = {
            'checkpoint': '/home/%s/%s/%s/src/ZHECKPOINT/%s'%(usr, master_folder, time_mark, note),
            'conclusion': '/home/%s/%s/%s/src/ZHECKPOINT/%s/experiment_conclusion.pkl'%(usr, master_folder, time_mark, note),
            'mark': time_mark,
            'time_mark': time_mark_only,
        }
        return future

    def worker(ith_run):
        if n_run_mode[ith_run] is None: 
            return local_worker(ith_run)
        else:
            return remote_worker(ith_run)
         


    def clean_process(pid):
        import psutil
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            try:
                print亮红('sending Terminate signal to', child)
                child.terminate()
                time.sleep(5)
                print亮红('sending Kill signal to', child)
                child.kill()
            except: pass
        parent.kill()

    def clean_up():
        print亮红('clean up!')
        parent_pid = os.getpid()   # my example
        clean_process(parent_pid)

    if debug:
        local_worker_std_console(0)
        return None

    if not skip_confirm:
        input('Confirm execution? 确认执行?')
        input('Confirm execution! 确认执行!')

    def count_down(DELAY):
        while DELAY > 0: 
            time.sleep(1); DELAY -= 1; print(f'\rCounting down {DELAY}', end='', flush=True)

    count_down(5)

    future = []
    for ith_run in range(n_run):
        future.append(worker(ith_run)); count_down(10)

    print('all submitted')
    return future

def objload(file):
    import pickle, os
    if not os.path.exists(file): 
        return
    with open(file, 'rb') as f:
        return pickle.load(f)


def clean_byobu_interface(future_list, n_run_mode):
    for ith_run, future in enumerate(future_list):
        from UTIL.exp_helper import get_ssh_sftp
        addr = n_run_mode[ith_run]['addr']
        usr = n_run_mode[ith_run]['usr']
        pwd = n_run_mode[ith_run]['pwd']
        ssh, sftp = get_ssh_sftp(addr, usr, pwd)
        time_mark_only = future['time_mark']
        stdin, stdout, stderr = ssh.exec_command(command='byobu kill-session -t %s'%(time_mark_only), timeout=1)
        print亮紫('byobu kill-session -t %s'%(time_mark_only))
        time.sleep(1)

def fetch_experiment_conclusion(step, future_list, n_run_mode):
    n_run = len(future_list)
    conclusion_list = []

    time_out = 4 * 3600 # 在一个小时后timeout
    time_start = time.time()

    for ith_run, future in enumerate(future_list):
        # step 1: transfer all files
        from UTIL.exp_helper import get_ssh_sftp
        addr = n_run_mode[ith_run]['addr']
        # assert False
        usr = n_run_mode[ith_run]['usr']
        pwd = n_run_mode[ith_run]['pwd']
        ssh, sftp = get_ssh_sftp(addr, usr, pwd)
        def remote_exist(remote_file, sftp):
            try:
                sftp.stat(future['conclusion'])
                return True
            except:
                return False
        while not remote_exist(future['conclusion'], sftp):
            used_time = time.time() - time_start
            print('Waiting', addr, future['conclusion'], 'Timeout in:', time_out - used_time)
            if used_time > time_out: 
                clean_byobu_interface(future_list, n_run_mode)
                raise TimeoutError
            time.sleep(10)
        if not os.path.exists('./ZHECKPOINT/AutoRL/'): os.makedirs('./ZHECKPOINT/AutoRL/')
        sftp.get(future['conclusion'], f'./ZHECKPOINT/AutoRL/conclusion_{step}_{ith_run}.pkl')
        conclusion_list.append(objload(f'./ZHECKPOINT/AutoRL/conclusion_{step}_{ith_run}.pkl'))

    clean_byobu_interface(future_list, n_run_mode)
    return conclusion_list

