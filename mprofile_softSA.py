import subprocess
import threading
import copy, os
import time
import json
from UTILS.colorful import *

# ubuntu command to kill process: kill -9 $(ps -ef | grep xrdp | grep -v grep | awk '{print $ 2}')

arg_base = ['python', 'main.py']
log_dir = '%s/'%time.time()
run_group = "bench"
# base_conf = 'train.json'

n_run = 3
conf_override = {
    "config.py->GlobalConfig-->note":       
                [
                    "SoftSA v2 r1",
                    "SoftSA v2 r2",
                    "SoftSA v2 r3",
                ],

    "config.py->GlobalConfig-->seed":       
                [
                    9991,
                    9992,
                    9993,
                ],
    "config.py->GlobalConfig-->device":       
                [
                    "cuda:0",
                    "cuda:1",
                    "cuda:3",
                ],
    "config.py->GlobalConfig-->gpu_party":       
                [
                    "Cuda0-Party0",
                    "Cuda1-Party0",
                    "Cuda3-Party0",
                ],

}

base_conf = {
    "config.py->GlobalConfig": {
        "note": "softSA",
        "env_name":"collective_assult",
        "env_path":"MISSIONS.collective_assult",
        "draw_mode": "Img",
        "num_threads": "64",
        "report_reward_interval": "64",
        "test_interval": "2048",
        "device": "cuda:3",
        "gpu_party": "Cuda3-Party0",
        "fold": "1",
        "seed": 9992,
        "backup_files":[
            "ALGORITHM/hmp_ak_iagentv2/net.py",
            "ALGORITHM/hmp_ak_iagentv2/ppo.py",
            "ALGORITHM/hmp_ak_iagentv2/trajectory.py",
        ]

    },

    "MISSIONS.collective_assult.collective_assult_parallel_run.py->ScenarioConfig": {
        "size": "5",
        "random_jam_prob": 0.1,
        "introduce_terrain":"True",
        "terrain_parameters": [0.05, 0.2],
        "num_steps": "180",
        "render":"False",
        "render_with_unity":"False",
        "MCOM_DEBUG":"False",
        "render_ip_with_unity": "cn-cd-dx-1.natfrp.cloud:55861",
        "half_death_reward": "True",
        "TEAM_NAMES": [
            "ALGORITHM.hmp_ak_iagentv2.foundation->ReinforceAlgorithmFoundation"
        ]
    },

    "ALGORITHM.hmp_ak_iagentv2.foundation.py->AlgorithmConfig": {
        "actor_attn_mod": "False",
        "observable_attn": "True",
        "train_traj_needed": "64",
        "ppo_epoch": 24,
        "lr": 5e-4,
        "add_prob_loss": "False",
        "load_checkpoint": "False",
        "seperate_critic": "True", 
        "agent_wise_attention": "False"
    }
}





assert '_' not in run_group, ('下划线在matlab中的显示效果不好')
log_dir = log_dir+run_group
if not os.path.exists('PROFILE/%s'%log_dir):
    os.makedirs('PROFILE/%s'%log_dir)
    os.makedirs('PROFILE/%s-json'%(log_dir))

new_json_paths = []
for i in range(n_run):
    conf = copy.deepcopy(base_conf)
    new_json_path = 'PROFILE/%s-json/run-%d.json'%(log_dir, i+1)
    for key in conf_override:
        tree_path, item = key.split('-->')
        conf[tree_path][item] = conf_override[key][i]
    with open(new_json_path,'w') as f:
        json.dump(conf, f, indent=4)
    print(conf)
    new_json_paths.append(new_json_path)











final_arg_list = []
printX = [print红,print绿,print黄,print蓝,print紫,print靛,print亮红,print亮绿,print亮黄,print亮蓝,print亮紫,print亮靛]

for ith_run in range(n_run):
    final_arg = copy.deepcopy(arg_base)
    final_arg.append('--cfg')
    final_arg.append(new_json_paths[ith_run])
    final_arg_list.append(final_arg)
    print('')

def worker(ith_run):
    log_path = open('PROFILE/%s-json/run-%d.log'%(log_dir, ith_run+1), 'w+')
    printX[ith_run%len(printX)](final_arg_list[ith_run])
    subprocess.run(final_arg_list[ith_run], stdout=log_path, stderr=log_path)

if __name__ == '__main__':
        
    input('确认执行？')
    input('确认执行！')

    t = 0
    while (t >= 0):
        print('运行倒计时：', t)
        time.sleep(1)
        t -= 1

    threads = [ threading.Thread( target=worker,args=(ith_run,) ) for ith_run in range(n_run) ]
    for thread in threads:
        thread.setDaemon(True)
        thread.start()
        print('错峰执行，启动', thread)
        for i in range(5):
            print('\r 错峰执行，启动倒计时%d     '%(5-i), end='', flush=True)
            time.sleep(1)

    while True:
        is_alive = [thread.is_alive() for thread in threads]
        if any(is_alive):
            time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
            print(time_now, 'I am still running!', is_alive)
            print靛('current scipt:%s, current log:%s'%(os.path.abspath(__file__), 'PROFILE/%s-json/run-%d.log'%(log_dir, ith_run+1)))
            time.sleep(120)
        else:
            break
    print('[profile] All task done!')
    for thread in threads:
        thread.join()
