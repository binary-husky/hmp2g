base = """

{
    "config.py->GlobalConfig": {
        "note": "batchexp",
        "env_name": "sr_tasks->hunter_invader3d_v2",
        "env_path": "MISSION.sr_tasks.multiagent.scenarios.hunter_invader3d_v2",
        "draw_mode": "Img",
        "num_threads": 32,
        "report_reward_interval": 32,              // reporting interval
        "test_interval": 512,                      // test every $test_interval episode
        "test_epoch": 128,                         // test every $test_interval episode
        "fold": 1,                                   // this 'folding' is designed for IPC efficiency, you can thank python GIL for such a strange design... 
        "n_parallel_frame": 5e7,
        "max_n_episode": 2e6,
        "backup_files": [                               // backup files, pack them up
           "ALGORITHM/experimental_coopspace_vma_varnet", 
           "MISSION/sr_tasks/multiagent"
        ],
        "device": "cuda",                             // choose from 'cpu' (no GPU), 'cuda' (auto select GPU), 'cuda:3' (manual select GPU) 
        "gpu_party": "off"                              // default is 'off', 
    },

    "MISSION.sr_tasks.multiagent.scenarios.hunter_invader3d_v2.py->ScenarioConfig": {
        "hunter_num": 15,
        "invader_num": 5,
        "num_landmarks": 6,
        "extreme_sparse": true,
        "render": false,
        "TEAM_NAMES": [ //select team algorithm
            "ALGORITHM.script_ai.manual->DummyAlgorithmFoundationHI3D",
            "ALGORITHM.experimental_coopspace_vma_varnet.reinforce_foundation->ReinforceAlgorithmFoundation"
        ]
    },

    "ALGORITHM.experimental_coopspace_vma_varnet.reinforce_foundation.py->CoopAlgConfig": {
        "g_num": 6,                    // d_k
        "train_traj_needed": 256,
        "n_pieces_batch_division": 1,
        "ppo_epoch": 16,
        "dropout_prob": 0.0,
    },
    "ALGORITHM.script_ai.manual.py->CoopAlgConfig": {
    }
}

"""


import commentjson as json
import numpy as np
base_conf = json.loads(base)
n_run = 4
n_run_mode = [
    {
        #"addr": "localhost:2266",172.18.29.20:20256
        "addr": "172.18.29.20:20256",
        "usr": "hmp",
        "pwd": "hmp"
    },
]*n_run
assert len(n_run_mode)==n_run

sum_note = "aii_experimental_coopspace_vma_varnet_debug_task15vs5"
conf_override = {

    "config.py->GlobalConfig-->note":
        [
            "run1",
            "run2",
            "run3",
            "run4",
        ],

    "config.py->GlobalConfig-->seed":       
        [
            np.random.randint(0, 10000) for _ in range(n_run)
        ],

    "config.py->GlobalConfig-->device":
        [
            'cuda',
            'cuda',
            'cuda',
            'cuda',
        ],
    "ALGORITHM.experimental_coopspace_vma_varnet.reinforce_foundation.py->CoopAlgConfig-->dropout_prob":
        [
            0.01,
            0.01,
            0.01,
            0.01,
        ],

}

def check_file_mod_time(base_conf):
    import glob, time, os
    fs = [k.split('->')[0].replace('.','/').replace('/py','.py')  for k in base_conf]
    fds = [os.path.dirname(f) for f in fs]; fs = []
    for fd in fds:
        fs.extend(glob.glob(fd+'/*.py'))
    def file_mod_time_till_now(f):
        filemt1= time.localtime(os.stat(f).st_mtime) #文件修改时间
        t1=time.mktime(filemt1)
        filemt2= time.localtime() #不带参数就是当前时间
        t2=time.mktime(filemt2)
        return (t2-t1)/60
    minute_write = [file_mod_time_till_now(f) for f in fs]
    i = np.argmin([file_mod_time_till_now(f) for f in fs])
    print(f'latest file is {fs[i]}, last modified {int(minute_write[i])} minutes ago')
    input('Confirm ?')

if __name__ == '__main__':
    # copy the experiments
    import shutil, os, argparse, time
    parser = argparse.ArgumentParser('Run Experiments')
    parser.add_argument('-d', '--debug', action='store_true', help='To debug?')
    args = parser.parse_args()
    check_file_mod_time(base_conf)
    file = os.path.abspath(__file__)
    os.chdir(os.path.dirname(file))
    # copy the experiments
    if not args.debug:
        shutil.copyfile(file, os.path.join(os.path.dirname(file), 'batch_experiment_backup.py'))
        shutil.copyfile(file, os.path.join(os.path.dirname(file), 
            f'private {time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())} batch_experiment_backup {sum_note}.py'))
    # run experiments remotely
    from UTIL.batch_exp import run_batch_exp
    print('Execute in server:', n_run_mode[0])
    run_batch_exp(sum_note, n_run, n_run_mode, base_conf, conf_override, file, debug=args.debug)
