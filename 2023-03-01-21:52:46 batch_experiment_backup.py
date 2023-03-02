base = """
{
    "config.py->GlobalConfig": {
        "note": "old_cargo_exp_mid_alg_hdim128",
        "env_name": "sr_tasks->cargo",
        "env_path": "MISSION.sr_tasks.multiagent.scenarios.cargo",
        "draw_mode": "Img",
        "num_threads": "512",
        "report_reward_interval": "512",
        "test_interval": "5120",
        "fold": "8",
        "n_parallel_frame": 50000000.0,
        "max_n_episode": 2000000.0,
        "backup_files": [
            "ALGORITHM/experimental_coopspace_vma_varnet",
            "MISSION/sr_tasks/multiagent"
        ],
        "gpu_party": "off"
    },
    "MISSION.sr_tasks.multiagent.scenarios.cargo.py->ScenarioConfig": {
        "MaxEpisodeStep": 150,
        "n_worker": 50,
        "weight_percent": 0.8,
        "n_cargo": 4,
        "render": "False",
        "TEAM_NAMES": [
            "ALGORITHM.experimental_coopspace_vma_varnet.reinforce_foundation->ReinforceAlgorithmFoundation"
        ]
    },
    "ALGORITHM.experimental_coopspace_vma_varnet.reinforce_foundation.py->CoopAlgConfig": {
        "n_pieces_batch_division": 1,
        "lr": 0.0001,
        "g_num": 6,
        "train_traj_needed": 512,
        "max_internal_step": 3,
        "decision_interval": 25,
        "head_start_cnt": 4,
        "head_start_hold_n": 1,
        "hidden_dim": 128,
        "ppo_epoch": 16
    }
}

"""


import commentjson as json
import numpy as np
base_conf = json.loads(base)
n_run = 4
n_run_mode = [
    {
        "addr": "localhost:2266",
        # "addr": "210.75.240.143:2236",
        "usr": "hmp",
        "pwd": "hmp"
    },
]*n_run
assert len(n_run_mode)==n_run

sum_note = "cargo-vma-6cluster-128hidden-old-normalization"
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
    print(f'latest file is {fs[i]}, last modified {minute_write[i]} minutes ago')
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
    shutil.copyfile(file, os.path.join(os.path.dirname(file), 'batch_experiment_backup.py'))
    shutil.copyfile(file, os.path.join(os.path.dirname(file), f'{time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())} batch_experiment_backup.py'))
    # run experiments remotely
    from UTIL.batch_exp import run_batch_exp
    print('Execute in server:', n_run_mode[0])
    run_batch_exp(sum_note, n_run, n_run_mode, base_conf, conf_override, file, debug=args.debug)