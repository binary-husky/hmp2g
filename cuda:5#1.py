base = """
{
    "config.py->GlobalConfig": {
        "note": "Run1-Lr-Study",   // 实验存储路径
        "env_name": "dca_multiteam",  // 环境（任务名称）
        "env_path": "MISSION.dca_multiteam", 
        "draw_mode": "Img",
        "num_threads": 32,    // 环境并行数量
        "report_reward_interval": 32,
        "test_interval": 65536,
        "test_epoch": 256,
        "mt_parallel": true,
        "device": "cpu", // 使用哪张显卡
        "fold": "1",        // 使用的进程数量 = 环境并行数量/fold
        "n_parallel_frame": 50000000.0,
        "max_n_episode": 4096.0,
        "seed": 22334, // 随机数种子
        "mt_act_order": "new_method",
        "backup_files": [
            "ALGORITHM/experimental_conc_mt_fuzzy5",
            "MISSION/dca_multiteam"
        ]
    },
    "MISSION.dca_multiteam.collective_assult_parallel_run.py->ScenarioConfig": {
        "N_TEAM": 2,
        "N_AGENT_EACH_TEAM": [20, 20],
        "introduce_terrain": true,
        "terrain_parameters": [0.15, 0.2],
        "size": "5",
        "random_jam_prob": 0.05,
        "MaxEpisodeStep": 150,     // 时间限制， 胜利条件：尽量摧毁、存活
        "render": false,           // 高效渲染,只有0号线程环境会渲染
        "RewardAsUnity": true,
        "half_death_reward": true,
        "TEAM_NAMES": [
            "ALGORITHM.experimental_conc_mt_fuzzy5.foundation->ReinforceAlgorithmFoundation",
            "TEMP.TEAM2.ALGORITHM.experimental_conc_mt_fuzzy5.foundation->ReinforceAlgorithmFoundation",
        ]
    },
    "ALGORITHM.experimental_conc_mt_fuzzy5.foundation.py->AlgorithmConfig": {
        "train_traj_needed": 32,
        "n_focus_on": 4,
        "lr": 0.0003,
        "ppo_epoch": 16,
        "lr_descent": false,
        "fuzzy_controller": true,
        "fuzzy_controller_param": [2, 2, 3, 0, 2],
        "fuzzy_controller_scale_param": [0.8117568492889404],
        "use_policy_resonance": false,
        "gamma": 0.99,
    },
    "TEMP.TEAM2.ALGORITHM.experimental_conc_mt_fuzzy5.foundation.py->AlgorithmConfig": {
        "train_traj_needed": 32,
        "n_focus_on": 4,
        "lr": 0.0003,
        "ppo_epoch": 16,
        "lr_descent": false,
        "use_policy_resonance": false,
        "gamma": 0.99,
    },
}
"""


import commentjson as json
import numpy as np
base_conf = json.loads(base)
n_run = 4
n_run_mode = [
    {
        "addr": "localhost:2266",
        "usr": "hmp",
        "pwd": "hmp"
    },
]*n_run
assert len(n_run_mode)==n_run

sum_note = "test-stable"
conf_override = {

    "config.py->GlobalConfig-->seed":       
        [
            np.random.randint(0, 10000) for _ in range(n_run)
        ],

    "config.py->GlobalConfig-->note":
        [
            "run1",
            "run2",
            "run3",
            "run4",
        ],

    ########################################
    "ALGORITHM.experimental_conc_mt_fuzzy5.foundation.py->AlgorithmConfig-->device_override":
        [
            "cuda:2",
            "cuda:2",
            "cuda:3",
            "cuda:3",
        ],

    "ALGORITHM.experimental_conc_mt_fuzzy5.foundation.py->AlgorithmConfig-->gpu_party_override":
        [
            "cuda2_party3", # 各子实验的party可以相同， 但每个实验的子队伍party建议设置为不同值
            "cuda2_party3",
            "cuda3_party3", # 各子实验的party可以相同， 但每个实验的子队伍party建议设置为不同值
            "cuda3_party3",
        ],

    ########################################
    "TEMP.TEAM2.ALGORITHM.experimental_conc_mt_fuzzy5.foundation.py->AlgorithmConfig-->device_override":
        [
            "cuda:3",
            "cuda:3",
            "cuda:2",
            "cuda:2",
        ],

    "TEMP.TEAM2.ALGORITHM.experimental_conc_mt_fuzzy5.foundation.py->AlgorithmConfig-->gpu_party_override":
        [
            "cuda3_party3",
            "cuda3_party3",
            "cuda2_party3",
            "cuda2_party3",
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
