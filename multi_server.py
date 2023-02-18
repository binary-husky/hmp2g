base = """
{
    "config.py->GlobalConfig": {
        "note": "Run1-Lr-Study",   // 实验存储路径
        "env_name": "dca_multiteam",  // 环境（任务名称）
        "env_path": "MISSION.dca_multiteam", 
        "draw_mode": "Img",
        "num_threads": 32,    // 环境并行数量
        "report_reward_interval": 32,
        "test_interval": 1024,
        "test_epoch": 256,
        "mt_parallel": true,
        "device": "cpu", // 使用哪张显卡
        "fold": "1",        // 使用的进程数量 = 环境并行数量/fold
        "n_parallel_frame": 50000000.0,
        "max_n_episode": 100000.0,
        "seed": 22334, // 随机数种子
        "mt_act_order": "new_method",
        "backup_files": [
            "ALGORITHM/experimental_conc_mt",
            "MISSION/dca_multiteam"
        ]
    },
    "MISSION.dca_multiteam.collective_assult_parallel_run.py->ScenarioConfig": {
        "N_TEAM": 2,
        "N_AGENT_EACH_TEAM": [35, 35],
        "introduce_terrain": true,
        "terrain_parameters": [0.15, 0.2],
        "size": "5",
        "random_jam_prob": 0.05,
        "MaxEpisodeStep": 150,     // 时间限制， 胜利条件：尽量摧毁、存活
        "render": false,           // 高效渲染,只有0号线程环境会渲染
        "RewardAsUnity": true,
        "half_death_reward": true,
        "TEAM_NAMES": [
            "ALGORITHM.experimental_conc_mt.foundation->ReinforceAlgorithmFoundation",
            "TEMP.TEAM2.ALGORITHM.experimental_conc_mt.foundation->ReinforceAlgorithmFoundation",
        ]
    },
    "ALGORITHM.experimental_conc_mt.foundation.py->AlgorithmConfig": {
        "train_traj_needed": 32,
        "n_focus_on": 3,
        "lr": 0.0001,
        "ppo_epoch": 16,
        "lr_descent": false,
        "use_policy_resonance": false,
        "gamma": 0.99,
    },
    "TEMP.TEAM2.ALGORITHM.experimental_conc_mt.foundation.py->AlgorithmConfig": {
        "train_traj_needed": 32,
        "n_focus_on": 3,
        "lr": 0.0001,
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

sum_note = "MMM2-conc4hist"
conf_override = {

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

    "config.py->GlobalConfig-->note":
        [
            "n_focus_on_run1_3focus",
            "n_focus_on_run2_3focus",
            "n_focus_on_run1_5focus",
            "n_focus_on_run2_5focus",
        ],

    "ALGORITHM.conc_4hist_scdb.foundation.py->AlgorithmConfig-->n_focus_on":
        [
            3,
            3,
            5,
            5,
        ],
}

if __name__ == '__main__':
    # copy the experiments
    import shutil, os
    shutil.copyfile(__file__, os.path.join(os.path.dirname(__file__), 'batch_experiment_backup.py'))
    # run experiments remotely
    from UTIL.batch_exp import run_batch_exp
    run_batch_exp(sum_note, n_run, n_run_mode, base_conf, conf_override, __file__)
