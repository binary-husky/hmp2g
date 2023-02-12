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
           "ALGORITHM/experimental_coopspace_vma", 
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
            "ALGORITHM.experimental_coopspace_vma.reinforce_foundation->ReinforceAlgorithmFoundation"
        ]
    },

    "ALGORITHM.experimental_coopspace_vma.reinforce_foundation.py->CoopAlgConfig": {
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
        "addr": "210.75.240.143:2236",
        "usr": "hmp",
        "pwd": "hmp"
    },
]*n_run
assert len(n_run_mode)==n_run

sum_note = "aii-simp-net-vma-avail-act"
conf_override = {

    "config.py->GlobalConfig-->note":
        [
            "run1-drop",
            "run2-drop",
            "run3-drop",
            "run4-drop",
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
    "ALGORITHM.experimental_coopspace_vma.reinforce_foundation.py->CoopAlgConfig-->dropout_prob":
        [
            0.0,
            0.0,
            0.0,
            0.0,
        ],

}

if __name__ == '__main__':
    # copy the experiments
    import shutil, os
    shutil.copyfile(__file__, os.path.join(os.path.dirname(__file__), 'batch_experiment_backup.py'))
    # run experiments remotely
    from UTIL.batch_exp import run_batch_exp
    print('Execute in server:', n_run_mode[0])
    run_batch_exp(sum_note, n_run, n_run_mode, base_conf, conf_override, __file__)