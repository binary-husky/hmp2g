base = """
{
    "config.py->GlobalConfig": {
        "note": "R1-abl-MMM2",
        "env_name": "sc2",
        "env_path": "MISSION.starcraft.sc2_env_wrapper",
        "draw_mode": "Img",
        "num_threads": 16,
        "report_reward_interval": 16,
        "test_interval": 2048,
        "test_epoch": 128,
        "device": "cuda",
        "max_n_episode": 1500000,
        "fold": 1,
        "seed": 1,
        "backup_files": [
            "ALGORITHM/experimental_pr_smac"
        ],
    },
    "MISSION.starcraft.sc2_env_wrapper.py->ScenarioConfig": {
        "map_": "MMM2",
        "sc_version": "2.4.6",
        "TEAM_NAMES": [
            "ALGORITHM.experimental_pr_smac.foundation->ReinforceAlgorithmFoundation"
        ]
    },
    "ALGORITHM.experimental_pr_smac.foundation.py->AlgorithmConfig": {
        "lr": 0.0001,
        "ppo_epoch": 16,
        "n_focus_on": -1,
        "n_entity_placeholder": -1,
        "train_traj_needed": 64,
        "distribution_precision": -1,
        "target_distribute": [
            -1
        ],
        "shell_obs_add_id": true,
        "shell_obs_add_previous_act": true,
        "gamma_in_reward_forwarding": true,
        "use_policy_resonance": true,
        "policy_resonance_method": "legacy",
        "advantage_norm": true,
        "BlockInvalidPg": false,
        "load_checkpoint": false,
        "use_conc_net": false
    },
    "ALGORITHM.experimental_pr_smac.stage_planner.py->PolicyRsnConfig": {
        "lockPrInOneBatch": false,
        "resonance_start_at_update": 5000,
        "yita_max": 0.7,
        "yita_shift_method": "slow-inc"
    }
}

"""


import commentjson as json
import numpy as np
base_conf = json.loads(base)
n_run = 6
n_run_mode = [
    {
        "addr": "210.75.240.143:2236",
        "usr": "hmp",
        "pwd": "hmp"
    },
]*n_run
assert len(n_run_mode)==n_run

sum_note = "old-pr-revise-on-smac-train_traj_needed64-half-prupdate-16thread"
conf_override = {

    "config.py->GlobalConfig-->note":
        [
            "run-1-3s5z_vs_3s6z",
            "run-2-3s5z_vs_3s6z",
            "run-3-3s5z_vs_3s6z",
            "run-4-3s5z_vs_3s6z",
            "run-5-3s5z_vs_3s6z",
            "run-6-3s5z_vs_3s6z",
        ],

    "MISSION.starcraft.sc2_env_wrapper.py->ScenarioConfig-->map_":
        [
            "3s5z_vs_3s6z",
            "3s5z_vs_3s6z",
            "3s5z_vs_3s6z",
            "3s5z_vs_3s6z",
            "3s5z_vs_3s6z",
            "3s5z_vs_3s6z",
        ],

    "config.py->GlobalConfig-->seed":
        [
            np.random.randint(0, 10000) for _ in range(n_run)
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