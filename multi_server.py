base = """
{
    "config.py->GlobalConfig": {
        "note": "z_test_ppoma",
        "env_name": "uhmap",
        "env_path": "Mission.uhmap",
        "draw_mode": "Img",
        "num_threads": 16,
        "report_reward_interval": 256,
        "test_interval": 2560,
        "test_epoch": 256,
        "interested_team": 0,
        "seed": 870,
        "device": "cuda",
        "max_n_episode": 5000000,
        "fold": 1,
        "backup_files": [
            "Algorithm/ppo_ma",
            "Mission/uhmap"
        ]
    },
    "Mission.uhmap.uhmap_env_wrapper.py->ScenarioConfig": {
        "n_team1agent": 10,
        "n_team2agent": 10,
        "MaxEpisodeStep": 125,
        "StepGameTime": 0.5,
        "StateProvided": false,
        "render": false,
        "UElink2editor": false,
        "AutoPortOverride": true,
        "HeteAgents": true,
        "UnrealLevel": "UhmapLargeScale",
        "SubTaskSelection": "UhmapLargeScale",
        "UhmapRenderExe": "/home/hmp/fuqingxu/UHMP/Build/LinuxNoEditor/UHMP.sh",
        "UhmapServerExe": "/home/hmp/fuqingxu/UHMP/Build/LinuxServer/UHMPServer.sh",
        "TimeDilation": 64,
        "TEAM_NAMES": [
            "Algorithm.ppo_ma.foundation->ReinforceAlgorithmFoundation",
            "Algorithm.script_ai.uhmap_ls->DummyAlgorithmLinedAttack"
        ]
    },
    "Mission.uhmap.SubTasks.UhmapLargeScaleConf.py->SubTaskConfig":{
        "agent_list": [
            { "team":0,  "tid":0,   "type":"RLA_UAV_Support", "init_fn_name":"init_air"      },
            { "team":0,  "tid":1,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":2,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":3,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":4,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":5,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":6,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":7,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":0,  "tid":8,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":0,  "tid":9,   "type":"RLA_UAV_Support", "init_fn_name":"init_air"      },

            { "team":1,  "tid":0,   "type":"RLA_UAV_Support", "init_fn_name":"init_air"      },
            { "team":1,  "tid":1,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":2,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":3,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":4,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":5,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":6,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":7,   "type":"RLA_CAR",         "init_fn_name":"init_ground"   },
            { "team":1,  "tid":8,   "type":"RLA_CAR_Laser",   "init_fn_name":"init_ground"   },
            { "team":1,  "tid":9,   "type":"RLA_UAV_Support", "init_fn_name":"init_air"      },
        ]
    },

    // --- Part3: config Algorithm 1/2 --- 
    "Algorithm.script_ai.uhmap_ls.py->DummyAlgConfig": {
        "reserve": ""
    },
    // --- Part3: config Algorithm 2/2 --- 
    "Algorithm.ppo_ma.shell_env.py->ShellEnvConfig": {
        "add_avail_act": true
    },
    "Algorithm.ppo_ma.foundation.py->AlgorithmConfig": {
        "train_traj_needed": 256,
        "use_normalization":true,
        "load_specific_checkpoint": "",
        "gamma": 0.99,
        "gamma_in_reward_forwarding": "True",
        "gamma_in_reward_forwarding_value": 0.95,
        "prevent_batchsize_oom": "True",
        "lr": 0.0004,
        "ppo_epoch": 24,
        "policy_resonance": false,
        "debug": true,
        "n_entity_placeholder": 11
    },


}
"""



import commentjson as json
import numpy as np
base_conf = json.loads(base)
n_run = 6
n_run_mode = [
    {
        "addr": "172.18.116.161:2266",
        "usr": "hmp",
        "pwd": "hmp"
    },
]*6
assert len(n_run_mode)==n_run
conf_override = {
    "config.py->GlobalConfig-->seed":       
        [
            np.random.randint(0, 10000) for _ in range(n_run)
        ],
    "config.py->GlobalConfig-->note":
        [
            "RVE-drone2-ppoma-traj-32-run1",
            "RVE-drone2-ppoma-traj-32-run2",
            
            "RVE-drone2-ppoma-traj-64-run1",
            "RVE-drone2-ppoma-traj-64-run2",
            
            "RVE-drone2-ppoma-traj-128-run1",
            "RVE-drone2-ppoma-traj-128-run2",
        ],
    "Algorithm.ppo_ma.foundation.py->AlgorithmConfig-->train_traj_needed":
        [
            32,
            32,
            64,
            64,
            128,
            128,
        ],
    "Algorithm.ppo_ma.foundation.py->AlgorithmConfig-->lr":
        [
            0.0001,
            0.0001,

            0.0001,
            0.0001,

            0.0002,
            0.0002,
        ],
}

if __name__ == '__main__':
    from UTIL.batch_exp import run_batch_exp
    run_batch_exp(n_run, n_run_mode, base_conf, conf_override)