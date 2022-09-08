# Using pymarl2 as Algorithm
## step 1: clone hmap code (dev-uhmp)
```
git clone https://github.com/binary-husky/hmp2g.git -b dev-uhmp
```


## step 2: clone hmap code (dev-uhmp)

``` sh
# change dir
cd ./hmp2g/thirdparty/pymarl2

# clone pymarl2
git clone https://github.com/binary-husky/pymarl-hmap-compat.git 

# rename pymarl2
mv pymarl-hmap-compat pymarl2src

# change dir
cd ../..
```

## step 3: run example
Save following file as ```debug.json```: 
``` json
{
    "config.py->GlobalConfig": {
        "note": "RVE-drone2-qmix-fixstate-run1",
        "env_name": "uhmap",
        "env_path": "Mission.uhmap",
        "draw_mode": "Img",
        "num_threads": 8,
        "report_reward_interval": 256,
        "test_interval": 5120,
        "test_epoch": 256,
        "interested_team": 0,
        "seed": 8529,
        "device": "cuda",
        "max_n_episode": 5000000,
        "fold": 1,
        "backup_files": [
            "Algorithm/pymarl2_compat",
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
            "Algorithm.pymarl2_compat.pymarl2_compat->PymarlFoundation",
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


    "Algorithm.script_ai.uhmap_ls.py->DummyAlgConfig": {
        "reserve": ""
    },
    "Algorithm.pymarl2_compat.pymarl2_compat.py->AlgorithmConfig": {
        "use_shell": "mini_shell_uhmap",
        "state_compat": "pad",
        "pymarl_config_injection": {
            "controllers.my_n_controller.py->PymarlAlgorithmConfig": {
                "use_normalization": "True",
                "use_vae": "False"
            },
            "config.py->GlobalConfig": {
                "batch_size": 128,
                "load_checkpoint": "False"
            }
        }
    }
}
```

Then start training with:
```python main.py -c debug.json```