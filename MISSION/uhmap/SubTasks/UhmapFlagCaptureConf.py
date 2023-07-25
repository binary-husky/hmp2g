from UTIL.config_args import ChainVar, ChainVarSimple

class SubTaskConfig():
    agent_list = [
        { "team": 0,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 0,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 0,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 0,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 0,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 0,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 0,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 0,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 0,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 0,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 0,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 0,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 0,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 0,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 0,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 0,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 0,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 0,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 0,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 0,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },


        { "team": 1,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 1,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 1,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 1,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 1,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 1,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 1,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 1,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 1,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 1,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 1,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 1,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 1,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 1,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 1,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 1,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 1,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 1,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 1,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },
        { "team": 1,   "type": "Lv3_MomentumAgentWithHp",   "init_fn_name": "init_drone"   },

    ]

    ActionFormat = 'ASCII'
    OBS_RANGE_PYTHON_SIDE = 10000
    MAX_NUM_OPP_OBS = 7
    MAX_NUM_ALL_OBS = 7
    MAX_OBJ_NUM_ACCEPT = 2
    obs_vec_length = 23

    obs_n_entity = 16   # MAX_NUM_OPP_OBS + MAX_NUM_ALL_OBS + MAX_OBJ_NUM_ACCEPT
    obs_n_entity_cv = ChainVarSimple('$MAX_NUM_OPP_OBS$ + $MAX_NUM_ALL_OBS$ + $MAX_OBJ_NUM_ACCEPT$')
