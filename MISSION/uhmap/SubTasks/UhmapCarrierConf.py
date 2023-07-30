
class SubTaskConfig():
    agent_list = [
        { "team": 0,    "type": "Carrier",          "init_fn_name": "init_carrier" },
        { "team": 0,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 0,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 0,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 0,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 0,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 0,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 0,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 0,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 0,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 0,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 0,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 0,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 0,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 0,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 0,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 0,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 0,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 0,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 0,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },


        { "team": 1,    "type": "Carrier",          "init_fn_name": "init_carrier" },
        { "team": 1,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 1,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 1,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 1,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 1,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 1,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 1,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 1,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 1,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 1,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 1,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 1,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 1,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 1,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 1,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 1,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 1,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 1,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
        { "team": 1,    "type": "SmallDrone",       "init_fn_name": "init_drone"   },
    ]

    obs_vec_length = 23
    ActionFormat = 'ASCII'

    OBS_RANGE_PYTHON_SIDE = 7000
    MAX_NUM_OPP_OBS = 6
    MAX_NUM_ALL_OBS = 6
    MAX_OBJ_NUM_ACCEPT = 1
    obs_n_entity = 13

    N_Carrier = 1