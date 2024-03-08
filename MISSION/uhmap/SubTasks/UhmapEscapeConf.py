
class SubTaskConfig():
    agent_list = [
        { 'team':0, 'type':'PosAttacker',       'init_fn_name':'init_attack',  },
        { 'team':0, 'type':'PosAttacker',       'init_fn_name':'init_attack',  },
        { 'team':0, 'type':'PosAttacker',       'init_fn_name':'init_attack',  },
        { 'team':0, 'type':'PosAttacker',       'init_fn_name':'init_attack',  },
        { 'team':0, 'type':'PosAttacker',       'init_fn_name':'init_attack',  },
        { 'team':0, 'type':'PosAttacker',       'init_fn_name':'init_attack',  },
        { 'team':0, 'type':'PosAttacker',       'init_fn_name':'init_attack',  },
        { 'team':0, 'type':'PosAttacker',       'init_fn_name':'init_attack',  },

        
        { 'team':1, 'type':'Lv3_DefenceTank',  'init_fn_name':'init_defence',  },
        { 'team':1, 'type':'Lv3_DefenceTank',  'init_fn_name':'init_defence',  },
        { 'team':1, 'type':'Lv3_DefenceTank',  'init_fn_name':'init_defence',  },
        { 'team':1, 'type':'Lv3_DefenceTank',  'init_fn_name':'init_defence',  },

    ]


    AgentPropertyDefaults = {
        'ClassName': 'RLA_CAR',     # FString ClassName = "";
        'DebugAgent': False,     
        'AgentTeam': 0,             # int AgentTeam = 0;
        'IndexInTeam': 0,           # int IndexInTeam = 0;
        'UID': 0,                   # int UID = 0;
        'MaxMoveSpeed': 600,        # move speed, test ok
        'InitLocation': { 'x': 0,  'y': 0, 'z': 0, },
        'InitRotation': { 'x': 0,  'y': 0, 'z': 0, },
        'AgentScale'  : { 'x': 1,  'y': 1, 'z': 1, },     # agent size, test ok
        'InitVelocity': { 'x': 0,  'y': 0, 'z': 0, },
        'AgentHp':100,
        "WeaponCD": 1,              # weapon fire rate
        "IsTeamReward": True,
        "Type": "",
        "DodgeProb": 0.8,           # probability of escaping dmg 闪避概率, test ok
        "ExplodeDmg": 25,           # ms explode dmg. test ok
        "FireRange": 1000.0,        # <= 1500
        "GuardRange": 1400.0,       # <= 1500
        "PerceptionRange": 1500.0,       # <= 1500
        'Color':'(R=0,G=1,B=0,A=1)',    # color
        "FireRange": 1000,
        'RSVD1':'',
        'RSVD2':'',
    }

    obs_vec_length = 23
    ActionFormat = 'ASCII'

    # temporary parameters
    OBS_RANGE_PYTHON_SIDE = 15000
    MAX_NUM_OPP_OBS = 10
    MAX_NUM_ALL_OBS = 5

    obs_n_entity = 15
    LaserDmg = 18