import gym
from gym import spaces
import numpy as np
import sys
sys.path.append('./MISSION/collective_assult')
# from .multi_discrete import MultiDiscrete
import time, os
import json
# from pyglet.gl import *
# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
from config import ChainVar
from UTIL.colorful import print亮绿
import cProfile, pstats
from UTIL.tensor_ops import dir2rad, arrange_id

def distance_matrix_AB(A, B):
    assert A.shape[-1] == 2 # assert 2D situation
    assert B.shape[-1] == 2 # assert 2D situation
    n_A_subject = A.shape[-2]
    n_B_subject = B.shape[-2]
    A = np.repeat(np.expand_dims(A,-2), n_B_subject, axis=-2) # =>(64, Na, Nb, 2)
    B = np.repeat(np.expand_dims(B,-2), n_A_subject, axis=-2) # =>(64, Nb, Na, 2)
    Bt = np.swapaxes(B,-2,-3) # =>(64, Na, Nb, 2)
    dis = Bt-A # =>(64, Na, Nb, 2)
    dis = np.linalg.norm(dis, axis=-1)
    return dis


class ScenarioConfig(object):  
    '''
        ScenarioConfig: This config class will be 'injected' with new settings from JSONC.
        (E.g., override configs with ```python main.py --cfg example.jsonc```)
        (As the name indicated, ChainVars will change WITH vars it 'chained_with' during config injection)
        (please see UTIL.config_args to find out how this advanced trick works out.)
    '''
    N_TEAM = 2

    N_AGENT_EACH_TEAM = [25, 25]

    AGENT_ID_EACH_TEAM = arrange_id(N_AGENT_EACH_TEAM)
    AGENT_ID_EACH_TEAM_CV = ChainVar(lambda N_AGENT_EACH_TEAM: arrange_id(N_AGENT_EACH_TEAM), chained_with=['N_AGENT_EACH_TEAM']) 
    TEAM_NAMES = [ 
        'ALGORITHM.Starcraft.star_foundation->StarFoundation',
        'ALGORITHM.Starcraft.star_foundation->StarFoundation', 
    ]

    size = 5.0

    random_jam_prob = 0.0
    introduce_terrain = False
    terrain_parameters = [0, 0]

    MaxEpisodeStep = 150
    StepsToNarrowCircle = 100

    init_distance = 2.5  # 5-2,5+2
    render = False
    render_with_unity = False

    map_ = 'default'
    benchmark = False

    obs_vec_length = 17


    # dec_dictionary = {'alive':0, 'pos':range(1,3), 'ang':3, 'vel':range(4,6), 'id':6}
    StateProvided = False
    AvailActProvided = False
    RewardAsUnity = False
    ObsAsUnity = False   # 减少IPC负担
    EntityOriented = True

    # 调试
    MCOM_DEBUG = False
    Terrain_DEBUG = False
    half_death_reward = True

    n_actions = 7

 
def make_collective_assult_env(env_id, rank):
    # scenario = gym.make('collective_assult-v1')
    from .envs.collective_assult_env import collective_assultEnvV1
    scenario = collective_assultEnvV1()
    # create world
    world = scenario.world
    world.max_time_steps = ScenarioConfig.MaxEpisodeStep
    # create multiagent environment
    if ScenarioConfig.benchmark:
        env = collective_assultGlobalEnv(world, scenario, scenario.benchmark_data, id=rank)
    else:
        env = collective_assultGlobalEnv(world, scenario, id=rank)
    return env


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class collective_assultGlobalEnv(gym.Env):

    def terminate(self):
        pass

    def __init__(self, world, scenario, info_callback=None,
                 done_callback=None, shared_viewer=True, id = -1):
        self.render_on = ScenarioConfig.render
        self.render_with_unity = ScenarioConfig.render_with_unity
        self.n_teams = len(ScenarioConfig.N_AGENT_EACH_TEAM)
        self.N_AGENT_EACH_TEAM = ScenarioConfig.N_AGENT_EACH_TEAM
        self.scenario = scenario
        self.s_cfg = ScenarioConfig
        # 当并行时，只有0号环境可以渲染
        self.id = id
        if self.id!=0: 
            self.render_on = False
            self.render_with_unity = False

        self.ob_rms = None
        self.world = world
        self.agents = self.world.agents
        # set required vectorized gym env property
        self.n = len(world.agents)
        # scenario callbacks
        self.reset_callback = self.scenario.reset_world
        self.reward_callback = self.scenario.reward
        self.observation_callback = self.scenario.observation
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = True #False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False


        # configure spaces
        self.action_space = []
        self.observation_space = []
        obs_shapes = []
        self.agent_num = len(self.agents)

        # action has 8 values: 
        # nothing, +forcex, -forcex, +forcey, -forcey, +rot, -rot, shoot 
        self.action_spaces = None
        self.observation_spaces = None

        self.env_specs = None
        self.action_range = [0., 1.]


    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []
        self.agents = self.world.agents
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space) # sets the actions in the agent object
        
        # calculate terrain
        self.world.set_terrain_adv()
        # advance world state
        ## actions are already set in the objects, so we can simply pass step without any argument
        self.world.step() # world is the collective_assult-v0 environment, step function is in core.py file
        
        # record observation for each agent
        if ScenarioConfig.ObsAsUnity:
            o_, _ = self._get_obs(self.agents[0])
            obs_n.append(o_)

        
        for agent in self.agents:
            if not ScenarioConfig.ObsAsUnity: 
                o_, _ = self._get_obs(agent)
                obs_n.append(o_)
            if not ScenarioConfig.RewardAsUnity: 
                reward_n.append(self._get_reward(agent))

        done, WinningResultInfo, ExtraReward = self._get_done()

        if ScenarioConfig.RewardAsUnity:
            reward_n = self._get_sparse_reward()
            reward_n += ExtraReward

        info = WinningResultInfo
        reward_n = np.array(reward_n)
        self.world.time_step += 1
        obs_n = np.array(obs_n)
        if self.render_on: 
            if self.render_with_unity:  assert False
            self.render()
        return obs_n, reward_n, done, info

    def reset(self):
        # reset world
        self.reset_callback()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.agents
        self.world.before_step_alive_agents = [a for a in self.world.agents if a.alive]
        # calculate terrain
        self.world.set_terrain_adv()
        if ScenarioConfig.ObsAsUnity:
            o_, _ = self._get_obs(self.agents[0])
            obs_n.append(o_)
        else:
            for agent in self.agents:
                o_, _ = self._get_obs(agent)
                obs_n.append(o_)       
        info = {}
        obs_n = np.array(obs_n)
        return obs_n, info

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0), None
        return self.observation_callback(agent, self.world)

    def get_rank(self, score_list):
        # print('各队伍得分:', score_list)
        rank = np.array([sum([ s2 > s1 for s2 in score_list ]) for s1 in score_list])
        if (rank == 0).all():  # if all team draw, then all team lose
            rank[:] = -1
        # print('各队伍排名:', rank)
        return rank

    # get done for the whole environment
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self):
        ExtraReward = np.array([0] * self.n_teams)
        alive_agent_each_team = np.array([0] * self.n_teams)
        for a in self.world.agents:
            if a.alive: alive_agent_each_team[a.team] += 1
        # sum(alive_agent_each_team>0)<=1 : only one team is still standing
        terminate_cond = sum(alive_agent_each_team>0)<=1 or \
            (self.world.time_step == self.world.max_time_steps-1)
        WinLoseRewardScale = 10
        if terminate_cond:
            team_ranking = self.get_rank(alive_agent_each_team)
            WinningResult = {
                "team_ranking": team_ranking,
                "end_reason": 'EndGame'
            }
            # 2 team
            #   Top 1: reward +10
            #   Top 2: reward +0
            # 3 team
            #   Top 1: reward +20
            #   Top 2: reward +10
            #   Top 3: reward +0
            # draw: reward -10

            for i, rank in enumerate(team_ranking):
                if rank >= 0: 
                    ExtraReward[i] = ((self.n_teams - 1) - rank) * WinLoseRewardScale
                else:
                    ExtraReward[i] = -WinLoseRewardScale
            return True, WinningResult, ExtraReward
        # otherwise not done
        return False, {}, ExtraReward


    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent)
    

    # get reward for teams
    def _get_sparse_reward(self):
        return self.scenario.sparse_reward()
    
    
    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.act = np.zeros(self.world.dim_p)
        action = [action]

        if agent.movable:
            # print('self.discrete_action_input', self.discrete_action_input) # True
            # physical action
            if self.discrete_action_input:
                agent.act = np.zeros(self.world.dim_p)     ## We'll use this now for Graph NN
                # process discrete action
                ## if action[0] == 0, then do nothing
                if action[0] == 1: agent.act[0] = +1.0
                if action[0] == 2: agent.act[0] = -1.0
                if action[0] == 3: agent.act[1] = +1.0
                if action[0] == 4: agent.act[1] = -1.0
                if action[0] == 5: agent.act[2] = +agent.max_rot
                if action[0] == 6: agent.act[2] = -agent.max_rot
                agent.can_fire = True #if action[0] == 7 else False

            else:
                if self.force_discrete_action:       
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:      ## this was begin used in PR2 Paper
                    # print('action', action)
                    agent.act[0] += action[0][1] - action[0][2]    ## each is 0 to 1, so total is -1 to 1
                    agent.act[1] += action[0][3] - action[0][4]    ## same as above
                    
                    ## simple shooting action
                    agent.can_fire = True if action[0][6]>0.5 else False   # a number greater than 0.5 would mean shoot

                    ## simple rotation model
                    agent.act[2] = 2*(action[0][5]-0.5)*agent.max_rot
            
                else:
                    agent.act = action[0]
            sensitivity = 5.0   # default if no value specified for accel
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.act[:2] *= sensitivity
            
            ## remove used actions
            action = action[1:]
        

        
        # make sure we used all elements of action
        assert len(action) == 0



    def render(self):

        if not hasattr(self, 'threejs_bridge'):
            from VISUALIZE.mcom import mcom
            self.threejs_bridge = mcom(path='TEMP/v2d_logger/', digit=8, rapid_flush=False, draw_mode='Threejs')
            self.threejs_bridge.v2d_init()
            # self.threejs_bridge.set_style('star')
            # self.threejs_bridge.set_style('grid')
            # self.threejs_bridge.set_style('grid3d')
            self.threejs_bridge.set_style('font', fontPath='/examples/fonts/ttf/FZYTK.TTF', fontLineHeight=1500) # 注意不可以省略参数键值'fontPath=','fontLineHeight=' ！！！
            # self.threejs_bridge.set_style('gray')
            self.threejs_bridge.set_style('skybox6side',    # 设置天空盒子，注意不可以省略参数键值 !!
                posx='/wget/snow_textures/posx.jpg',   
                negx='/wget/snow_textures/negx.jpg',   
                posy='/wget/snow_textures/negy.jpg',
                negy='/wget/snow_textures/posy.jpg',
                posz='/wget/snow_textures/posz.jpg',
                negz='/wget/snow_textures/negz.jpg',
            )
            self.threejs_bridge.其他几何体之旋转缩放和平移('tower', 'BoxGeometry(1,1,1)',   0,0,0,  0.25,0.25,0.25, 0,0,-3) # 长方体
            self.threejs_bridge.advanced_geometry_material('tower', 
                map='/wget/hex_texture.jpg',
            )
            self.threejs_bridge.time_cnt = 0
            self.threejs_bridge.其他几何体之旋转缩放和平移('tower2', 'BoxGeometry(1,1,1)',   0,0,0,  0,0,5, 0,0,-4) # 长方体

            self.threejs_bridge.geometry_rotate_scale_translate('box',   0, 0,       0,       3, 2, 1,         0, 0, 0)
            self.threejs_bridge.geometry_rotate_scale_translate('cone',  0, np.pi/2, 0,       1.2, 0.9, 0.9,   1.5,0,0.5) # x -> y -> z
            self.threejs_bridge.terrain_theta=0
            for _ in range(20):
                # 等待加载字体
                self.threejs_bridge.空指令()
                self.threejs_bridge.v2d_show()

        if self.world.render_reset_flag:
            self.world.render_reset_flag = False
            self.threejs_bridge.set_env('clear_everything')
            self.threejs_bridge.v2d_show()

        t = self.threejs_bridge.time_cnt
        self.threejs_bridge.time_cnt += 1

        n_circle_mark = 32
        phase = np.arange(start=t/20, stop=t/20+np.pi * 2, step=np.pi * 2 / n_circle_mark)
        d = self.world.current_cir_size
        for i in range(n_circle_mark):
            offset = 1000+i
            if self.s_cfg.introduce_terrain:
                z = self.world.get_terrain(np.array([[np.sin(phase[i])*d, np.cos(phase[i])*d]]), theta=self.world.init_theta, A=self.s_cfg.terrain_parameters[0], B=self.s_cfg.terrain_parameters[1])
                z = z[0]
            else:
                z = 0
            self.threejs_bridge.v2dx(f'tower|{offset}|White|0.15', np.sin(phase[i])*d, np.cos(phase[i])*d, z+0.1, ro_x=0, ro_y=0, ro_z=t/20,label_bgcolor='Aqua',
                label='', label_offset = np.array([0,0,0.15]), label_color='Indigo', opacity=0.8)

        # self.threejs_bridge.v2dx('tower|1001|%s|0.15'%('White'), 5, -5, 1.5, ro_x=0, ro_y=0, ro_z=t/20,label_bgcolor='Aqua',
        #     label='', label_offset = np.array([0,0,0.15]), label_color='Indigo', opacity=0.8)
        # self.threejs_bridge.v2dx('tower|1002|%s|0.15'%('White'), -5, 5, 1.5, ro_x=0, ro_y=0, ro_z=t/20,label_bgcolor='Aqua',
        #     label='', label_offset = np.array([0,0,0.15]), label_color='Indigo', opacity=0.8)
        # self.threejs_bridge.v2dx('tower|1003|%s|0.15'%('White'), -5, -5, 1.5, ro_x=0, ro_y=0, ro_z=t/20,label_bgcolor='Aqua',
        #     label='', label_offset = np.array([0,0,0.15]), label_color='Indigo', opacity=0.8)

        show_lambda = 2
        
        if self.threejs_bridge.terrain_theta != self.world.init_theta:
            self.threejs_bridge.terrain_theta = self.world.init_theta
            terrain_A = self.s_cfg.terrain_parameters[0]
            terrain_B = self.s_cfg.terrain_parameters[1]
            self.threejs_bridge.set_env('terrain', theta=self.world.init_theta, terrain_A=terrain_A, terrain_B=terrain_B, show_lambda=show_lambda)

        n_red= len([0 for agent in self.world.agents if agent.alive and agent.team==1])
        n_blue = len([0 for agent in self.world.agents if agent.alive and agent.team==0])
        n_green = len([0 for agent in self.world.agents if agent.alive and agent.team==2])
        reward_blue = "%.2f"%self.scenario.reward_acc[0]
        reward_red = "%.2f"%self.scenario.reward_acc[1]
        reward_green = "%.2f"%self.scenario.reward_acc[2]
        # who_is_winning = '<Blue>Blue<Black> is leading' if n_blue > n_red else '<Red>Red<Black> is leading'
        who_is_winning = ''
        self.threejs_bridge.v2dx('tower2|1104|Gray|0.2', 0, 2, 1, ro_x=0, ro_y=0, ro_z=0, label_bgcolor='GhostWhite',
            label=  f'<Blue>Blue<Black>Agents Remain: <Blue>{n_blue}<Black>,Reward:<Blue>{reward_blue}\n' + 
                    f'<Red>Red<Black>Agents Remain: <Red>{n_red}<Black>,Reward:<Red>{reward_red} \n'
                    f'<Green>Green<Black>Agents Remain: <Green>{n_green}<Black>,Reward:<Green>{reward_green}\n<End>', 
            label_color='DarkGreen', opacity=0)

        _color = ['blue', 'red', 'green']
        _base_color = ['CornflowerBlue', 'LightPink', 'green']
        _dead_color = ["#000033", "#330000", '#003300']
        _flash_color = ['DeepSkyBlue', 'Magenta', 'green']
        for index, agent in enumerate(self.world.agents):
            x = agent.pos[0]; y = agent.pos[1]
            dir_ = dir2rad(agent.vel)
            color = _color[agent.team]
            base_color = _base_color[agent.team]
            if not agent.alive:
                base_color = color = _dead_color[agent.team]

            size = 0.025 if agent.alive else 0.01

            self.threejs_bridge.v2dx(
                'cone|%d|%s|%.3f'%(agent.iden, color, size),
                x, y, (agent.terrain-1)*show_lambda,
                ro_x=0, ro_y=0, ro_z=agent.atk_rad, 
                label='',
                label_color='white', attack_range=0, opacity=1)
            self.threejs_bridge.v2dx(
                'box|%d|%s|%.3f'%(agent.iden+500, base_color, size),
                x, y, (agent.terrain-1)*show_lambda-0.025,
                ro_x=0, ro_y=0, ro_z=dir_,  # Euler Angle y-x-z
                label='', label_color='white', attack_range=0, opacity=1)

            if agent.wasHitBy is not None:
                flash_type = 'lightning'
                flash_color = _flash_color[agent.wasHitBy.team]
                self.threejs_bridge.flash(flash_type, src=agent.wasHitBy.iden, dst=agent.iden, dur=0.2, size=0.03, color=flash_color)
                agent.wasHitBy = None
                
        # if self.s_cfg.Terrain_DEBUG:
        #     # 临时 - 调试地形
        #     # 计算双方距离

        #     # blue_pos = np.array([agent.pos for agent in self.world.agents if agent.team==0])
        #     # red_pos = np.array([agent.pos for agent in self.world.agents if agent.team==1])
        #     # distance = distance_matrix_AB(blue_pos, red_pos)
        #     for blue_agent in [agent for agent in self.world.agents if agent.team==0]:
        #         for red_agent in [agent for agent in self.world.agents if agent.team==1]:



        #             dis = np.linalg.norm(red_agent.pos - blue_agent.pos)


        #             if dis <= blue_agent.shootRad*blue_agent.terrain:
        #                 self.threejs_bridge.发送线条(
        #                     'simple|2000%s|MidnightBlue|0.03'%(str(blue_agent.iden)+'-'+str(red_agent.iden)), # 填入核心参量： “simple|线条的唯一ID标识|颜色|整体大小”
        #                     x_arr=np.array([blue_agent.pos[0],          red_agent.pos[0],        ]),   # 曲线的x坐标列表
        #                     y_arr=np.array([blue_agent.pos[1],          red_agent.pos[1],        ]),   # 曲线的y坐标列表
        #                     z_arr=np.array([(blue_agent.terrain-1)*show_lambda,   (red_agent.terrain-1)*show_lambda, ]),   # 曲线的z坐标列表
        #                     tension=0,  # 曲线的平滑度，0为不平滑，推荐不平滑
        #                 )
        #                 agent = blue_agent
        #                 x = agent.pos[0]; y = agent.pos[1]
        #                 dir_ = dir2rad(agent.vel)
        #                 color = 'red' if agent.team==1 else 'blue'
        #                 base_color = 'LightPink' if agent.team==1 else 'CornflowerBlue'
        #                 if not agent.alive:
        #                     color = "#330000" if agent.team==1 else "#000033"
        #                     base_color = "#330000" if agent.team==1 else "#000033"
        #                 size = 0.025 if agent.alive else 0.01

        #                 self.threejs_bridge.v2dx(
        #                     'cone|%d|%s|%.3f'%(agent.iden, color, size),
        #                     x, y, (agent.terrain-1)*show_lambda, label_bgcolor='Black',
        #                     ro_x=0, ro_y=0, ro_z=agent.atk_rad,  # Euler Angle y-x-z
        #                     label='T %.3f, R %.3f, D %.3f'%(agent.terrain, agent.shootRad*agent.terrain, dis), label_color='white', attack_range=0, opacity=1)

        #             if dis <= red_agent.shootRad*red_agent.terrain:
        #                 self.threejs_bridge.发送线条(
        #                     'simple|3000%s|Pink|0.03'%(str(red_agent.iden)+'-'+str(blue_agent.iden)), # 填入核心参量： “simple|线条的唯一ID标识|颜色|整体大小”
        #                     x_arr=np.array([red_agent.pos[0],           blue_agent.pos[0],        ]),   # 曲线的x坐标列表
        #                     y_arr=np.array([red_agent.pos[1],           blue_agent.pos[1],        ]),   # 曲线的y坐标列表
        #                     z_arr=np.array([(red_agent.terrain-1)*show_lambda+0.03,    (blue_agent.terrain-1)*show_lambda+0.03, ]),   # 曲线的z坐标列表
        #                     tension=0,  # 曲线的平滑度，0为不平滑，推荐不平滑
        #                 )
        #                 agent = red_agent
        #                 x = agent.pos[0]; y = agent.pos[1]
        #                 dir_ = dir2rad(agent.vel)
        #                 color = 'red' if agent.team==1 else 'blue'
        #                 base_color = 'LightPink' if agent.team==1 else 'CornflowerBlue'
        #                 if not agent.alive:
        #                     color = "#330000" if agent.team==1 else "#000033"
        #                     base_color = "#330000" if agent.team==1 else "#000033"
        #                 size = 0.025 if agent.alive else 0.01

        #                 self.threejs_bridge.v2dx(
        #                     'cone|%d|%s|%.3f'%(agent.iden, color, size),
        #                     x, y, (agent.terrain-1)*show_lambda, label_bgcolor='Black',
        #                     ro_x=0, ro_y=0, ro_z=agent.atk_rad,  # Euler Angle y-x-z
        #                     label='T %.3f, R %.3f, D %.3f'%(agent.terrain, agent.shootRad*agent.terrain, dis), label_color='white', attack_range=0, opacity=1)

        self.threejs_bridge.v2d_show()

