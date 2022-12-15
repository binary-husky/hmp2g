import time, gym
import numpy as np
from UTIL.tensor_ops import my_view, get_binary_n_rows
from ..core import World, Agent
PI = np.pi

class collective_assultEnvV1(gym.Env):  
    def __init__(self):
        from ..collective_assult_parallel_run import ScenarioConfig
        self.init_dis = ScenarioConfig.init_distance
        self.half_death_reward = ScenarioConfig.half_death_reward
        self.random_jam_prob = ScenarioConfig.random_jam_prob
        self.world = World() 
        size = ScenarioConfig.size
        self.world.wall_pos=[-1*size,1*size,-1*size,1*size]
        self.world.init_box=[-1*5,1*5,-1*5,1*5]
        self.world.numAgents = sum(ScenarioConfig.N_AGENT_EACH_TEAM)
        self.n_teams = len(ScenarioConfig.N_AGENT_EACH_TEAM)
        self.N_AGENT_EACH_TEAM = ScenarioConfig.N_AGENT_EACH_TEAM
        self.AGENT_ID_EACH_TEAM = ScenarioConfig.AGENT_ID_EACH_TEAM
        for team, team_agent_uids in enumerate(ScenarioConfig.AGENT_ID_EACH_TEAM):
            for tid, uid in enumerate(team_agent_uids):
                agent = Agent(iden=uid)
                agent.collide = False
                agent.collide_wall = True
                agent.silent = True
                agent.accel = 3
                agent.max_speed = 1.0   #
                agent.max_rot = 0.17 ## a
                agent.uid = uid # unique ID in game
                agent.tid = tid # unique ID in team
                agent.team = team 
                self.world.agents.append(agent)

        self.world.time_step = 0
        #  set inside malib/environments/collective_assult 最大步数为100 在外围初始化
        self.world.max_time_steps = None 
        self.reset_world()        


    def reset_world(self):
        self.world.render_reset_flag = True
        self.world.time_step = 0
        self.world.bullets = [] ##
        self.world.numAliveAgents = self.world.numAgents
        theta_terrain = (2*np.random.rand()-1)*np.pi # -PI ~ +PI
        self.world.init_theta = theta_terrain
        self.reward_acc = np.array([0.]*self.n_teams)
        random_theta = (2*np.random.rand()-1)*np.pi # -PI ~ +PI

        xMin, xMax, yMin, yMax = self.world.init_box
        xMid = xMin/2 + xMax/2
        yMid = yMin/2 + yMax/2
        xInitDis = self.init_dis
        row = 5
        row_dis = 0.1

        for _, agent in enumerate(self.world.agents):
            agent.alive = True
            agent.vel = np.zeros(self.world.dim_p-1)    ##
            n_agent_in_this_team = self.N_AGENT_EACH_TEAM[agent.team]

            team_pos_dir = (2*PI/self.n_teams) * agent.team + random_theta
            team_rotate = np.array([[np.cos(team_pos_dir), -np.sin(team_pos_dir)], [np.sin(team_pos_dir), np.cos(team_pos_dir)]])
            team_face_dir = team_pos_dir + PI
            
            x_ = xMid+ xInitDis/2 + row_dis * (agent.tid%row)
            y_ = yMid+ ( (agent.tid//row )/ (n_agent_in_this_team/row) - 0.5) *  (yMax-yMin) * 0.25
            agent.pos = np.dot((x_,y_), team_rotate.T)
            noise = (np.random.rand()-0.5)/12
            agent.atk_rad = team_face_dir + noise


            agent.hit = False
            agent.wasHit = False

    def sparse_reward(self):
        reward_each_team = []
        total_agent_num = sum(self.N_AGENT_EACH_TEAM)
        for team, uid_list in enumerate(self.AGENT_ID_EACH_TEAM):
            team_agents = [self.world.agents[i] for i in uid_list]
            hit_list_cnt = sum([a.hit for a in team_agents if (a.alive or a.justDied)])
            be_hit_list_cnt = sum([a.wasHit for a in team_agents if (a.alive or a.justDied)])
            # print(f"{hit_list_cnt},{be_hit_list_cnt}")
            reward_team = hit_list_cnt / total_agent_num
            reward_team -= be_hit_list_cnt / total_agent_num
            reward_each_team.append(reward_team)
            assert len(reward_each_team) == team+1
        self.reward_acc += np.array(reward_each_team)
        return reward_each_team

    def reward(self, agent):
        main_reward = 0
        if agent.alive or agent.justDied:
            if agent.hit:
                main_reward += 1
            if agent.wasHit:
                main_reward = -1 if not self.half_death_reward else -0.5
        return main_reward

    raw_obs_size = -1
    class raw_obs_array(object):
        def __init__(self):
            if collective_assultEnvV1.raw_obs_size==-1:
                self.a_group = []
                self.nosize = True
            else:
                self.a_group = np.zeros(shape=(collective_assultEnvV1.raw_obs_size), dtype=np.float32)
                self.nosize = False
                self.p = 0

        def append(self, buf):
            if self.nosize:
                self.a_group.append(buf)
            else:
                L = len(buf)
                self.a_group[self.p:self.p+L] = buf[:]
                self.p += L

        def get(self):
            if self.nosize:
                self.a_group = np.concatenate(self.a_group)
                collective_assultEnvV1.raw_obs_size = len(self.a_group)
            return self.a_group

        

    @staticmethod
    def item_random_mv(src,dst,prob,rand=False):
        assert len(src.shape)==1; assert len(dst.shape)==1
        if rand: np.random.shuffle(src)
        len_src = len(src)
        n_mv = (np.random.rand(len_src) < prob).sum()
        item_mv = src[range(len_src-n_mv,len_src)]
        src = src[range(0,0+len_src-n_mv)]
        dst = np.concatenate((item_mv, dst))
        return src, dst

    def prepare_common_data_for_obs(self):
        self.obs_arr = self.raw_obs_array()
        n_row = max(self.N_AGENT_EACH_TEAM)
        bi_hot = get_binary_n_rows(n_row, n_bit=8)
        for agent in self.world.agents:
            self.obs_arr.append([agent.alive])
            self.obs_arr.append(agent.pos)
            self.obs_arr.append([agent.atk_rad])
            self.obs_arr.append(agent.vel)
            self.obs_arr.append([agent.iden])
            self.obs_arr.append([agent.terrain])
            self.obs_arr.append(bi_hot[agent.tid])
            self.obs_arr.append([agent.team])
        return my_view(self.obs_arr.get().astype(np.float32), [self.world.numAgents, -1])

    
    def observation(self, agent, world, get_obs_dim=False):
        if get_obs_dim: return 12*17

        if agent.iden == 0:
            self.obs = self.prepare_common_data_for_obs()
            self.dec = {'alive':0, 
                        'pos':range(1,3), 
                        'ang':3, 
                        'vel':range(4,6), 
                        'id':6, 
                        'terrain':7, 
                        'bi_hot':range(8, 16),
                        'team':16,
                        }
            self.obs_range = 2.0
            self.dis = distance_matrix(self.obs[:,self.dec['pos']])
            # set almost inf distance for dead agents
            self.alive_all = np.array([a.alive for a in self.world.agents])

            self.dis[~self.alive_all,:] = +np.inf
            self.dis[:,~self.alive_all] = +np.inf


        ally_uid = self.AGENT_ID_EACH_TEAM[agent.team]
        foe_uid = []
        for team, uid_list in enumerate(self.AGENT_ID_EACH_TEAM):
            if team!=agent.team:
                foe_uid.extend(uid_list)

        ally_alive = self.alive_all[ally_uid]
        foe_alive = self.alive_all[foe_uid]
        # A_id = agent.iden
        a2h_dis = self.dis[agent.uid, foe_uid] # self.blue2red_dis[A_id]
        a2f_dis = self.dis[agent.uid, ally_uid] # self.blue2blue_dis[A_id]
        ally_emb = self.obs[ally_uid]
        foe_emb = self.obs[foe_uid]

        vis_n = 6
        h_iden_sort = np.argsort(a2h_dis)[:vis_n] 
        f_iden_sort = np.argsort(a2f_dis)[:vis_n] 
        if not agent.alive:
            agent_obs = np.zeros(shape=(ally_emb.shape[-1] *vis_n*2,))
            info_n = {'vis_f': None, 'vis_h':None, 'alive': False}
            return agent_obs, info_n


        # observe hostile:: dis array([4, 6, 3, 5, 2, 7])  shuf array([5, 2, 3, 6, 7, 4])
        a2h_dis_sorted = a2h_dis[h_iden_sort]
        hostile_vis_mask = (a2h_dis_sorted<=self.obs_range) & (foe_alive[h_iden_sort])
        vis_index = h_iden_sort[hostile_vis_mask]
        invis_index = h_iden_sort[~hostile_vis_mask]
        vis_index, invis_index = self.item_random_mv(src=vis_index, dst=invis_index,prob=self.random_jam_prob, rand=True)
        _ind = np.concatenate((vis_index, invis_index))
        _msk = np.concatenate((vis_index<0, invis_index>=0)) # "<0" project to False; ">=0" project to True
        a2h_sort = foe_emb[_ind]
        a2h_sort[_msk] = 0
        a2h_sort_filtered = a2h_sort
        
        a2f_dis_sorted = a2f_dis[f_iden_sort]
        friend_vis_mask = (a2f_dis_sorted<=(self.obs_range*1.5)) & (ally_alive[f_iden_sort])
        vis_index = f_iden_sort[friend_vis_mask]
        self_index = vis_index[:1]  # 自身的索引
        vis_index = vis_index[1:]  # 可见友方的索引
        invis_index = f_iden_sort[~friend_vis_mask] # 不可见友方的索引
        vis_index, invis_index = self.item_random_mv(src=vis_index, dst=invis_index,prob=self.random_jam_prob, rand=True)
        _ind = np.concatenate((self_index, vis_index, invis_index))
        _msk = np.concatenate((self_index<0, vis_index<0, invis_index>=0)) # "<0" project to False; ">=0" project to True
        a2f_sort = ally_emb[_ind]
        a2f_sort[_msk] = 0
        a2f_sort_filtered = a2f_sort

        agent_obs = np.concatenate((a2f_sort_filtered.flatten(), a2h_sort_filtered.flatten()))
        
        # info_n = {'vis_f': f_iden_sort, 'vis_h':h_iden_sort[a2h_dis_sorted<self.obs_range], 'alive': True}
        info_n = {}
        return agent_obs, info_n

def distance_matrix(A):
    assert A.shape[-1] == 2 # assert 2D situation
    n_subject = A.shape[-2] # is 2
    A = np.repeat(np.expand_dims(A,-2), n_subject, axis=-2) # =>(64, 100, 100, 2)
    At = np.swapaxes(A,-2,-3) # =>(64, 100, 100, 2)
    dis = At-A # =>(64, 100, 100, 2)
    dis = np.linalg.norm(dis, axis=-1)
    return dis