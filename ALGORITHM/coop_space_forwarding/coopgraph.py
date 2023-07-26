from .reinforce_foundation import CoopAlgConfig
from UTIL.colorful import *
from .my_utils import copy_clone, my_view, add_onehot_id_at_last_dim, add_obs_container_subject
from numba import njit, jit
import numpy as np
import pickle, os

class CoopGraph(object):
    def __init__(self, n_agent, n_thread, 
                    n_entity=0, 
                    load_checkpoint=False, 
                    ObsAsUnity=None,
                    agent_uid = None,
                    entity_uid = None,
                    pos_decs = None,
                    vel_decs = None,
                    logdir = None,
                    test_mode = False,
                    n_basic_dim = None,
                    ):
        self.n_basic_dim = n_basic_dim
        self.test_mode = test_mode
        self.n_agent = n_agent
        self.n_entity = n_entity
        self.n_thread = n_thread
        self.entity_uid = entity_uid
        self.ObsAsUnity = ObsAsUnity
        self.agent_uid = agent_uid
        self.pos_decs = pos_decs 
        self.vel_decs = vel_decs 
        self.logdir = logdir
        # define graph nodes
        self.n_cluster = CoopAlgConfig.g_num if not CoopAlgConfig.one_more_container else CoopAlgConfig.g_num+1
        self.n_container_R = self.n_cluster
        self.n_subject_R = self.n_agent
        self.n_container_L = self.n_entity if not CoopAlgConfig.reverse_container else self.n_cluster
        self.n_subject_L = self.n_cluster if not CoopAlgConfig.reverse_container else self.n_entity
        self.debug_cnt = 0
        # graph state
        self._Edge_R_    = np.zeros(shape=(self.n_thread, self.n_subject_R), dtype=np.int64)
        self._Edge_L_    = np.zeros(shape=(self.n_thread, self.n_subject_L), dtype=np.int64)
        self._SubFifo_R_ = np.ones(shape=(self.n_thread, self.n_container_R, self.n_subject_R), dtype=np.int64) * -1
        self._SubFifo_L_ = np.ones(shape=(self.n_thread, self.n_container_L, self.n_subject_L), dtype=np.int64) * -1

        # load checkpoint
        if load_checkpoint or self.test_mode:
            assert os.path.exists(f'{self.logdir}/history_cpt/init.pkl')
            pkl_file = open(f'{self.logdir}/history_cpt/init.pkl', 'rb')
            dict_data = pickle.load(pkl_file)
            self._Edge_R_init = dict_data["_Edge_R_init"] if "_Edge_R_init" in dict_data else dict_data["_division_obsR_init"]
            self._Edge_L_init = dict_data["_Edge_L_init"] if "_Edge_L_init" in dict_data else dict_data["_division_obsL_init"]
        else:
            self._Edge_R_init = self.__random_select_init_value_(self.n_container_R, self.n_subject_R)
            self._Edge_L_init = self.__random_select_init_value_(self.n_container_L, self.n_subject_L)
            # assert not os.path.exists(f'{self.logdir}/history_cpt/init.pkl')
            pickle.dump({"_Edge_R_init":self._Edge_R_init, "_Edge_L_init":self._Edge_L_init}, open('%s/history_cpt/init.pkl'%self.logdir,'wb+'))

    def attach_encoding_to_obs_masked(self, obs, mask):
        live_obs = obs[mask]
        Active_Edge_R    = self._Edge_R_    [mask]
        Active_Edge_L    = self._Edge_L_    [mask]
        # Active_SubFifo_R = self._SubFifo_R_ [mask]
        # Active_SubFifo_L = self._SubFifo_L_ [mask]

        _n_cluster = self.n_cluster if not CoopAlgConfig.one_more_container else self.n_cluster+1
        if self.ObsAsUnity:
            about_all_objects = live_obs
        else:
            about_all_objects = live_obs[:,0,:]
        objects_emb  = my_view(x=about_all_objects, shape=[0,-1,self.n_basic_dim]) # select one agent

        agent_pure_emb = objects_emb[:,self.agent_uid,:]
        entity_pure_emb = objects_emb[:,self.entity_uid,:]
        
        n_thread = live_obs.shape[0]
        cluster_pure_emb = np.zeros(shape=(n_thread, _n_cluster, 0)) # empty

        agent_hot_emb = add_onehot_id_at_last_dim(agent_pure_emb)
        entity_hot_emb = add_onehot_id_at_last_dim(entity_pure_emb)
        cluster_hot_emb = add_onehot_id_at_last_dim(cluster_pure_emb)
        cluster_hot_emb, agent_hot_emb  = add_obs_container_subject(container_emb=cluster_hot_emb, subject_emb=agent_hot_emb, div=Active_Edge_R)
        if not CoopAlgConfig.reverse_container:
            entity_hot_emb, cluster_hot_emb = add_obs_container_subject(container_emb=entity_hot_emb, subject_emb=cluster_hot_emb, div=Active_Edge_L)
        else:
            cluster_hot_emb, entity_hot_emb = add_obs_container_subject(container_emb=cluster_hot_emb, subject_emb=entity_hot_emb, div=Active_Edge_L)

        agent_final_emb = agent_hot_emb
        entity_final_emb = entity_hot_emb
        cluster_final_emb = cluster_hot_emb

        agent_emb  = objects_emb    [:,    self.agent_uid, :]
        agent_pos  = agent_emb      [:, :, self.pos_decs    ]
        agent_vel  = agent_emb      [:, :, self.vel_decs    ]
        entity_pos = entity_pure_emb[:, :, self.pos_decs    ]
        entity_vel = entity_pure_emb[:, :, self.vel_decs    ]

        all_emb = {
            'agent_final_emb':agent_final_emb,      # for RL
            'entity_final_emb': entity_final_emb,   # for RL
            'cluster_final_emb': cluster_final_emb, # for RL

        }
        act_dec = {
            'agent_pos': agent_pos,   # for decoding action
            'agent_vel': agent_vel,   # for decoding action
            'entity_pos': entity_pos,  # for decoding action
            'entity_vel': entity_vel  # for decoding action
        }
        return all_emb, act_dec

    def adjust_edge(self, Active_action, mask, hold):
        hold_n = hold[mask]
        assert Active_action.shape[0] == len(hold_n)
        container_actR = copy_clone(Active_action[:,(0,1)])
        container_actL = copy_clone(Active_action[:,(2,3)])

        Active_Edge_R    = self._Edge_R_    [mask]
        Active_Edge_L    = self._Edge_L_    [mask]
        Active_SubFifo_R = self._SubFifo_R_ [mask]
        Active_SubFifo_L = self._SubFifo_L_ [mask]

        Active_Edge_R, Active_SubFifo_R = self.swap_according_to_aciton(container_actR, div=Active_Edge_R, fifo=Active_SubFifo_R, hold_n=hold_n)
        Active_Edge_L, Active_SubFifo_L = self.swap_according_to_aciton(container_actL, div=Active_Edge_L, fifo=Active_SubFifo_L)

        self._Edge_R_    [mask] = Active_Edge_R    
        self._Edge_L_    [mask] = Active_Edge_L    
        self._SubFifo_R_ [mask] = Active_SubFifo_R 
        self._SubFifo_L_ [mask] = Active_SubFifo_L 


    def get_graph_encoding_masked(self, mask):
        return  copy_clone(self._Edge_R_   [mask]), \
                copy_clone(self._Edge_L_   [mask]), \
                copy_clone(self._SubFifo_R_[mask]), \
                copy_clone(self._SubFifo_L_[mask])

    def link_agent_to_target(self):
        if not CoopAlgConfig.reverse_container:
            cluster_entity_div = self._Edge_L_   # 每个cluster在哪个entity容器中
        else:   # figure out cluster_entity_div with fifo # 每个cluster指向那个entity
            cluster_entity_div = np.ones(shape=(self.n_thread, self.n_cluster), dtype=np.int64) * self.n_entity #point to n_entity+1
            for thread, jth_cluster, pos in np.argwhere(self._SubFifo_L_ >= 0):
                cluster_entity_div[thread, jth_cluster] = self._SubFifo_L_[thread, jth_cluster, pos]    # 指向队列中的最后一个目标
            if CoopAlgConfig.one_more_container: 
                cluster_entity_div[:,self.n_cluster] = self.n_entity

        agent_entity_div = np.take_along_axis(cluster_entity_div, axis=1, indices=self._Edge_R_)
        final_indices = np.expand_dims(agent_entity_div, axis=-1).repeat(3, axis=-1)

        return final_indices

    def random_disturb(self, prob, mask):
        random_hit = (np.random.rand(self.n_thread) < prob)
        ones = np.ones_like(random_hit, dtype=np.int64)
        mask = mask&random_hit
        if not any(mask): return
        Active_action = np.zeros(shape=(self.n_thread, 4), dtype=np.int64)

        for procindex in range(self.n_thread):
            # 无随机扰动
            if not random_hit[procindex]: continue
            # 有随机扰动
            r_act = np.random.choice( a=range(self.n_container_R), size=(2), replace=False, p=None) # 不放回采样
            l_act = np.random.choice( a=range(self.n_container_L), size=(2), replace=False, p=None) # 不放回采样
            Active_action[procindex,:] = np.concatenate((r_act,l_act))

        self.adjust_edge(Active_action[mask], mask=mask, hold=ones)
        self.debug_cnt += 1
        # print(self.debug_cnt)
        return mask

    def reset_terminated_threads(self, just_got_reset):
        for procindex in range(self.n_thread):
            if not just_got_reset[procindex]: 
                continue # otherwise reset

            if CoopAlgConfig.use_fixed_random_start:
                assert self._Edge_R_init is not None
                _Edge_R_ = self._Edge_R_init
                _Edge_L_ = self._Edge_L_init
            else:
                assert False

            self._Edge_R_[procindex,:] = _Edge_R_
            self._Edge_L_[procindex,:] = _Edge_L_
            for container in range(self.n_container_R):
                self._SubFifo_R_[procindex,container] = np.ones(self.n_subject_R) *-1
                index_ = np.where(_Edge_R_ == container)[0]
                self._SubFifo_R_[procindex,container,:len(index_)] = index_
            for container in range(self.n_container_L):
                self._SubFifo_L_[procindex,container] = np.ones(self.n_subject_L) *-1
                index_ = np.where(_Edge_L_ == container)[0]
                self._SubFifo_L_[procindex,container,:len(index_)] = index_
            pass


    @staticmethod
    def __random_select_init_value_(n_container, n_subject):
        t_final = []; entropy = np.array([])
        for _ in range(20): # max entropy in samples
            tmp = np.random.randint(low=0, high=n_container, size=(n_subject,), dtype=np.int64); t_final.append(tmp)
            entropy = np.append(entropy, sum([ -(sum(tmp==i)/n_subject)*np.log(sum(tmp==i)/n_subject) if sum(tmp==i)!=0 else -np.inf for i in range(n_container)]))
        return t_final[np.argmax(entropy)]


    @staticmethod
    @njit
    def swap_according_to_aciton(act, div, fifo, hold_n=None):
        def push(vec, item):
            insert_pos=0; len_vec = len(vec)
            while insert_pos<len_vec and vec[insert_pos]!=-1: insert_pos+=1
            assert insert_pos < len_vec
            vec[insert_pos] = item
        def pop(vec):
            p = vec[0]; assert p>=0
            vec[:-1]=vec[1:]; vec[-1] = -1
            return p
        n_thread = act.shape[0]
        if hold_n is None:
            hold_n = np.ones((n_thread,), np.int_)
        act_switch_1 = act[:, 0]
        act_switch_2 = act[:, 1]
        for 目标组, 移除组, i in zip(act_switch_1, act_switch_2, range(n_thread)):
            if 目标组 == 移除组:  continue
            else:
                for _ in range(hold_n[i]):     # check this       
                    移除组智能体成员 = np.where(div[i] == 移除组)[0]
                    if len(移除组智能体成员) == 0: continue  # 已经是空组别
                    转移体 = pop(fifo[i, 移除组])
                    div[i, 转移体] = 目标组
                    push(fifo[i, 目标组], 转移体)
        new_div = div
        new_fifo = fifo

        return new_div, new_fifo

    def render_thread0_graph(self, 可视化桥, step, internal_step):
        if 可视化桥 is None: return
        agent_cluster = self._Edge_R_[0]
        cluster_target = self._Edge_L_[0]

        可视化桥.发送几何体(
            'box|0|green|0.3',
            2,
            0,
            -1,
            ro_x=0, ro_y=0, ro_z=0,  # Euler Angle y-x-z
            label=f'step {step}, internal_step {internal_step}', 
            label_color='BlueViolet', 
            opacity=0
        )

        for i in range(self.n_agent):
            uid = i+100
            y = i - self.n_agent/2 # in range (0~self.n_agent)
            可视化桥.发送几何体(
                f'box|{uid}|Red|0.1',     # 填入 ‘形状|几何体之ID标识|颜色|大小’即可
                0, y, 0, ro_x=0, ro_y=0, ro_z=0,    # 三维位置+欧拉旋转变换，六自由度
                track_n_frame=0)              

        for i in range(self.n_cluster):
            uid = i+200
            y = i - self.n_cluster/2 # in range (0~self.n_agent)
            可视化桥.发送几何体(
                f'box|{uid}|Blue|0.1',     # 填入 ‘形状|几何体之ID标识|颜色|大小’即可
                2, y, 0, ro_x=0, ro_y=0, ro_z=0,    # 三维位置+欧拉旋转变换，六自由度
                track_n_frame=0)                

        for i in range(self.n_agent):
            uid = i+300
            y = i - self.n_agent/2 # in range (0~self.n_agent)
            可视化桥.发送几何体(
                f'box|{uid}|Green|0.1',     # 填入 ‘形状|几何体之ID标识|颜色|大小’即可
                4, y, 0, ro_x=0, ro_y=0, ro_z=0,    # 三维位置+欧拉旋转变换，六自由度
                track_n_frame=0)


        for i in range(self.n_agent):
            # uid = i+400
            agent_uid = i+100
            cluster_uid = agent_cluster[i]+200
            可视化桥.发射光束(
                'beam',         # 有 beam 和 lightning 两种选择
                src=agent_uid,   # 发射者的几何体的唯一ID标识
                dst=cluster_uid,  # 接收者的几何体的唯一ID标识
                dur=0.1,        # 光束持续时间，单位秒，绝对时间，不受播放fps的影响
                size=0.03,      # 光束粗细
                color='DeepSkyBlue' # 光束颜色
            )


        for i in range(self.n_cluster):
            # uid = i+500
            cluster_uid = i+200
            target_uid = cluster_target[i]+300
            可视化桥.发射光束(
                'beam',         # 有 beam 和 lightning 两种选择
                src=cluster_uid,   # 发射者的几何体的唯一ID标识
                dst=target_uid,  # 接收者的几何体的唯一ID标识
                dur=0.1,        # 光束持续时间，单位秒，绝对时间，不受播放fps的影响
                size=0.03,      # 光束粗细
                color='DeepSkyBlue' # 光束颜色
            )
        可视化桥.结束关键帧()
