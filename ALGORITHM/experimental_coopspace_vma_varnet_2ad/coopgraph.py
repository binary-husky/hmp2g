from .reinforce_foundation import CoopAlgConfig
from UTIL.colorful import *
from UTIL.tensor_ops import my_view, repeat_at
from .my_utils import copy_clone, add_onehot_id_at_last_dim, add_obs_container_subject
from numba import njit, jit
import numpy as np
import pickle, os

def zeros_like_except_dim(array, except_dim, n):
    shape_ = list(array.shape)
    shape_[except_dim] = n
    return np.zeros(shape=shape_, dtype=array.dtype)

def pad_at_dim(array, dim, n, pad=0):
    extra_n = n - array.shape[dim]
    padding = zeros_like_except_dim(array, except_dim=dim, n=extra_n) + pad
    return np.concatenate((array, padding), axis=dim)


def one_hot_anycol_anybatch(n, n_col, batch_dims=[]):
    """
    This function generates arbitrary number, batch dimensions, and feature dimensions of one-hot encodings.

    Inputs
    n: the length of one-hot encoding.
    n_col: the column of one-hot encodings to be generated. It means there are n_col records that need one-hot encoding.
    batch_dims: the dimensional of batch. An optional parameter. The default value is an empty list, which means there is no batch dimensional.

    one_hot_anycol_anybatch(4, 6, batch_dims=[2])
    array([[[1., 0., 0., 0., 1., 0.],
            [0., 1., 0., 0., 0., 1.],
            [0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0.]],

        [[1., 0., 0., 0., 1., 0.],
            [0., 1., 0., 0., 0., 1.],
            [0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0.]]])
    """
    assert n_col >= n
    n_repeat_needed = int(np.ceil(n_col / n))
    tmp = np.tile(np.eye(n), (*batch_dims, 1, n_repeat_needed))
    return tmp[..., :n_col]

class CoopGraph(object):
    def __init__(self, n_agent, n_thread, 
                    n_entity=0, 
                    load_checkpoint=False, 
                    ObsAsUnity=None,
                    agent_uid = None,
                    sub_cluster_size = 1,
                    target_uid = None,
                    pos_decs = None,
                    vel_decs = None,
                    logdir = None,
                    test_mode = False,
                    n_basic_dim = None,
                    n_primitive_act = None,
                ):
        self.n_primitive_act = n_primitive_act
        
        self.n_basic_dim = n_basic_dim
        self.sub_cluster_size = sub_cluster_size
        self.test_mode = test_mode
        self.n_agent = n_agent
        self.n_entity = n_entity
        self.n_thread = n_thread
        self.target_uid = target_uid
        self.ObsAsUnity = ObsAsUnity
        self.agent_uid = agent_uid
        self.pos_decs = pos_decs 
        self.vel_decs = vel_decs 
        self.logdir = logdir
        # define graph nodes
        self.n_cluster = CoopAlgConfig.g_num if not CoopAlgConfig.one_more_container else CoopAlgConfig.g_num+1
        self.n_container_AC = self.n_cluster
        self.n_subject_AC = self.n_agent // self.sub_cluster_size
        self.n_container_CT = self.n_entity if not CoopAlgConfig.reverse_container else self.n_cluster
        self.n_subject_CT = self.n_cluster if not CoopAlgConfig.reverse_container else self.n_entity
        self.debug_cnt = 0
        # graph state
        self._Edge_AC_    = np.zeros(shape=(self.n_thread, self.n_subject_AC), dtype=np.int64)
        self._Edge_CT_    = np.zeros(shape=(self.n_thread, self.n_subject_CT), dtype=np.int64)
        self._SubFifo_AC_ = np.ones(shape=(self.n_thread, self.n_container_AC, self.n_subject_AC), dtype=np.int64) * -1
        self._SubFifo_CT_ = np.ones(shape=(self.n_thread, self.n_container_CT, self.n_subject_CT), dtype=np.int64) * -1

        # load checkpoint
        if load_checkpoint or self.test_mode:
            assert os.path.exists(f'{self.logdir}/history_cpt/init.pkl')
            pkl_file = open(f'{self.logdir}/history_cpt/init.pkl', 'rb')
            dict_data = pickle.load(pkl_file)
            self._Edge_AC_init = dict_data["_Edge_AC_init"] if "_Edge_AC_init" in dict_data else dict_data["_division_obs_AC_init"]
            self._Edge_CT_init = dict_data["_Edge_CT_init"] if "_Edge_CT_init" in dict_data else dict_data["_division_obs_CT_init"]
        else:
            self._Edge_AC_init = self.__random_select_init_value_(self.n_container_AC, self.n_subject_AC)
            self._Edge_CT_init = self.__random_select_init_value_(self.n_container_CT, self.n_subject_CT)
            # assert not os.path.exists(f'{self.logdir}/history_cpt/init.pkl')
            pickle.dump({"_Edge_AC_init":self._Edge_AC_init, "_Edge_CT_init":self._Edge_CT_init}, open('%s/history_cpt/init.pkl'%self.logdir,'wb+'))

    def get_current_container_size(self, Active_SubFifo):
        return (Active_SubFifo >= 0).sum(-1)




    def get_avail_act(self, mask):
        # container_AC_act = copy_clone(Active_action[:,(0,1)])
        # container_CT_act = copy_clone(Active_action[:,(2,3)])

        Active_SubFifo_AC = self._SubFifo_AC_ [mask]
        container_size_AC = self.get_current_container_size(Active_SubFifo_AC)
        act1avail = np.ones_like(container_size_AC)
        act2avail = container_size_AC > 0

        Active_SubFifo_CT = self._SubFifo_CT_ [mask]
        container_size_CT = self.get_current_container_size(Active_SubFifo_CT)
        act3avail = np.ones_like(container_size_CT)
        act4avail = container_size_CT > 0

        pad_length = max(act1avail.shape[-1], act3avail.shape[-1])

        assert act1avail.shape[-1] < act3avail.shape[-1]

        act1avail = pad_at_dim(act1avail, dim=-1, n=pad_length, pad=np.nan)
        act2avail = pad_at_dim(act2avail, dim=-1, n=pad_length, pad=np.nan)
        # act3avail = pad_at_dim(act3avail, dim=-1, n=pad_length, pad=np.nan)
        # act4avail = pad_at_dim(act4avail, dim=-1, n=pad_length, pad=np.nan)

        avail_act = np.stack((act1avail,act2avail,act3avail,act4avail), axis=1)

        return avail_act
        # Active_SubFifo_CT = self._SubFifo_CT_ [mask]

    def attach_encoding_to_obs_masked(self, obs, mask):
        ##########################################################
        Active_Edge_AC    = self._Edge_AC_    [mask]
        Active_Edge_CT    = self._Edge_CT_    [mask]
        live_obs = obs[mask]
        n_thread = live_obs.shape[0]

        _n_cluster = self.n_cluster if not CoopAlgConfig.one_more_container else self.n_cluster+1
        if self.ObsAsUnity:
            about_all_objects = live_obs
        else:
            about_all_objects = live_obs[:,0,:]
        objects_emb  = my_view(x=about_all_objects, shape=[0,-1,self.n_basic_dim]) # select one agent

        agent_pure_emb = objects_emb[:,self.agent_uid,:]
        cluster_pure_emb = np.zeros(shape=(n_thread, _n_cluster, 0)) # empty
        cluster_hot_emb = add_onehot_id_at_last_dim(cluster_pure_emb)
        if self.sub_cluster_size!=1:
            agent_pure_emb = my_view(agent_pure_emb, [0, self.n_agent//self.sub_cluster_size, self.sub_cluster_size, 0])    # (32, 30, 8)
            t = []
            for i in range(self.sub_cluster_size):
                t.append(add_onehot_id_at_last_dim(agent_pure_emb[:,:,i,:]))
            cluster_hot_emb_ = copy_clone(cluster_hot_emb)
            for i in range(self.sub_cluster_size):
                cluster_hot_emb, t[i] = add_obs_container_subject(container_emb=cluster_hot_emb_, subject_emb=t[i], div=Active_Edge_AC)
            agent_hot_emb = np.stack(t, -2)
            agent_hot_emb = my_view(agent_hot_emb, [0, 0, -1]) 
        else:
            agent_hot_emb = add_onehot_id_at_last_dim(agent_pure_emb)
            cluster_hot_emb, agent_hot_emb  = add_obs_container_subject(container_emb=cluster_hot_emb, subject_emb=agent_hot_emb, div=Active_Edge_AC)

        target_pure_emb_o = target_pure_emb = objects_emb[:,self.target_uid,:]


        prim_move_target = one_hot_anycol_anybatch(self.n_primitive_act, target_pure_emb.shape[-1], target_pure_emb.shape[:-2])
        target_pure_emb = np.concatenate((prim_move_target, target_pure_emb), -2)


        target_hot_emb = add_onehot_id_at_last_dim(target_pure_emb)
        target_hot_emb, cluster_hot_emb = add_obs_container_subject(container_emb=target_hot_emb, subject_emb=cluster_hot_emb, div=Active_Edge_CT)


        agent_final_emb = agent_hot_emb
        target_final_emb = target_hot_emb
        cluster_final_emb = cluster_hot_emb

        agent_emb  = objects_emb      [:,    self.agent_uid, :]
        agent_pos  = agent_emb        [:, :, self.pos_decs    ]
        agent_vel  = agent_emb        [:, :, self.vel_decs    ]
        target_pos = target_pure_emb_o[:, :, self.pos_decs    ]
        target_vel = target_pure_emb_o[:, :, self.vel_decs    ]

        all_emb = {
            'agent_final_emb':agent_final_emb,      # for RL
            'target_final_emb': target_final_emb,   # for RL
            'cluster_final_emb': cluster_final_emb, # for RL

        }
        act_dec = {
            'agent_pos': agent_pos,   # for decoding action
            'agent_vel': agent_vel,   # for decoding action
            'target_pos': target_pos,  # for decoding action
            'target_vel': target_vel  # for decoding action
        }
        return all_emb, act_dec

    def adjust_edge(self, Active_action, mask, hold):
        hold_n = hold[mask]
        assert Active_action.shape[0] == len(hold_n)
        container_AC_act = copy_clone(Active_action[:,(0,1)])
        container_CT_act = copy_clone(Active_action[:,(2,3)])

        Active_Edge_AC    = self._Edge_AC_    [mask]
        Active_Edge_CT    = self._Edge_CT_    [mask]
        Active_SubFifo_AC = self._SubFifo_AC_ [mask]
        Active_SubFifo_CT = self._SubFifo_CT_ [mask]

        Active_Edge_AC, Active_SubFifo_AC = self.swap_according_to_aciton(container_AC_act, div=Active_Edge_AC, fifo=Active_SubFifo_AC, hold_n=hold_n)
        Active_Edge_CT, Active_SubFifo_CT = self.swap_according_to_aciton(container_CT_act, div=Active_Edge_CT, fifo=Active_SubFifo_CT)

        self._Edge_AC_    [mask] = Active_Edge_AC    
        self._Edge_CT_    [mask] = Active_Edge_CT    
        self._SubFifo_AC_ [mask] = Active_SubFifo_AC 
        self._SubFifo_CT_ [mask] = Active_SubFifo_CT 


    def get_graph_encoding_masked(self, mask):
        return  copy_clone(self._Edge_AC_  [mask]), \
                copy_clone(self._Edge_CT_  [mask]), \
                copy_clone(self._SubFifo_AC_[mask]), \
                copy_clone(self._SubFifo_CT_[mask])

    def link_agent_to_target(self):
        if not CoopAlgConfig.reverse_container:
            cluster_target_div = self._Edge_CT_   # 每个cluster在哪个entity容器中
        else:   # figure out cluster_target_div with fifo # 每个cluster指向那个entity
            cluster_target_div = np.ones(shape=(self.n_thread, self.n_cluster), dtype=np.int64) * self.n_entity #point to n_entity+1
            for thread, jth_cluster, pos in np.argwhere(self._SubFifo_CT_ >= 0):
                cluster_target_div[thread, jth_cluster] = self._SubFifo_CT_[thread, jth_cluster, pos]    # 指向队列中的最后一个目标
            if CoopAlgConfig.one_more_container: 
                cluster_target_div[:,self.n_cluster] = self.n_entity

        agent_target_link = np.take_along_axis(cluster_target_div, axis=1, indices=self._Edge_AC_)
        # deal with sub_cluster_size
        if self.sub_cluster_size != 1:
            agent_target_link = my_view(repeat_at(agent_target_link, insert_dim=-1, n_times=self.sub_cluster_size),[0,-1])
        return agent_target_link
    

    @staticmethod
    # @jit(forceobj=True)
    def dir_to_action3d(vec, vel):
        dis = np.expand_dims(np.linalg.norm(vec, axis=2) + 1e-16, axis=-1)
        vec2 = np.cross(np.cross(vec, vel, axis=-1),vec, axis=-1)
        def np_mat3d_normalize_each_line(mat):
            return mat / np.expand_dims(np.linalg.norm(mat, axis=2) + 1e-16, axis=-1)
        # desired_speed = 0.8
        vec_dx = np_mat3d_normalize_each_line(vec)
        vec_dv = np_mat3d_normalize_each_line(vel)*0.8
        vec = np_mat3d_normalize_each_line(vec_dx+vec_dv)
        vec = np.where(dis<0.15, vec2, vec)
        return vec
    
    ##################### ######################
    def target_micro_control(self, agent_target_link, act_dec):
        # assert False, 待测试
        is_primitive_action = agent_target_link < self.n_primitive_act
        as_primitive_action = agent_target_link
        as_coop_action = agent_target_link - self.n_primitive_act

        dim = 3

        agent_target_link_primitive = np.where(is_primitive_action,  as_primitive_action,   np.nan)
        agent_target_link_coop      = np.where(~is_primitive_action, as_coop_action,        0)

        final_indices = np.expand_dims(agent_target_link_coop, axis=-1).repeat(dim, axis=-1)
        delta_pos, target_vel = self.目标解析(final_indices, act_dec)
        delta_pos[is_primitive_action] = np.nan
        target_vel[is_primitive_action] = np.nan
        all_action_coop = self.dir_to_action3d(vec=delta_pos, vel=target_vel) # 矢量指向selected entity

        all_action_primitive = np.zeros_like(all_action_coop) + np.nan
        all_action_primitive[agent_target_link_primitive == 0] = [+1,0,0]
        all_action_primitive[agent_target_link_primitive == 1] = [-1,0,0]
        all_action_primitive[agent_target_link_primitive == 2] = [0,+1,0]
        all_action_primitive[agent_target_link_primitive == 3] = [0,-1,0]
        all_action_primitive[agent_target_link_primitive == 4] = [0,0,+1]
        all_action_primitive[agent_target_link_primitive == 5] = [0,0,-1]

        all_action = np.where(
            np.expand_dims(is_primitive_action, axis=-1).repeat(dim, axis=-1), 
            all_action_primitive, 
            all_action_coop
        )
        assert not np.isnan(all_action).any()
        return all_action
    
    ##################### ######################
    def 目标解析(self, link_indices, act_dec):
        target_pos, agent_pos, target_vel = (act_dec['target_pos'], act_dec['agent_pos'], act_dec['target_vel'])

        if not CoopAlgConfig.reverse_container:
            final_sel_pos = target_pos
        else:   # 为没有装入任何entity的container解析一个nan动作
            final_sel_pos = np.concatenate( (target_pos,  np.zeros(shape=(self.n_thread, 1, 3))+np.nan ) , axis=1)
            
        sel_target_pos  = np.take_along_axis(final_sel_pos, axis=1, indices=link_indices)
        sel_target_vel  = np.take_along_axis(target_vel, axis=1, indices=link_indices)
        delta_pos = sel_target_pos - agent_pos
        return delta_pos, sel_target_vel
    ##################### ######################



    def random_disturb(self, prob, mask):
        random_hit = (np.random.rand(self.n_thread) < prob)
        ones = np.ones_like(random_hit, dtype=np.int64)
        mask = mask&random_hit
        if not any(mask): return
        Active_action = np.zeros(shape=(self.n_thread, 4), dtype=np.int64)
        avail_act = self.get_avail_act(mask=np.array([True]*self.n_thread))

        for procindex in range(self.n_thread):
            # 无随机扰动
            if not mask[procindex]: continue
            # 有随机扰动
            avail_act_thread = avail_act[procindex]
            free_to_choose = [np.where(x==1)[0] for x in avail_act_thread]
            choose_res = [np.random.choice(x) for x in free_to_choose]
            # r_act = np.random.choice( a=range(self.n_container_AC), size=(2), replace=False, p=None) # 不放回采样
            # l_act = np.random.choice( a=range(self.n_container_CT), size=(2), replace=False, p=None) # 不放回采样
            Active_action[procindex,:] = choose_res # np.concatenate((r_act,l_act))

        self.adjust_edge(Active_action[mask], mask=mask, hold=ones)
        self.debug_cnt += 1
        # print(self.debug_cnt)
        return

    def reset_terminated_threads(self, just_got_reset):
        for procindex in range(self.n_thread):
            if not just_got_reset[procindex]: 
                continue # otherwise reset

            if CoopAlgConfig.use_fixed_random_start:
                assert self._Edge_AC_init is not None
                _Edge_AC_ = self._Edge_AC_init
                _Edge_CT_ = self._Edge_CT_init
            else:
                assert False

            self._Edge_AC_[procindex,:] = _Edge_AC_
            self._Edge_CT_[procindex,:] = _Edge_CT_
            for container in range(self.n_container_AC):
                self._SubFifo_AC_[procindex,container] = np.ones(self.n_subject_AC) *-1
                index_ = np.where(_Edge_AC_ == container)[0]
                self._SubFifo_AC_[procindex,container,:len(index_)] = index_
            for container in range(self.n_container_CT):
                self._SubFifo_CT_[procindex,container] = np.ones(self.n_subject_CT) *-1
                index_ = np.where(_Edge_CT_ == container)[0]
                self._SubFifo_CT_[procindex,container,:len(index_)] = index_
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
                    if len(移除组智能体成员) == 0: 
                        assert False
                        continue  # 已经是空组别
                    转移体 = pop(fifo[i, 移除组])
                    div[i, 转移体] = 目标组
                    push(fifo[i, 目标组], 转移体)
        new_div = div
        new_fifo = fifo

        return new_div, new_fifo

    def render_thread0_graph(self, 可视化桥, step, internal_step):
        if 可视化桥 is None: return
        agent_cluster = self._Edge_AC_[0]
        cluster_target = self._Edge_CT_[0]

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

        for i in range(self.n_subject_AC):
            uid = i+100
            y = i - self.n_subject_AC/2 # in range (0~self.n_subject_AC)
            可视化桥.发送几何体(
                f'box|{uid}|Red|0.1',     # 填入 ‘形状|几何体之ID标识|颜色|大小’即可
                0, y, 0, ro_x=0, ro_y=0, ro_z=0,    # 三维位置+欧拉旋转变换，六自由度
                track_n_frame=0)              

        for i in range(self.n_cluster):
            uid = i+200
            y = i - self.n_cluster/2 # in range (0~self.n_subject_AC)
            可视化桥.发送几何体(
                f'box|{uid}|Blue|0.1',     # 填入 ‘形状|几何体之ID标识|颜色|大小’即可
                2, y, 0, ro_x=0, ro_y=0, ro_z=0,    # 三维位置+欧拉旋转变换，六自由度
                track_n_frame=0)                

        for i in range(self.n_subject_AC):
            uid = i+300
            y = i - self.n_subject_AC/2 # in range (0~self.n_subject_AC)
            可视化桥.发送几何体(
                f'box|{uid}|Green|0.1',     # 填入 ‘形状|几何体之ID标识|颜色|大小’即可
                4, y, 0, ro_x=0, ro_y=0, ro_z=0,    # 三维位置+欧拉旋转变换，六自由度
                track_n_frame=0)


        for i in range(self.n_subject_AC):
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
