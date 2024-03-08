from typing import List
from ..agent import Agent
from ..env_cmd import CmdEnv
from UTIL.colorful import *
from UTIL.tensor_ops import dir2rad, np_softmax, reg_deg_at, repeat_at, reg_rad
from .maneuver import maneuver_cold_to_ms, maneuver_vertical_to_ms, maneuver_angle_to_ms
import copy
import random
import numpy as np
import time
from .tools import distance_matrix


# 定义了地图中一些特殊点和量
class Special():
    NE = np.array([125000, 125000])
    N = np.array([0, 125000])
    E = np.array([125000, 0])
    NW = np.array([-125000, 125000])
    SE = np.array([125000, -125000])
    SW = np.array([-125000, -125000])

    init_X = 125000
    init_Y = 0
    init_Z = 9900

    init_speed_vip = 400
    init_speed_drone = 300 * 0.6

    # Dive_Horizental = 10000  # 躲导弹时，下潜点的水平距离
    Dive_Z = 5000

# 无人机的参量
class Drone():
    MIN_X = -145e3
    MAX_X = 145e3
    MIN_Y = -145e3
    MAX_Y = 145e3
    
    MaxSpeed = 300
    MinSpeed = 101
    FollowSpeed = 300
    MaxHeight = 10000
    EscapeHeight = 5000
    MinHeight = 2000
    MaxAcc = 2
    MaxOverload = 12

    # 无人机雷达
    #     1) 探测方位范围： [-30 度, 30 度]
    #     2) 探测俯仰范围： [-10 度, 10 度]
    #     3) 探测距离范围： 60 公里

    RadarHorizon = 30
    RadarVertical = 30
    RadarDis = 60e3



    AttackRadarHorizon = 20
    AttackRadarVertical = 10
    AttackRadarDis = 45e3

    DeadZooRadarHorizon = 12
    DeadZooRadarVertical = 12
    DeadZooRadarDis = 20e3

    prepare_escape_distance = 12e3
    escape_angle = 180

    Flying_to_distance = 55e3
    DeadZooRadarHorizon = 3


# 有人机的参量
class Vip():
    MIN_X = -145e3
    MAX_X = 145e3
    MIN_Y = -145e3
    MAX_Y = 145e3

    MaxSpeed = 400
    MinSpeed = 151
    FollowSpeed = 400
    MaxHeight = 15000
    EscapeHeight = 5000
    MinHeight = 2000
    MaxAcc = 1
    MaxOverload = 6

    # 有人机雷达
    #     1) 探测方位范围： [-60 度, 60 度]
    #     2) 探测俯仰范围： [-60 度, 60 度]
    #     3) 探测距离范围： 80 公里


    RadarHorizon = 60
    RadarVertical = 60
    RadarDis = 80e3

    AttackRadarHorizon = 50
    AttackRadarVertical = 50
    AttackRadarDis = 60e3

    DeadZooRadarHorizon = 50
    DeadZooRadarVertical = 50
    DeadZooRadarDis = 20e3

    prepare_escape_distance = 30e3
    escape_distance = 1.8e3

    Flying_to_distance = 55e3
    DeadZooRadarHorizon = 3

    # emergent_escape_close_distance = 1.3e3


# 飞机的类
class Plane(object):
    dis_mat = None
    dis_mat_id2index = None
    # self.all_ms = []
    def __init__(self, data=None, manager=None) -> None:
        self.manager = manager
        super().__init__()
        if data is not None: self.update_info(data, sim_time=0, init=True)
        self.attacked_by_ms = []
        self.previous_ms_list = []
        self.latest_update_time = 0
        self.KIA = False

    def update_info(self, data, sim_time, init=False):
        self.latest_update_time = sim_time
        self.alive = (data['Availability'] != 0)
        self.data = data
        if hasattr(self, 'ID'):
            assert data['ID'] == self.ID
        for key in data:
            setattr(self, key, data[key])

        # first update
        if not hasattr(self, 'is_drone'):
            self.is_drone = (self.Type == 2)
            self.is_vip = (self.Type == 1)
            Ability = Drone if self.is_drone else Vip
            for key in Ability.__dict__:
                if '__' in key: continue
                setattr(self, key, getattr(Ability, key))
        self.pos3d = np.array([self.X, self.Y, self.Alt], dtype=float)
        self.pos2d = np.array([self.X, self.Y], dtype=float)
        self.h_angle = 90-self.Heading*180/np.pi

        if init and ('LeftWeapon' not in data):
            self.OpLeftWeapon = 4 if self.is_vip else 2
            self.PlayerHeightSetting = 9999 # z >= 9000 and z <= 10000

    def incoming_msid(self):
        return 0

    def step_init(self):
        self.previous_ms_list = copy.copy(self.attacked_by_ms)
        self.attacked_by_ms = []
        self.wish_list_by = []
        self.my_ms = []
        self.fired_ms = []  # 发射出去，正在飞行的导弹
        self.step_state = ''
        self.alive = False  # 如果没有被置 True....

    def get_dis(self, target_id):
        index1 = Plane.dis_mat_id2index[self.ID]
        index2 = Plane.dis_mat_id2index[target_id]
        return Plane.dis_mat[index1, index2]

    def delta_oppsite_to_ms(self):
        delta_pos = np.array([self.pos3d[:2] - np.array([ms['X'], ms['Y']], dtype=float)
                              for ms in self.attacked_by_ms])
        dis = np.array([self.get_dis(ms['ID']) for ms in self.attacked_by_ms])
        delta_dir = delta_pos / repeat_at(dis, -1, 2)
        weight = np_softmax(dis.max() - dis)
        delta = (delta_dir * repeat_at(weight, -1, 2)).sum(0)
        # math_rad_dir = dir2rad(delta)
        return delta / (np.linalg.norm(delta)+1e-7)

    # 获取距离最近的导弹是哪个
    def nearest_ms(self):
        # 距离最近导弹在 self.attacked_by_ms 中的index
        if len(self.attacked_by_ms)==0: return None
        index = np.argmin(np.array([ self.get_dis(ms['ID']) for ms in self.attacked_by_ms ]))
        return self.manager.find_ms_by_id(self.attacked_by_ms[index]['ID'])

    def ms_number_changed(self):
        return len(self.previous_ms_list) != len(self.attacked_by_ms)

    def pos2d_prediction(self):
        heading_rad = self.h_angle*np.pi/180
        return self.pos2d + 10e3*np.array([np.cos(heading_rad), np.sin(heading_rad)])

    def get_nearest_op_with_ms(self):
        dis = np.array([self.get_dis(op.ID) for op in self.manager.op_planes if op.OpLeftWeapon>0])
        if len(dis)>0:
            return self.manager.op_planes[np.argmin(dis)]
        else:
            return None

    def get_nearest_threat(self):
        res = None
        threat_dis = 9999e3
        ms = self.nearest_ms()
        nearest_op = self.get_nearest_op_with_ms()
        if ms is not None and threat_dis > self.get_dis(ms.ID):
            res = ms
            threat_dis = self.get_dis(ms.ID)
        if nearest_op is not None and threat_dis > self.get_dis(nearest_op.ID):
            res = nearest_op
            threat_dis = self.get_dis(nearest_op.ID)
        return res, threat_dis



# 导弹的类
class MS(object):
    def __init__(self, data) -> None:
        super().__init__()

        # import os
        # folder = './log'
        # if not os.path.exists(folder):  #判断是否存在文件夹如果不存在则创建为文件夹
        #     os.makedirs(folder)

        self.previous_pos3d = None
        self.previous_speed = None
        self.ms_at_terminal = False     # 导弹是否进入末制导
        self.tracking_target = True     # 导弹是否在跟踪目标
        self.delta_traj = []
        self.target_speed = []
        self.self_speed = []
        self.distance = []
        self.time = []
        self.delta_dis = []
        self.ms_speed = []
        self.flying_time = 0
        self.previous_change_taking_effect = 0

        self.speed_peak = False
        if data is not None: self.update_info(data, 0)
        self.flying_dis = 0

        self.debug_estimate_next_pos = None
        self.debug_estimate_uav_next_pos = None
        self.latest_update_time = 0
        self.KIA = False

    def estimate_terminal_dis(self):
        p_dis = self.distance[-1]
        p_time = self.flying_time
        p_ddis = self.D_list[p_time] if p_time < len(self.D_list) else self.D_list[-1]
        self.dis_arr = []
        while True:
            tmp = p_dis + p_ddis
            if tmp < 0:
                self.dis_arr.append(tmp)
                return p_dis, self.speed_list[p_time] if p_time < len(self.speed_list) else self.speed_list[-1]
            p_dis = tmp
            p_time += 1
            p_ddis = self.D_list[p_time] if p_time < len(self.D_list) else self.D_list[-1]
            self.dis_arr.append(p_dis)

    @staticmethod
    def interpolant(x, list):
        len_list = len(list)
        left = -1
        right = len_list
        for i in range(len_list):
            if x > list[i][0]:
                left = i
            else:
                break
        for i in reversed(range(len_list)):
            if x < list[i][0]:
                right = i
            else:
                break
        # ## print(list[left], list[right])
        if left == -1:
            assert right == 0
            left += 1
            right += 1
        if right == len_list:
            assert left == len_list - 1
            left -= 1
            right -= 1
        assert list[right][0] != list[left][0]
        return (x - list[left][0]) / (list[right][0] - list[left][0]) * (list[right][1] - list[left][1]) + list[left][1]

    def init_d_list(self):
        for i, speed in enumerate(self.speed_list):
            if speed >= 999.0: return

            self.D_list[i] = self.interpolant(self.speed_list[i], self.speed2ddis)

    def init_speed_list(self, speed):
        self.speed_list[0] = speed
        pointer = 1
        while True:
            self.speed_list[pointer] = self.speed_list[pointer - 1] + 98.066000
            if self.speed_list[pointer] >= 1000:
                self.speed_list[pointer] = 1000
                return
            pointer += 1
            if pointer > 10:
                return

    def update_info(self, data, sim_time):
        self.latest_update_time = sim_time

        self.alive = (data['Availability'] != 0)
        self.data = data
        if hasattr(self, 'ID'):
            assert data['ID'] == self.ID
        for key in data:
            setattr(self, key, data[key])

        self.pos3d = np.array([self.X, self.Y, self.Alt], dtype=float)
        self.pos2d = np.array([self.X, self.Y], dtype=float)
        if self.previous_pos3d is not None:
            self.flying_dis += np.linalg.norm(self.pos3d - self.previous_pos3d)

        self.is_op_ms = hasattr(self.target, 'OpLeftWeapon')
        if self.target is None: 
            self.is_op_ms = (not hasattr(self.host, 'OpLeftWeapon'))
            return  # 对应的敌方（我方）目标已经死了

        self.delta_traj.append(self.target.pos3d - self.pos3d)
        self.distance.append(np.linalg.norm(self.target.pos3d - self.pos3d))
        self.ms_at_terminal = (self.distance[-1] < 20e3)    # 当与目标的距离小于20km时，进入末制导
        min_distance = min(self.distance)   # 导弹飞行历程中，距离目标的最小距离
        # 是否打空，变成无效导弹
        self.tracking_target = False if self.distance[-1] > (min_distance + 1e3) else True
        if not self.tracking_target: self.ms_at_terminal = True # 如果已经变成无效导弹，也设置成末制导状态，毕竟不需要再管它了
        self.target_speed.append(self.target.Speed)
        self.target_speed.append(self.Speed)
        self.h_angle = 90 - self.Heading * 180 / np.pi
        if self.Speed >= 1000:
            self.speed_peak = True
        # if self.flying_time == 0:
            # self.init_speed_list(self.Speed)
            # self.init_d_list()
        starget_dir = dir2rad(self.target.pos2d - self.pos2d) * 180 / np.pi
        starget_dir = reg_deg_at(starget_dir, ref=self.h_angle)

        if len(self.distance) >= 2:
            self.time.append(self.flying_time)
            self.delta_dis.append(self.distance[-1] - self.distance[-2])
            self.ms_speed.append(self.Speed)
        # self.ter_dis_est, self.ter_ms_speed = self.estimate_terminal_dis()

        ### print亮红('ID', self.ID, 'impact dst warning! ', self.ter_dis_est)
        # if (not hasattr(self.target,'OpLeftWeapon')) and self.tracking_target: # 我方是目标
        #     with open('./log/%s'%str(self.ID), 'a+') as f:
        #         f.write('导弹速度 %.2f, 目标距离 %.2f, T估计 %.2f, %s \n'%(self.Speed, self.distance[-1], self.ter_dis_est, str(self.dis_arr)))
        # self.impact_eta = len(self.dis_arr) - 1

        # self.debug_estimate_next_pos = self.pos3d.copy()
        # self.debug_estimate_next_pos[2] += self.Speed * np.sin(self.Pitch)
        # self.debug_estimate_next_pos[0] += self.Speed * np.cos(self.Pitch) * np.cos(self.h_angle * np.pi / 180)
        # self.debug_estimate_next_pos[1] += self.Speed * np.cos(self.Pitch) * np.sin(self.h_angle * np.pi / 180)

        # self.debug_estimate_uav_next_pos = self.target.pos3d.copy()
        # self.debug_estimate_uav_next_pos[2] += self.target.Speed * np.sin(self.target.Pitch)
        # self.debug_estimate_uav_next_pos[0] += self.target.Speed * np.cos(self.target.Pitch) * np.cos(
        #     self.target.h_angle * np.pi / 180)
        # self.debug_estimate_uav_next_pos[1] += self.target.Speed * np.cos(self.target.Pitch) * np.sin(
        #     self.target.h_angle * np.pi / 180)


        self.previous_speed = self.Speed
        self.previous_pos3d = self.pos3d
        pass

    def step_init(self):
        self.flying_time += 1
        self.alive = False

    def end_life(self):
        if self.target is None: return  # 对应的敌方（我方）目标已经死了
        import sys, logging
        # ## print亮靛('alive:',self.target.alive)
        # if (not hasattr(self.target,'OpLeftWeapon')) and self.tracking_target: # 我方是目标
        #     with open('./log/%s'%str(self.ID), 'a+') as f:
        #         f.write('命中速度 %.2f \n'%self.Speed)
        #         f.write('存活？ %s \n'%str(self.target.alive))
        #         f.write('*************************\n')
        #     if self.target.alive:
        #         import os
        #         os.remove('./log/%s'%str(self.ID))
        # ## print("".join([ "%.3f, "%d for d in self.ms_speed]))
        return


class Baseclass(Agent):
    def __init__(self, name, config):
        super(Baseclass, self).__init__(name, config["side"])
        self.init()

    def init(self):
        self.n_uav = 4
        self.cmd_list = []
        self.state_recall = {}
        self.STATE = 'Full_assault'
        self.individual_state = ['assault'] * self.n_uav
        self.id_mapping = None
        self.Name_mapping = None
        self.my_planes = None
        self.op_planes = None
        self.ms = []
        self.escape_angle = 100  # self.conf['escape_angle']
        self.vip_escape_angle = 0  # self.conf['escape_angle']
        self.escape_distance = 1.3e3  # self.conf['escape_distance']
        self.prepare_escape_distance = 10e3
        self.obs_side_1 = []
        self.our_vantage = False
        self.coop_list = None
        self.Fleet_Attack = False
        # self.vip_escape_distance = 60e3  # self.conf['escape_distance']
        # self.vip_prepare_escape_distance = 75e3

        self.initial_attack_order_adjusted = False
        self.Id2PlaneLookup = {}
        self.Id2MissleLookup = {}

        self.Name2PlaneLookup = {}


    def find_plane_by_name(self, name):
        if name in self.Name2PlaneLookup:
            if (self.Name2PlaneLookup[name].Name == name):
                if self.Name2PlaneLookup[name].KIA:
                    return None
                else:
                    return self.Name2PlaneLookup[name]

        # otherwise, no match, or dictionary with no record
        for p in self.my_planes + self.op_planes:
            if p.Name == name: 
                self.Name2PlaneLookup[name] = p # register
                return p

        # otherwise, no match at all
        return None
        

    def find_planes_by_squad(self, squad_name):
        return [p for p in self.my_planes  if p.squad_name == squad_name]


    # add a ID->plane obj table
    def find_plane_by_id(self, ID):
        if ID in self.Id2PlaneLookup:
            if (self.Id2PlaneLookup[ID].ID == ID):
                if self.Id2PlaneLookup[ID].KIA:
                    return None
                else:
                    return self.Id2PlaneLookup[ID]

        # otherwise, no match, or dictionary with no record
        for p in self.my_planes + self.op_planes:
            if p.ID == ID: 
                self.Id2PlaneLookup[ID] = p # register
                return p

        # otherwise, no match at all
        return None




    def find_ms_by_id(self, ID):
        if ID in self.Id2MissleLookup:
            if (self.Id2MissleLookup[ID].ID == ID):
                if self.Id2MissleLookup[ID].KIA:
                    return None
                else:
                    return self.Id2MissleLookup[ID]
        for ms in self.ms:
            if ms.ID == ID: 
                self.Id2MissleLookup[ID] = ms
                return ms
                
        return None

    def reset(self, **kwargs):
        self.init()
        pass

    # 加载观测量
    def my_process_observation_and_show(self, obs_side, sim_time):

        # need init?
        if self.my_planes is None:
            self.my_planes = [Plane(manager=self) for _ in range(self.n_uav +1)]
            self.op_planes = [Plane(manager=self) for _ in range(self.n_uav +1)]

            # register info for the first time!
            for idx, uvas_info in enumerate(obs_side['platforminfos']):
                self.my_planes[idx].update_info(uvas_info, sim_time, init=True)
            for idx, uvas_info in enumerate(obs_side['trackinfos']):
                self.op_planes[idx].update_info(uvas_info, sim_time, init=True)

            if len(obs_side['platforminfos']) != (self.n_uav + 1):
                print亮红('Error! 没有从起始状态执行,可能导致一系列错误!')
                self.my_planes = list(filter(lambda p: hasattr(p, 'ID'), self.my_planes))

            if len(obs_side['trackinfos']) != (self.n_uav + 1):
                print亮红('Error! 没有从起始状态执行,可能导致一系列错误!')
                self.op_planes = list(filter(lambda p: hasattr(p, 'ID'), self.op_planes))

            for obj in self.my_planes + self.op_planes + self.ms:
                obj.step_init()

            # self.init_attack_order()

        # set every plane object to step_init
        for obj in self.my_planes + self.op_planes + self.ms:
            obj.step_init()

        everything = []
        # part 1
        my_entity_infos = obs_side['platforminfos']
        enemy_entity_infos = obs_side['trackinfos']
        if len(my_entity_infos) < 1:
            return {
                'my_plane_fallen': 0,
                'op_plane_fallen': 0,
                'my_ms_miss_dead': 0,
                'op_ms_miss_dead': 0,
                'my_ms_miss_alive': 0,
                'op_ms_miss_alive': 0,
                'my_ms_hit': 0,
                'op_ms_hit': 0,
            }
        for uvas_info in (my_entity_infos + enemy_entity_infos):
            if not (uvas_info['ID'] != 0 and uvas_info['Availability'] > 0.0001): continue 
            uvas_info["Z"] = uvas_info["Alt"]
            p = self.find_plane_by_id(uvas_info['ID'])
            if p is None:
                continue
            p.update_info(uvas_info, sim_time)
            everything.append(uvas_info) # for grand dis matrix calculation


        # part 2
        missile_infos = obs_side['missileinfos']
        for missile_info in missile_infos:
            ms_active = (missile_info['ID'] != 0 and missile_info['Availability'] > 0.0001)
            assert ms_active, ('when will ms be in-active?')
            if not ms_active: continue
            missile_info["Z"] = missile_info["Alt"]
            host_id = missile_info['LauncherID']
            target_id = missile_info['EngageTargetID']
            host = self.find_plane_by_id(host_id)
            if host is not None:
                host.my_ms.append(missile_info)
            target = self.find_plane_by_id(target_id)
            if target is not None:
                target.attacked_by_ms.append(missile_info)
            # if host_id in target.hanging_attack_by: target.hanging_attack_by.remove(host_id)    # hanging_attack_by
            ms = self.find_ms_by_id(missile_info["ID"])
            if ms is None:
                missile_info.update({ 'host':host, 'target':target })
                self.ms.append(MS(missile_info))
                if host is not None and hasattr(host, 'OpLeftWeapon'):
                    host.OpLeftWeapon -= 1
            else:
                ms.update_info(missile_info, sim_time)

            ms = self.find_ms_by_id(missile_info["ID"])
            if host is not None:
                assert ms is not None
                host.fired_ms.append(ms)

            everything.append(missile_info) # for grand dis matrix calculation

        # part 3 distance matrix
        everything_pos = np.array([np.array([p['X'], p['Y'], p['Z']]) for p in everything])
        self.id2index = {p['ID']: i for i, p in enumerate(everything)}
        self.active_index = self.id2index
        self.everything_dis = distance_matrix(everything_pos)
        Plane.dis_mat = self.everything_dis
        Plane.dis_mat_id2index = self.id2index

        for m in self.ms:
            if not m.alive: 
                m.KIA = True
                m.end_life()

        for p in (self.my_planes+self.op_planes):
            if not p.alive: p.KIA = True

        for p in self.my_planes:
            p.in_radar = self.get_in_radar_op(p)

        for op in self.op_planes:
            op.in_radar = self.get_in_radar_me(op)

        
        return self.reward_related_info(sim_time)





    def reward_related_info(self, sim_time):
        reward_related_info = {
            'my_plane_fallen': 0,
            'op_plane_fallen': 0,
            'my_ms_miss_dead': 0,
            'op_ms_miss_dead': 0,
            'my_ms_miss_alive': 0,
            'op_ms_miss_alive': 0,
            'my_ms_hit': 0,
            'op_ms_hit': 0,
        }
        # <1> reward on fallen fighters
        # if a plane does not update its info, it's down
        for p in self.my_planes:
            if (p.latest_update_time == sim_time-1):
                # just be destoried
                reward_related_info["my_plane_fallen"] += 1
        for op in self.op_planes:
            if (op.latest_update_time == sim_time-1):
                # just be destoried
                reward_related_info["op_plane_fallen"] += 1

        # <2> reward on escaping ms
        for ms in self.ms:
            if (ms.latest_update_time == sim_time-1):
                # ms hit or miss
                # if ms.target.alive:
                #     print('missle miss')
                # else 
                target_dead = (not ms.target.alive)
                if target_dead and (ms.target.latest_update_time == sim_time-1):
                    # nice shoot
                    if ms.is_op_ms:
                        reward_related_info['op_ms_hit'] += 1
                    else:
                        reward_related_info['my_ms_hit'] += 1
                elif target_dead:
                    # target is already dealt with by other ms
                    if ms.is_op_ms:
                        reward_related_info['op_ms_miss_dead'] += 1
                    else:
                        reward_related_info['my_ms_miss_dead'] += 1
                elif ms.target.alive:
                    # print('missle miss')
                    # ms miss!
                    if ms.is_op_ms:
                        reward_related_info['op_ms_miss_alive'] += 1
                    else:
                        reward_related_info['my_ms_miss_alive'] += 1
                else:
                    assert False, "pretty sure there cannot be other possibilities"

        return reward_related_info

    def get_in_radar_me(self, op): #
        in_radar = [] #
        if op.KIA: return in_radar
        for p in self.my_planes: # 
            if p.KIA: continue
            dis = self.get_dis(p.ID, op.ID) #
            if dis > op.RadarDis: #
                continue
            # 检查角度是否满足
            delta = p.pos2d - op.pos2d  #
            theta = 90 - op.Heading * 180 / np.pi   #
            theta = reg_rad(theta*np.pi/180)
            delta2 = np.matmul(np.array([[np.cos(theta), np.sin(theta)],
                                         [-np.sin(theta), np.cos(theta)]]), delta.T)
            deg = dir2rad(delta2) * 180 / np.pi
            if deg >= -op.RadarHorizon and deg <= op.RadarHorizon:
                in_radar.append(p)
        return in_radar

    # 获取进入飞机雷达范围内的敌机
    def get_in_radar_op(self, p):
        in_radar = []
        if p.KIA: return in_radar
        for op in self.op_planes:
            if op.KIA: continue
            dis = self.get_dis(op.ID, p.ID)
            if dis > p.RadarDis:
                continue
            # 检查角度是否满足
            delta = op.pos2d - p.pos2d
            theta = 90 - p.Heading * 180 / np.pi
            theta = reg_rad(theta*np.pi/180)
            delta2 = np.matmul(np.array([[np.cos(theta), np.sin(theta)],
                                         [-np.sin(theta), np.cos(theta)]]), delta.T)
            deg = dir2rad(delta2) * 180 / np.pi
            if deg >= -p.RadarHorizon and deg <= p.RadarHorizon:
                in_radar.append(op)
        return in_radar


    def id2index(self, arr):
        return [self.id2index[id_] for id_ in arr]

    def get_dis(self, id1, id2):
        index1 = self.id2index[id1]
        index2 = self.id2index[id2]
        return self.everything_dis[index1, index2]

    def observe(self, sim_time, obs_side, **kwargs) -> List[dict]:
        return self.my_process_observation_and_show(obs_side, sim_time)

    def process_decision(self, time, obs_side):
        self.time_of_game = time
        if time == 3: self.init_pos(self.cmd_list)
        if time >= 4: self.make_decision()
        return

    # 初始化位置
    def init_pos(self, cmd_list):
        '''
            y     [-150000, +150000]
            red_x [-150000, -125000]
            blue_x [125000,  150000]
        '''
        Special.init_Y = (2*np.random.rand() - 1)*60000  # -125000 #
        leader_original_pos = {}  # 用以初始化当前方的位置
        if self.name == "red":
            # leader_original_pos = {"X": -Special.init_X, "Y": Special.init_Y, "Z": Special.init_Z}
            leader_original_pos = {"X": -Special.init_X, "Y": -Special.init_Y, "Z": Special.init_Z}
            init_dir = 90
        else:
            leader_original_pos = {"X": Special.init_X, "Y": Special.init_Y, "Z": Special.init_Z}
            init_dir = 360 - 90

        interval_distance = 20000   # 间隔 5000米排列
        sub_index = 0

        for p in self.my_planes:
            if p.is_vip:
                cmd_list.append(
                    CmdEnv.make_entityinitinfo(
                        p.ID,
                        leader_original_pos['X'], leader_original_pos['Y'], leader_original_pos['Z'],
                        Special.init_speed_vip, init_dir))
            elif p.is_drone:
                sub_pos = copy.deepcopy(leader_original_pos)
                offset = 0 if sub_index<=1 else 1e3
                if sub_index & 1 == 0:
                    cmd_list.append(
                        CmdEnv.make_entityinitinfo(p.ID,
                                                   sub_pos['X'], sub_pos['Y'] + interval_distance + offset,
                                                   sub_pos['Z'], Special.init_speed_drone, init_dir))
                else:
                    cmd_list.append(
                        CmdEnv.make_entityinitinfo(p.ID,
                                                   sub_pos['X'], sub_pos['Y'] - interval_distance - offset,
                                                   sub_pos['Z'], Special.init_speed_drone, init_dir))
                    # interval_distance *= 2 # 编号翻倍
                sub_index += 1
            else:
                assert False, ('???')

    # 初始化攻击序列，游戏开始后，会二次调整
    def init_attack_order(self):
        a_plane = self.my_planes[0]
        if "蓝" in a_plane.Name:
            color = "蓝"
            op_color = "红"
        elif "红" in a_plane.Name:
            color = "红"
            op_color = "蓝"
        else:
            assert False
        self.color = color
        self.op_color = op_color
        attack_order = {
            color + "无人机1": [
                op_color + "无人机1",
                op_color + "无人机2",
                op_color + "有人机",
            ],
            color + "无人机2": [
                op_color + "无人机1",
                op_color + "无人机2",
                op_color + "有人机",
            ],
            color + "无人机3": [
                op_color + "无人机4",
                op_color + "无人机3",
                op_color + "有人机",
            ],
            color + "无人机4": [
                op_color + "无人机4",
                op_color + "无人机3",
                op_color + "有人机",
            ],
            color + "有人机": [
                op_color + "无人机1",
                op_color + "无人机2",
                op_color + "无人机3",
                op_color + "无人机4",
                op_color + "有人机",
            ],
        }

        attack_squad = {
            color + "有人机": {
                "squad": "U1",
                "save_last_ammo": False,
                "leader": color + "无人机1",
                "formation_mate": [color + "无人机1", color + "无人机2"]
            },
            color + "无人机1": {
                "squad": "U1",
                "save_last_ammo": False,
                "leader": None,
                "formation_mate": [color + "无人机2", color + "有人机"]
            },
            color + "无人机2": {
                "squad": "U1",
                "save_last_ammo": False,
                "leader": color + "无人机1",
                "formation_mate": [color + "无人机1", color + "有人机"]
            },
            color + "无人机3": {
                "squad": "U2",
                "save_last_ammo": True,
                "leader": None,
                "formation_mate": [color + "无人机4"]
            },
            color + "无人机4": {
                "squad": "U2",
                "save_last_ammo": False,
                "leader": color + "无人机3",
                "formation_mate": [color + "无人机3"]
            },

        }
        for p in self.my_planes:
            p.attack_order = attack_order[p.Name]
            p.squad_name = attack_squad[p.Name]["squad"]
            p.squad_leader = attack_squad[p.Name]["leader"]
            p.formation_mate = attack_squad[p.Name]["formation_mate"]
            p.save_last_ammo = attack_squad[p.Name]["save_last_ammo"]
        pass

    def check_and_make_linepatrolparam(self, receiver, coord_list, cmd_speed, cmd_accmag, cmd_g, force_old_way=False):
        host = self.find_plane_by_id(receiver)
 
        def old_way(point):
            if point['X'] > host.MAX_X:
                point['X'] = host.MAX_X          #; ## print红('if point[X] > host.MAX_X: point[X] = host.MAX_X;')
            if point['X'] < host.MIN_X:
                point['X'] = host.MIN_X          #; ## print红('if point[X] < host.MIN_X: point[X] = host.MIN_X;')
            if point['Y'] > host.MAX_Y: 
                point['Y'] = host.MAX_Y          #; ## print红('if point[Y] > host.MAX_Y: point[Y] = host.MAX_Y;')
            if point['Y'] < host.MIN_Y: 
                point['Y'] = host.MIN_Y          #; ## print红('if point[Y] < host.MIN_Y: point[Y] = host.MIN_Y;')
            if point['Z'] < host.MinHeight: 
                point['Z'] = host.MinHeight      #; ### print红('if point[Z] < host.MinHeight: point[Z] = host.MinHeight;')
            if point['Z'] > host.MaxHeight: 
                point['Z'] = host.MaxHeight      #; ### print红('if point[Z] > host.MaxHeight: point[Z] = host.MaxHeight;')
            return point

        def avail_coord(point):
            if point['X'] > host.MAX_X:
                return False
            if point['X'] < host.MIN_X:
                return False
            if point['Y'] > host.MAX_Y: 
                return False
            if point['Y'] < host.MIN_Y: 
                return False
            if point['Z'] < host.MinHeight: 
                return False
            if point['Z'] > host.MaxHeight: 
                return False
            return True
        def avail_coord_np(point):
            if point[0] > host.MAX_X:
                return False
            if point[0] < host.MIN_X:
                return False
            if point[1] > host.MAX_Y: 
                return False
            if point[1] < host.MIN_Y: 
                return False
            if point[2] < host.MinHeight: 
                return False
            if point[2] > host.MaxHeight: 
                return False
            return True

        for i, point in enumerate(coord_list):
            if avail_coord(point): continue
            if force_old_way:
                coord_list[i] = old_way(point)
                continue
            # 坐标需要修正！！
            arr = np.array([point['X'],point['Y'],point['Z']])
            vec_dir_3d = arr - host.pos3d # 从host指向arr
            vec_dir_3d_unit = vec_dir_3d / (np.linalg.norm(vec_dir_3d)+1e-7)
            res_len = self.prob_len(
                starting_point=host.pos3d,
                direction=vec_dir_3d_unit,
                dx = 100,
                lamb = avail_coord_np,
                max_try = 1000, # 100km 
            )
            res_avail = res_len*vec_dir_3d_unit + host.pos3d
            if res_len < 300:
                # ???? 这种情况比较危险，采用旧的方式
                coord_list[i] = old_way(point)
            else:
                coord_list[i] = old_way({"X": res_avail[0], "Y": res_avail[1], "Z": res_avail[2]})
        return CmdEnv.make_linepatrolparam(receiver, coord_list, cmd_speed, cmd_accmag, cmd_g)


    @staticmethod
    def prob_len(starting_point, direction, dx, lamb, max_try):
        for i in range(max_try):
            dst = starting_point + direction*dx*(i+1)
            if lamb(dst):
                continue
            else:
                return dx*(i+1)
        return dx*max_try

    def has_overall_adv(self):
        def delta_oppsite_to_op(p):
            # 获取敌方飞机的相反方向
            # 修正1：不对无弹的敌方飞机做出响应
            delta_pos = np.array([p.pos2d - op.pos2d for op in self.op_planes if op.OpLeftWeapon>0])
            dis = np.array([p.get_dis(op.ID) for op in self.op_planes if op.OpLeftWeapon>0])
            delta_dir = delta_pos / repeat_at(dis, -1, 2)
            weight = np_softmax(dis.max() - dis)
            delta = (delta_dir * repeat_at(weight, -1, 2)).sum(0)
            # math_rad_dir = dir2rad(delta)
            return delta / (np.linalg.norm(delta)+1e-7)

        def get_nearest_op_with_ms(p):
            delta_pos = np.array([p.pos2d - op.pos2d for op in self.op_planes if op.OpLeftWeapon>0])
            dis = np.array([p.get_dis(op.ID) for op in self.op_planes if op.OpLeftWeapon>0])
            if len(dis)>0:
                return self.op_planes[np.argmin(dis)]
            else:
                return None

        all_op_ammo = sum([op.OpLeftWeapon for op in self.op_planes])   # 所有敌方导弹的总和
        plane_num_adv = len(self.my_planes) - len(self.op_planes)       # 我战机数量减敌战机数量
        vip_p = [p for p in self.my_planes][0]

        if plane_num_adv > 0:
            near_op = get_nearest_op_with_ms(vip_p)
            if near_op is None:
                #
                return False, None

            delta = delta_oppsite_to_op(vip_p)

            # circle_evade：局部滞回变量
            if not hasattr(vip_p, 'circle_evade'):
                vip_p.circle_evade = False
            # 滞回变量状态转换
            # how_far_to_wall = self.prob_len(
            #         starting_point=vip_p.pos2d, 
            #         direction=delta/(np.linalg.norm(delta)+1e-7), 
            #         dx=5e3, 
            #         lamb=lambda x: (np.abs(x[0])<145e3 and np.abs(x[1])<145e3), 
            #         max_try=50)
            
            if vip_p.circle_evade:
                if np.linalg.norm(vip_p.pos2d) < 110e3:
                    vip_p.circle_evade = False
            else:
                if np.linalg.norm(vip_p.pos2d) > 125e3:
                    vip_p.circle_evade = True
            # 滞回变量作用
            if vip_p.circle_evade:
                delta_vertical_01 = np.cross([delta[0], delta[1], 0], [0,0,1] )[:2]
                delta_vertical_02 = np.cross([delta[0], delta[1], 0], [0,0,-1] )[:2]
                delta_vertical_01 = delta_vertical_01/(np.linalg.norm(delta_vertical_01)+1e-7)
                delta_vertical_02 = delta_vertical_02/(np.linalg.norm(delta_vertical_02)+1e-7)
                r = self.thresh_hold_projection(np.linalg.norm(vip_p.pos2d), 
                        min_x=110e3, y_min_x=0.0, 
                        max_x=140e3, y_max_x=1.0)
                delta_01 = delta*(1-r) + delta_vertical_01*r
                delta_02 = delta*(1-r) + delta_vertical_02*r
                # 计算两种规避方式的回避余地
                delta_space_01 = self.prob_len(starting_point=vip_p.pos2d, 
                    direction=delta_vertical_01, 
                    dx=5e3, 
                    lamb=lambda x: (np.abs(x[0])<145e3 and np.abs(x[1])<145e3), 
                    max_try=50)
                delta_space_02 = self.prob_len(starting_point=vip_p.pos2d, 
                    direction=delta_vertical_02, 
                    dx=5e3, 
                    lamb=lambda x: (np.abs(x[0])<145e3 and np.abs(x[1])<145e3), 
                    max_try=50)
                if delta_space_01 > delta_space_02:
                    delta = delta_01
                else:
                    delta = delta_02



            H2 = delta * 100e3 + vip_p.pos2d


            goto_location = [ {
                    "X": H2[0],
                    "Y": H2[1],
                    "Z": vip_p.Z
                }]

            if vip_p.advantage_state == 'going_away':
                if self.get_dis(near_op.ID, vip_p.ID) > 75e3:
                    vip_p.advantage_state = 'default'
                    return False, None
                else:
                    return True, goto_location
            elif vip_p.advantage_state == 'default':
                if self.get_dis(near_op.ID, vip_p.ID) <= 55e3:
                    vip_p.advantage_state = 'going_away'
                    return True, goto_location
                return False, None
        else:
            return False, None





'''
有人机平台
    1) 最大速度： 400 米/秒
    2) 最小速度： 150 米/秒
    3) 最高飞行高度： 15000 米
    4) 最低飞行高度： 2000 米
    5) 最大加速度： 1G（G = 9.8 米/平方秒）
    6) 最大法向过载： 6G
    7) 挂载武器： 4 枚中距空空导弹

有人机雷达
    1) 探测方位范围： [-60 度, 60 度]
    2) 探测俯仰范围： [-60 度, 60 度]
    3) 探测距离范围： 80 公里

无人机平台
    1) 最大速度： 300 米/秒
    2) 最小速度： 100 米/秒
    3) 最高飞行高度： 10000 米
    4) 最低飞行高度： 2000 米
    5) 最大加速度： 2G（G = 9.8 米/平方秒）
    6) 最大法向过载： 12G
    7) 挂载武器： 2 枚中距空空导弹

无人机雷达
    1) 探测方位范围： [-30 度, 30 度]
    2) 探测俯仰范围： [-10 度, 10 度]
    3) 探测距离范围： 60 公里

空空导弹
    1) 最大速度： 1000 米/秒
    2) 最小速度： 400 米/秒
    3) 最高飞行高度： 30000 米
    4) 最低飞行高度： 2000 米
    5） 最大加速度： 10G（G = 9.8 米/平方秒）
    5) 最大法向过载： 20G
    6) 最大飞行时间： 120 秒
    7) 最大飞行距离： 100 公里
    8) 导引头截获距离： 20 公里
    9) 杀伤半径： 100 米
    10) 发动机工作时间： 30 秒（发动机停止工作后，导弹减速）


指令说明 make_entityinitinfo
功能说明 初始化实体的位置姿态信息
参数含义
receiver 执行指令的飞机编号（整型，举例： 11）
x 目标点 x 轴坐标（浮点型，单位：米）
y 目标点 y 轴坐标（浮点型，单位：米）
z 目标点 z 轴坐标（浮点型，单位：米）
init_speed 初始速度(浮点型， 单位：米/秒)
init_heading 初始朝向(浮点型， 单位：度， [0,360]，与正北方向的夹角)
调用和构建指令示例
Cmd = EnvCmd.make_entityinitinfo(11, 150000, -100000, 3000, 200, 30)

指令说明 make_linepatrolparam
功能说明 航线巡逻
参数含义
receiver 执行指令的飞机编号（整型，举例： 11）
coord_list
航路点列表
（每个列表元素为一个包含浮点型 x,y,z 的路径点，单位：米）
cmd_speed 巡逻速度(浮点型， 单位：米/秒)
cmd_accmag 指令加速度(浮点型， 单位：米/平方秒)
cmd_g 指令过载(浮点型，单位： G)
调用和构建指令示例
#航路点列表
coord_list = [{"x": 500, "y": 400, "z": 2000},{"x": 900, "y": 300, "z": 3000}];
#构建指令
Cmd = EnvCmd.make_linepatrolparam(11, coord_list, 250, 1.0, 3)

指令说明 make_areapatrolparam
功能说明 区域巡逻
参数含义
receiver 执行指令的飞机编号（整型，举例： 11）
x 区域中心 x 轴坐标（浮点型，单位：米）
y 区域中心 y 轴坐标（浮点型，单位：米）
z 区域中心 z 轴坐标（浮点型，单位：米）
area_length 区域长度（浮点型，单位：米）
area_width 区域宽度（浮点型，单位：米）
cmd_speed 巡逻速度（浮点型，单位：米/秒）
cmd_accmag 指令加速度(浮点型， 单位：米/平方秒)
cmd_g 指令过载(浮点型，单位： G)
调用和构建指令示例

指令说明 make_areapatrolparam
功能说明 区域巡逻
参数含义
receiver 执行指令的飞机编号（整型，举例： 11）
x 区域中心 x 轴坐标（浮点型，单位：米）
y 区域中心 y 轴坐标（浮点型，单位：米）
z 区域中心 z 轴坐标（浮点型，单位：米）
area_length 区域长度（浮点型，单位：米）
area_width 区域宽度（浮点型，单位：米）
cmd_speed 巡逻速度（浮点型，单位：米/秒）
cmd_accmag 指令加速度(浮点型， 单位：米/平方秒)
cmd_g 指令过载(浮点型，单位： G)
调用和构建指令示例

指令说明 make_followparam
功能说明 跟随机动
参数含义
receiver 执行指令的飞机编号（整型，举例： 11）
tgt_id 目标飞机编号（整型， 友方敌方飞机均可，举例： 3）
cmd_speed 指令速度（浮点型，单位：米/秒）
cmd_accmag 指令加速度(浮点型， 单位：米/平方秒)
cmd_g 指令过载(浮点型，单位： G)
调用和构建指令示例
Cmd = EnvCmd. make_followparam(11, 3, 200, 1.0, 4.0)

指令说明 make_attackparam
功能说明 打击目标
参数含义
receiver 执行指令的飞机编号（整型，举例： 11）
tgt_id 敌方飞机编号（整型，举例： 2）
fire_range 开火范围（浮点型，最大探测范围的比例， [0,1]）
调用和构建指令示例
Cmd = EnvCmd. make_attackparam(11, 2, 0.6)
'''
