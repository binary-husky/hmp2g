import numpy as np
from .agent.env_cmd import CmdEnv
from UTIL.tensor_ops import my_view
class Converter:

    planeId2IntList = {
        '红有人机':  0, '蓝有人机':  0,
        '红无人机1': 1, '蓝无人机1': 1,
        '红无人机2': 2, '蓝无人机2': 2,
        '红无人机3': 3, '蓝无人机3': 3,
        '红无人机4': 4, '蓝无人机4': 4,
    }

    def convert_obs_as_unity(self, obs, get_size=False):
        if get_size: return 20*14

        # player_obs, opp_obs = (obs[self.player_color], obs[self.opp_color])
        player_obs, _ = (obs[self.player_color], obs[self.opp_color])

        player_encoded_obs = np.zeros(shape=(20, 14), dtype=np.float32); p=0
        # planeId2Int = planeId2IntList[['Name']]
        for plane in player_obs['platforminfos']:
            player_encoded_obs[p, 0] = self.planeId2IntList[plane['Name']]
            player_encoded_obs[p, 1] = plane['Type'] - 1
            player_encoded_obs[p, 2] = plane['Availability']
            player_encoded_obs[p, 3] = plane["X"]
            player_encoded_obs[p, 4] = plane["Y"]
            player_encoded_obs[p, 5] = plane["Alt"]
            player_encoded_obs[p, 6] = plane['Heading']
            player_encoded_obs[p, 7] = plane['Pitch']
            player_encoded_obs[p, 8] = plane['Roll']
            player_encoded_obs[p, 9] = plane['Speed']
            player_encoded_obs[p, 10] = plane['CurTime']
            player_encoded_obs[p, 11] = plane['AccMag']
            player_encoded_obs[p, 12] = plane['NormalG']
            player_encoded_obs[p, 13] = plane['LeftWeapon']
            p+=1

        for plane in player_obs['trackinfos']:
            player_encoded_obs[p, 0]  = self.planeId2IntList[plane['Name']]
            player_encoded_obs[p, 1]  = plane['Type'] - 1
            player_encoded_obs[p, 2]  = plane['Availability']
            player_encoded_obs[p, 3]  = plane["X"]
            player_encoded_obs[p, 4]  = plane["Y"]
            player_encoded_obs[p, 5]  = plane["Alt"]
            player_encoded_obs[p, 6]  = plane['Heading']
            player_encoded_obs[p, 7]  = plane['Pitch']
            player_encoded_obs[p, 8]  = plane['Roll']
            player_encoded_obs[p, 9]  = plane['Speed']
            player_encoded_obs[p, 10] = plane['CurTime']
            player_encoded_obs[p, 11] = 0
            player_encoded_obs[p, 12] = 0
            player_encoded_obs[p, 13] = -1
            p+=1

        player_encoded_obs = player_encoded_obs.flatten()
        return player_encoded_obs


    def encode_individual_plane(self, arr, plane, index, offset_from=None, action_context=None):
        # self-observation
        arr[0] = index
        arr[1] = plane.Type - 1
        arr[2] = plane.Availability

        arr[3] = plane.X 
        arr[4] = plane.Y 
        arr[5] = plane.Z 
        if offset_from is not None:
            arr[3] -= offset_from.X
            arr[4] -= offset_from.Y
            arr[5] -= offset_from.Z

        arr[6] = plane.Heading
        arr[7] = plane.Pitch
        arr[8] = plane.Roll
        arr[9] = plane.Speed
        arr[10] = plane.CurTime
        arr[11] = plane.MaxAcc
        arr[12] = plane.MaxOverload
        arr[13] = plane.LeftWeapon if hasattr(plane, 'LeftWeapon') else plane.OpLeftWeapon
        if action_context is not None:
            arr[14:] = action_context
        else:
            arr[14:] = 0
        return
    
    def convert_obs_individual(self, obs, get_size=False):
        if get_size: return 20*(14+self.n_action_dimension)
        player_encoded_obs = np.zeros(shape=(5, 20, 14+self.n_action_dimension), dtype=np.float32)

        if self.player_color == "red":
            ai_planes = red_planes = self.observer.my_planes
            op_planes = blue_planes = self.observer.op_planes
        else:
            op_planes = red_planes = self.observer.my_planes
            ai_planes = blue_planes = self.observer.op_planes 
        
        for plane_observing_index, plane_observing in enumerate(ai_planes):
            pointer = 0
            self.encode_individual_plane(
                    player_encoded_obs[plane_observing_index, pointer], plane_observing, index=plane_observing_index, offset_from=None, action_context=self.action_context[plane_observing_index]
            ); pointer += 1

            for j, p in enumerate(ai_planes):
                self.encode_individual_plane(
                    player_encoded_obs[plane_observing_index, pointer], p, index=j, offset_from=plane_observing, action_context=None
                ); pointer += 1
            for j, p in enumerate(op_planes):
                self.encode_individual_plane(
                    player_encoded_obs[plane_observing_index, pointer], p, index=j+5, offset_from=plane_observing, action_context=None
                ); pointer += 1
        player_encoded_obs = my_view(player_encoded_obs, [0, -1])
        return player_encoded_obs

    def tran_target(self, TT):
        Trans = {
            "red":  ['蓝有人机','蓝无人机1','蓝无人机2','蓝无人机3','蓝无人机4', '红有人机','红无人机1','红无人机2','红无人机3','红无人机4'],
            "blue": ['红有人机','红无人机1','红无人机2','红无人机3','红无人机4', '蓝有人机','蓝无人机1','蓝无人机2','蓝无人机3','蓝无人机4',],
        }
        if TT < 10:
            target_name = Trans[self.player_color][int(TT)]
        else:
            assert False, ('here is the missile target')
        target = self.observer.find_plane_by_name(target_name)
        return target

    # 获取动作所在的组别，以及在组别内的次序
    def get_action_dim_corr_dict(self):
        if hasattr(self, 'action_dim_corr_dict'):

            return self.action_dim_corr_dict, self.action_sel_corr_dict

        self.action_dim_corr_dict = np.zeros(self.n_actions, dtype=int)
        self.action_sel_corr_dict = np.zeros(self.n_actions, dtype=int)
        tmp = [self.n_switch_actions,  self.n_target_actions]

        for j in range(len(tmp)):
            for i in range(sum(tmp[:j]),     sum(tmp[:(j+1)])): 
                self.action_dim_corr_dict[i] = j
                self.action_sel_corr_dict[i] = i - sum(tmp[:j])

        return self.action_dim_corr_dict, self.action_sel_corr_dict


    def parse_raw_action(self, raw_action):
        print('raw_actions:', raw_action)
        cmd_buffer = []
        raw_action = raw_action.astype(int)
        if not self.ScenarioConfig.use_simple_action_space:
            assert raw_action.shape[0] == self.n_agents
            assert raw_action.shape[1] == 4
        else:
            assert raw_action.shape[0] == self.n_agents
            assert len(raw_action.shape) == 1
        assert self.n_opp == self.n_agents

        if self.player_color=='red':
            player_planes, opp_planes = (self.observer.my_planes, self.observer.op_planes)
        else:
            player_planes, opp_planes = (self.observer.op_planes, self.observer.my_planes)

        next_avail_act = np.zeros(shape=(self.n_agents, self.n_actions))

        for index, p in enumerate(player_planes):
            # Assert the order of planes are correct
            if index==0: assert '有人机' in p.Name
            else: assert str(index) in p.Name

            if not self.ScenarioConfig.use_simple_action_space:
                # 
                AT = raw_action[index, self.AT_SEL]   # action type
                TT = raw_action[index, self.TT_SEL]   # target type (opp + teammate + incoming ms)
                # HT = raw_action[index, 2]   # height sel # 暂时没用
                SP = raw_action[index, self.SP_SEL]   # speed sel
            else:
                # self.action_context = np.zeros(shape=(self.n_agents, ScenarioConfig.n_action_dimension))
                action_dim_corr_dict, action_sel_corr_dict = self.get_action_dim_corr_dict()
                act = raw_action[index]

                # [0, 1, 2, 3, 4, 5]
                # [AT0, AT1, AT2, AT3, AT4, AT5]

                # [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                # [TT0, TT1, TT2, TT3, TT4, TT5, TT6, TT7, TT8, TT9]

                dim = action_dim_corr_dict[act]
                sel = action_sel_corr_dict[act]
                # previous_action_context = self.action_context.copy()
                print('previous self.action_context[index]:', self.action_context[index])
                self.action_context[index, dim] = sel
                AT = self.action_context[index, self.AT_SEL]   # action type
                TT = self.action_context[index, self.TT_SEL]   # target type (opp + teammate + incoming ms)
                print('afterward self.action_context[index]:', self.action_context[index], 'plane name:', p.Name, 'AT:',AT,'TT:',TT)

                # HT = self.action_context[index, 2]   # height sel   # # 暂时没用
                # SP = self.action_context[index, self.SP_SEL]   # speed sel
                if dim==0:
                    # chosing AT this moment, avoid AT chosing next time
                    next_avail_act[index] = action_dim_corr_dict
                    # block 
                    if AT == 5:
                        next_avail_act[index, self.ScenarioConfig.Ally_Bits_Mask] = 0 # do not shoot ally !
                    if AT >=1 and AT<=4:
                        next_avail_act[index, self.ScenarioConfig.Ally_Bits_Mask] = 0 # do not follow ally !

                elif dim==1:
                    # chosing TT this moment, avoid TT chosing next time
                    next_avail_act[index] = 1 - action_dim_corr_dict
                    next_avail_act[index, int(AT)] = 0
                    if self.ScenarioConfig.Disable_AT0: next_avail_act[index, 0] = 0
                    if self.ScenarioConfig.Disable_AT1: next_avail_act[index, 1] = 0
                    if self.ScenarioConfig.Disable_AT2: next_avail_act[index, 2] = 0
                    if self.ScenarioConfig.Disable_AT3: next_avail_act[index, 3] = 0
                    if self.ScenarioConfig.Disable_AT4: next_avail_act[index, 4] = 0
                    if self.ScenarioConfig.Disable_AT5: next_avail_act[index, 5] = 0
                    if self.ScenarioConfig.Disable_AT6: next_avail_act[index, 6] = 0
                else:
                    assert False
                print('next_avail_act:', next_avail_act[index])

            # switch case for AT
            if AT==0:   # Do nothing
                # next_avail_act = np.array()
                pass
            elif AT==1: # parse_act_case_track
                cmd_buffer = self.parse_act_case_track(cmd_buffer, p, TT, rad=0)
            elif AT==2: # parse_act_case_reverseTrack
                cmd_buffer = self.parse_act_case_track(cmd_buffer, p, TT, rad=np.pi)
            elif AT==3: # parse_act_case_3clockTrack
                cmd_buffer = self.parse_act_case_track(cmd_buffer, p, TT, rad=-np.pi/2)
            elif AT==4: # parse_act_case_9clockTrack
                cmd_buffer = self.parse_act_case_track(cmd_buffer, p, TT, rad=+np.pi/2)
            elif AT==5:
                cmd_buffer = self.parse_act_case_fire(cmd_buffer, p, TT)
            elif AT==6:
                cmd_buffer = self.parse_act_final_SP(cmd_buffer, p, TT)
            else:
                assert False

            if not self.ScenarioConfig.use_simple_action_space:
                assert False
                # cmd_buffer = self.parse_act_final_HT(cmd_buffer, p, HT)# 暂时没用
                cmd_buffer = self.parse_act_final_SP(cmd_buffer, p, SP)    
            # else:
                # if previous_action_context[index, 2] != HT: cmd_buffer = self.parse_act_final_HT(cmd_buffer, p, HT) # 暂时没用
                # if previous_action_context[index, self.SP_SEL] != SP: cmd_buffer = self.parse_act_final_SP(cmd_buffer, p, SP)

        return cmd_buffer, next_avail_act

    def parse_act_case_fire(self, cmd_buffer, p, TT):
        target = self.tran_target(TT)
        # if (self.player_color=='red' and p.LeftWeapon<=1) or (self.player_color=='blue' and p.OpLeftWeapon<=1):
        #     print('saving ms, do not fire')
        #     return cmd_buffer


        cmd_buffer.append(CmdEnv.make_attackparam(p.ID, target.ID, 1))
        return cmd_buffer


    def parse_act_final_HT(self, cmd_buffer, p, HT):
        p.PlayerHeightSetting = p.MinHeight + (p.MaxHeight - p.MinHeight)*( HT/(self.ScenarioConfig.HeightLevels-1) )
        return cmd_buffer

    def parse_act_final_SP(self, cmd_buffer, p, SP):
        cmd_speed = p.MinSpeed + (p.MaxSpeed - p.MinSpeed)*( SP/(self.ScenarioConfig.n_target_actions-1) ) # , i = [0,1,2,3,4]
        cmd_speed = np.clip(cmd_speed, a_min=p.MinSpeed, a_max=p.MaxSpeed)
        # 指令速度:1 / 指令加速度:2 / 指令速度和加速度:3 / 指令过载:4 / 指令速度和过载:5 / 指令加速度和过载:6 / 指令速度和加速度和过载:7
        cmd_buffer.append(CmdEnv.make_motioncmdparam(p.ID, 1, cmd_speed, p.MaxAcc, p.MaxOverload))
        return cmd_buffer

    def parse_act_case_track(self, cmd_buffer, p, TT, rad):
        target = self.tran_target(TT)
        delta_to_TT =  target.pos2d - p.pos2d # 向量的方向指向目标
        unit_delta = np.matmul(
            delta_to_TT,
            np.array([[np.cos(rad), np.sin(rad)],
                     [-np.sin(rad), np.cos(rad)]])
        )
        
        H2 = unit_delta[:2] + p.pos2d
        goto_location = [{
            "X": H2[0],
            "Y": H2[1],
            "Z": target.Z   # p.PlayerHeightSetting
        }]
        cmd_buffer.append(self.observer.check_and_make_linepatrolparam(
            p.ID,
            goto_location,
            np.clip(p.Speed, a_min=p.MinSpeed, a_max=p.MaxSpeed),
            p.MaxAcc,
            p.MaxOverload
        ))
        return cmd_buffer
