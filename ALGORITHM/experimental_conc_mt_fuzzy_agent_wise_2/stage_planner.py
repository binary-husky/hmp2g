import math
import numpy as np
from .foundation import AlgorithmConfig
from UTIL.colorful import *
from UTIL.tensor_ops import repeat_at
class PolicyRsnConfig:
    method = 'legacy' # 'legacy'
    resonance_start_at_update = 0
    yita_min_prob = 0.15  #  should be >= (1/n_action)
    yita_max = 0.25
    yita_inc_per_update = 1 # (increase to 0.75 in 500 updates)
    
    yita_shift_method = 'slow-inc'
    yita_shift_cycle = 1000

    lockPrInOneBatch = False

class StagePlanner:
    def __init__(self, n_agent, mcv) -> None:
        self.method = AlgorithmConfig.policy_resonance_method
        if self.method == 'legacy':
            self.yita = 0
        else:
            self.yita = None

        if PolicyRsnConfig.resonance_start_at_update == 0:
            self.resonance_active = True
        else:
            self.resonance_active = False

        if PolicyRsnConfig.yita_shift_method == 'feedback':
            from .scheduler import FeedBackPolicyResonance
            self.feedback_controller = FeedBackPolicyResonance(mcv)
        else:
            self.feedback_controller = None

        self.update_cnt = 0
        self.mcv = mcv
        self.trainer = None
        self.n_agent = n_agent

        if self.method == 'level':
            mapLevel2PrNumLs = np.floor(np.arange(AlgorithmConfig.distribution_precision) / AlgorithmConfig.distribution_precision * n_agent)
            self.mapLevel2PrNumLs = mapLevel2PrNumLs.astype(np.int64)
            self.mapPrNum2LevelLs = {j:i for i,j in enumerate(self.mapLevel2PrNumLs)}

    def mapLevel2PrNum(self, i):
        """
            if self.method == 'level'
        """
        return self.mapLevel2PrNumLs[i]

    def mapPrNum2Level(self, j):
        """
            if self.method == 'level'
        """
        assert j in self.mapPrNum2LevelLs, f'j={j},self.mapPrNum2LevelLs={self.mapPrNum2LevelLs}'
        return self.mapPrNum2LevelLs[j]

    def n_pr_distribution(self, n_thread, n_agent):
        """
            if self.method == 'level'
        """
        randl = np.random.choice(AlgorithmConfig.target_distribute, n_thread, replace=True)
        npr = np.array(list(map(self.mapLevel2PrNum, randl)))
        return npr, randl

    def uprate_eprsn(self, n_thread):
        if self.method == 'level':
            # eprsn_yita = self.yita
            n_pr_agent, randl = self.n_pr_distribution(n_thread, self.n_agent) # np.random.rand(n_thread) < eprsn_yita
            self.eprsn = []
            if PolicyRsnConfig.lockPrInOneBatch:
                n_pr = n_pr_agent[0]
                res = self._generate_random_n_hot_vector(vlength=self.n_agent, n=int(n_pr))
                for n_pr in n_pr_agent: 
                    self.eprsn.append(res)
            else:
                for n_pr in n_pr_agent: 
                    self.eprsn.append(self._generate_random_n_hot_vector(vlength=self.n_agent, n=int(n_pr)))
            self.eprsn = np.stack(self.eprsn)
            self.randl = randl
            return self.eprsn, self.randl
        else:
            # policy resonance
            assert self.yita is not None
            eprsn_yita = self.yita if AlgorithmConfig.use_policy_resonance else 0
            self.eprsn = (np.random.rand(n_thread) < eprsn_yita)
            
            self.eprsn = repeat_at(self.eprsn, insert_dim=-1, n_times=self.n_agent)
            self.randl = np.array([-1]*n_thread)
            return self.eprsn, self.randl


    def is_resonance_active(self,):
        return self.resonance_active
    
    def is_body_freeze(self,):
        raise RuntimeError
    
    def get_yita(self):
        return self.yita
    
    def get_yita_min_prob(self):
        return PolicyRsnConfig.yita_min_prob
    
    def can_exec_trainning(self):
        return True

    def update_plan(self):
        self.update_cnt += 1
        if AlgorithmConfig.use_policy_resonance:
            if self.resonance_active:
                self.when_pr_active()
            elif not self.resonance_active:
                self.when_pr_inactive()
        return

    def update_test_winrate(self, win_rate):
        if self.feedback_controller is not None:
            self.feedback_controller.step(win_rate)
    
    def activate_pr(self):
        self.resonance_active = True

    def when_pr_inactive(self):
        assert not self.resonance_active
        if PolicyRsnConfig.resonance_start_at_update >= 0:
            # mean need to activate pr later
            if self.update_cnt > PolicyRsnConfig.resonance_start_at_update:
                # time is up, activate pr
                self.activate_pr()
        # log
        pr = 1 if self.resonance_active else 0
        self.mcv.rec(pr, 'resonance')
        self.mcv.rec(self.yita, 'yita')

    def when_pr_active(self):
        assert self.resonance_active
        self._update_yita()
        # log
        pr = 1 if self.resonance_active else 0
        self.mcv.rec(pr, 'resonance')
        self.mcv.rec(self.yita, 'yita')

    def _update_yita(self):
        '''
            increase self.yita by @yita_inc_per_update per function call
        '''
        if PolicyRsnConfig.yita_shift_method == '-cos':
            self.yita = PolicyRsnConfig.yita_max
            t = -math.cos(2*math.pi/PolicyRsnConfig.yita_shift_cycle * self.update_cnt) * PolicyRsnConfig.yita_max
            if t<=0: self.yita = 0
            else: self.yita = t
            print亮绿('yita update:', self.yita)

        elif PolicyRsnConfig.yita_shift_method == '-sin':
            self.yita = PolicyRsnConfig.yita_max
            t = -math.sin(2*math.pi/PolicyRsnConfig.yita_shift_cycle * self.update_cnt) * PolicyRsnConfig.yita_max
            if t<=0: self.yita = 0
            else: self.yita = t
            print亮绿('yita update:', self.yita)

        elif PolicyRsnConfig.yita_shift_method == 'slow-inc':
            self.yita += PolicyRsnConfig.yita_inc_per_update
            if self.yita > PolicyRsnConfig.yita_max:
                self.yita = PolicyRsnConfig.yita_max
            print亮绿('yita update:', self.yita)

        elif PolicyRsnConfig.yita_shift_method == 'feedback':
            self.yita = self.feedback_controller.recommanded_yita
            print亮绿('yita update:', self.yita)
            
        else:
            assert False


    def _generate_random_n_hot_vector(self, vlength, n):
        pick = np.random.choice(vlength, n, replace=False)
        tmp = np.zeros(vlength, dtype=np.bool)
        tmp[pick] = True
        return tmp