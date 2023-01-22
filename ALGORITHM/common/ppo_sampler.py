import torch, math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from random import randint, sample
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from UTIL.colorful import *
from UTIL.tensor_ops import _2tensor, __hash__, repeat_at, _2cpu2numpy
from UTIL.tensor_ops import my_view, scatter_with_nan, sample_balance
from config import GlobalConfig as cfg
from UTIL.gpu_share import GpuShareUnit

class TrajPoolSampler():
    def __init__(self, 
            n_div, 
            traj_pool, 
            flag, 
            req_dict, 
            return_rename,
            value_rename,
            advantage_rename,
            prevent_batchsize_oom=False, 
            advantage_norm=True,
            exclude_eprsn_in_norm=False,
            mcv=None):
        self.n_pieces_batch_division = n_div
        self.prevent_batchsize_oom = prevent_batchsize_oom    
        self.mcv = mcv
        self.advantage_norm = advantage_norm
        if self.prevent_batchsize_oom:
            assert self.n_pieces_batch_division==1, 'self.n_pieces_batch_division should be 1'

        self.num_batch = None
        self.container = {}
        self.warned = False
        assert flag=='train'
        req_dict_rename = req_dict
        if cfg.ScenarioConfig.AvailActProvided:
            req_dict.append('avail_act')
            req_dict_rename.append('avail_act')
        # replace 'obs' to 'obs > xxxx'
        for key_index, key in enumerate(req_dict):
            key_name =  req_dict[key_index]
            key_rename = req_dict_rename[key_index]
            if not hasattr(traj_pool[0], key_name):
                real_key_list = [real_key for real_key in traj_pool[0].__dict__ if (key_name+'>' in real_key)]
                assert len(real_key_list) > 0, ('check variable provided!', key,key_index)
                for real_key in real_key_list:
                    mainkey, subkey = real_key.split('>')
                    req_dict.append(real_key)
                    req_dict_rename.append(key_rename+'>'+subkey)
        self.big_batch_size = -1  # vector should have same length, check it!
        
        # load traj into a 'container'
        for key_index, key in enumerate(req_dict):
            key_name =  req_dict[key_index]
            key_rename = req_dict_rename[key_index]
            if not hasattr(traj_pool[0], key_name): continue
            set_item = np.concatenate([getattr(traj, key_name) for traj in traj_pool], axis=0)
            if not (self.big_batch_size==set_item.shape[0] or (self.big_batch_size<0)):
                print('error')
            assert self.big_batch_size==set_item.shape[0] or (self.big_batch_size<0), (key,key_index)
            self.big_batch_size = set_item.shape[0]
            self.container[key_rename] = set_item    # assign value to key_rename

        # normalize advantage inside the batch
        self.container[advantage_rename] = self.container[return_rename] - self.container[value_rename]
        if self.advantage_norm:
            if exclude_eprsn_in_norm:
                assert 'eprsn' in self.container
                no_pr = ~self.container['eprsn'].all(-1)
                m = self.container[advantage_rename][no_pr].mean()
                s = self.container[advantage_rename][no_pr].std()
            else:
                m = self.container[advantage_rename].mean()
                s = self.container[advantage_rename].std()

            self.container[advantage_rename] = ( self.container[advantage_rename] - m ) / (s + 1e-5)
        # size of minibatch for each agent
        self.mini_batch_size = math.ceil(self.big_batch_size / self.n_pieces_batch_division)  

        # do once
        self.do_once_fin = False

    def __len__(self):
        return self.n_pieces_batch_division

    def reminder(self, n_sample):
        if not self.do_once_fin:
            self.do_once_fin = True
            drop_percent = (self.big_batch_size-n_sample) / self.big_batch_size*100
            if self.mcv is not None: self.mcv.rec(drop_percent, 'drop percent')
            if drop_percent > 20: 
                print_ = print亮红
                print_('droping %.1f percent samples..'%(drop_percent))
                assert False, "GPU OOM!"
            else:
                print_ = print
                print_('droping %.1f percent samples..'%(drop_percent))

    def get_sampler(self):
        if not self.prevent_batchsize_oom:
            sampler = BatchSampler(SubsetRandomSampler(range(self.big_batch_size)), self.mini_batch_size, drop_last=False)
        else:
            max_n_sample = self.determine_max_n_sample()
            n_sample = min(self.big_batch_size, max_n_sample)
            self.reminder(n_sample)
            sampler = BatchSampler(SubsetRandomSampler(range(n_sample)), n_sample, drop_last=False)
        return sampler

    def reset_and_get_iter(self):
        self.sampler = self.get_sampler()
        for indices in self.sampler:
            selected = {}
            for key in self.container:
                selected[key] = self.container[key][indices]
            for key in [key for key in selected if '>' in key]:
                # re-combine child key with its parent
                mainkey, subkey = key.split('>')
                if not mainkey in selected: selected[mainkey] = {}
                selected[mainkey][subkey] = selected[key]
                del selected[key]
            yield selected


    def determine_max_n_sample(self):
        assert self.prevent_batchsize_oom
        if not hasattr(TrajPoolSampler,'MaxSampleNum'):
            # initialization
            TrajPoolSampler.MaxSampleNum =  [int(self.big_batch_size*(i+1)/50) for i in range(50)]
            max_n_sample = self.big_batch_size
        elif TrajPoolSampler.MaxSampleNum[-1] > 0:  
            # meaning that oom never happen, at least not yet
            # only update when the batch size increases
            if self.big_batch_size > TrajPoolSampler.MaxSampleNum[-1]:
                TrajPoolSampler.MaxSampleNum.append(self.big_batch_size)
            max_n_sample = self.big_batch_size
        else:
            # meaning that oom already happened, choose TrajPoolSampler.MaxSampleNum[-2] to be the limit
            assert TrajPoolSampler.MaxSampleNum[-2] > 0
            max_n_sample = TrajPoolSampler.MaxSampleNum[-2]
        return max_n_sample
