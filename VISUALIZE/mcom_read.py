def validate_path():
    import os, sys
    dir_name = os.path.dirname(__file__)
    root_dir_assume = os.path.abspath(os.path.dirname(__file__) +  '/..')
    os.chdir(root_dir_assume)
    sys.path.append(root_dir_assume)
    
validate_path() # validate path so you can run from base directory


import os
import socket
import time
import traceback
import numpy as np
from colorama import init
from multiprocessing import Process, Pipe
init()
def get_files_to_read(base_path):
    starting_file_index = -1
    ending_file_index = -1
    pointer = 0
    while True:
        es = os.path.exists(base_path+'mcom_buffer_%d____starting_session.txt'%pointer)
        ee = os.path.exists(base_path+'mcom_buffer_%d.txt'%pointer)
        if (not es) and (not ee): break
        assert not (ee and es), ('?')
        if es: starting_file_index = pointer; ending_file_index = pointer
        if ee: ending_file_index = pointer
        pointer += 1
        assert pointer < 1e3
    assert starting_file_index>=0 and ending_file_index>=0, ('查找日志失败:', base_path)

    file_path = []
    for i in range(starting_file_index, ending_file_index+1):
        if i==starting_file_index: file_path.append(base_path+'mcom_buffer_%d____starting_session.txt'%i)
        else: file_path.append(base_path+'mcom_buffer_%d.txt'%i)
        assert os.path.exists(file_path[0]), ('?')
    return file_path

def read_experiment(base_path):
    files_to_read = get_files_to_read(base_path)
    cmd_lines = []
    for file in files_to_read:
        f = open(file, 'r')
        lines = f.readlines()
        cmd_lines.extend(lines)
    dictionary = {}

    def rec(value,name): 
        if name not in dictionary:
            dictionary[name] = []
        dictionary[name].append(value)
        return

    for cmd_str in cmd_lines:
        if '>>' in cmd_str:
            cmd_str_ = cmd_str[2:].strip('\n')
            if not cmd_str_.startswith('rec('): continue
            eval('%s'%cmd_str_)
    return dictionary

def stack_cutlong(arr_list, min_len=None):
    if min_len is None:
        min_len = min([len(item) for item in arr_list])
    print([len(item) for item in arr_list],'\tselect:', min_len)
    return np.stack([item[:min_len] for item in arr_list])


def smooth(data, sm=1):
    if sm > 1:
        y = np.ones(sm)*1.0/sm
        d = np.convolve(y, data, 'valid')#"same")
    else:
        d = data
    return np.array(d)


def tsplot(ax, data, label, resize_x, smooth_sm=None, **kw):
    if smooth_sm is not None:
        print('警告 smooth_sm=',smooth_sm)
        data = smooth(data, smooth_sm)

    print('警告 resize_x=',resize_x)
    x = np.arange(data.shape[1])
    x = resize_x*x
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd, est + sd)
    ax.fill_between(x,cis[0],cis[1],alpha=0.4, **kw)
    ax.plot(x,est, linewidth=1.5, label=label, **kw)
    ax.margins(x=0)

party = [
    {
        "Method": "AddBn",
        "path": [
            "RESULT/Run1_wr_reward_T0addhist",
        ]
    },

]
for ex in party:
    for i, path in enumerate(ex['path']):
        ex['path'][i] = ex['path'][i] + '/logger/'
samples = []

for ex in party:
    pathes = ex['path']
    for path in pathes:
        ex['readings of %s'%path] = read_experiment(path)
        print('readings of %s'%path)



for ex in party:
    for path in ex['path']:
        t0_ratio = ex['readings of %s'%path][' top-rank ratio of=team-0']
        t0_ratio = np.array(t0_ratio)
        acc = np.zeros_like(t0_ratio)
        for i in range(len(t0_ratio)):
            acc[i] = t0_ratio[:i].mean()

print(party)
