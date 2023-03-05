import numpy as np
import skfuzzy as fuzz
import itertools
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
def projection(x, from_range, to_range):
    return ((x - from_range[0]) * (to_range[1] - to_range[0]) / (from_range[1] - from_range[0])) + to_range[0]

def gen_antecedent(key, min, max):
    d = (max - min) / 1000
    antecedent = ctrl.Antecedent(np.arange(min, max+1e-10, d), key) # Antecedent 前提变量 [0 ~ 1]
    names = ['small', 'medium', 'large']
    antecedent.automf(names=names)
    # for mfn in  ['very small', 'small', 'medium', 'large', 'very large']:
    #     plt.plot(antecedent.universe, antecedent[mfn].mf, linewidth=1.5, label=mfn)
    # plt.title(f'Membership functions of {key}')
    # plt.legend()
    # plt.show()
    return antecedent

def gen_consequent(key, min, max, defuzzify_method='centroid', consequent_num_mf=5):
    d = (max - min) / 1000
    consequent = ctrl.Consequent(np.arange(min, max+1e-10, d), key, defuzzify_method=defuzzify_method) # consequent 前提变量 [0 ~ 1]
    if consequent_num_mf==5:
        names = ['very small', 'small', 'medium', 'large', 'very large']
        consequent.automf(names=names)
    elif consequent_num_mf==7:
        names = ['extreme small', 'very small', 'small', 'medium', 'large', 'very large', 'extreme large']
        consequent.automf(names=names)
    else:
        assert False

    # for mfn in  ['very small', 'small', 'medium', 'large', 'very large']:
    #     plt.plot(consequent.universe, consequent[mfn].mf, linewidth=1.5, label=mfn)
    # plt.title(f'Membership functions of {key}')
    # plt.legend()
    # plt.show()
    return consequent
    
# input_arr = [consequent_1_select, consequent_2_select, consequent_3_select]
def gen_rule_list(input_arr, antecedent, consequent_arr):
    antecedent_membership = [t for t in antecedent.terms]
    consequent_membership = [t for t in consequent_arr[0].terms]    # assume all consequent has same memberships
    # try:
    assert len(antecedent_membership) * len(consequent_arr) == len(input_arr)
    # except:
        # pass
    rule_list = []
    p = 0
    for consequent in consequent_arr:
        for k in antecedent_membership:
            rule_list.append(ctrl.Rule(antecedent[k],  consequent[consequent_membership[input_arr[p]]])); p += 1
    assert p == len(input_arr)
    return rule_list


def gen_feedback_sys_generic(
        antecedent_list,
        consequent_key,
        consequent_min,
        consequent_max,
        consequent_num_mf,
        fuzzy_controller_param, 
        defuzzify_method='centroid',
        compute_fn=None
    ):
    antecedent_key, antecedent_min, antecedent_max = antecedent_list[0]

    # input [-1.0, +1.0] ---> fuzzy
    fuzzy_antecedent = gen_antecedent(key=antecedent_key, min=antecedent_min, max=antecedent_max)
    
    # defuzzy --> [-1.5, +1.5] --> [-31.6, 0.0316]
    fuzzy_consequent = gen_consequent(key=consequent_key, min=consequent_min, max=consequent_max, defuzzify_method=defuzzify_method, consequent_num_mf=consequent_num_mf)

    rule_list = gen_rule_list(
        input_arr=fuzzy_controller_param, 
        antecedent=fuzzy_antecedent, 
        consequent_arr=[fuzzy_consequent], 
    )

    controller = ctrl.ControlSystem(rule_list)
    feedback_sys = ctrl.ControlSystemSimulation(controller)
    feedback_sys.compute_fn = lambda x: compute_fn(feedback_sys, x)

    feedback_sys.register_input = fuzzy_antecedent
    feedback_sys.register_output = fuzzy_consequent

    return feedback_sys

def int2consq(a, fuzzy_consequent):
    q = ['extreme small', 'very small', 'small', 'medium', 'large', 'very large', 'extreme large']
    return fuzzy_consequent[q[a]]

def gen_feedback_sys_generic_multi_input(
        antecedent_list,
        consequent_key,
        consequent_min,
        consequent_max,
        consequent_num_mf,
        fuzzy_controller_param, 
        defuzzify_method='centroid',
        compute_fn=None
    ):
    fuzzy_antecedents = {}
    for antecedent_key, antecedent_min, antecedent_max in antecedent_list:
        fuzzy_antecedent = gen_antecedent(key=antecedent_key, min=antecedent_min, max=antecedent_max)
        fuzzy_antecedents[antecedent_key] = fuzzy_antecedent

 
    # defuzzy --> [-1.5, +1.5] --> [-31.6, 0.0316]
    fuzzy_consequent = gen_consequent(key=consequent_key, min=consequent_min, max=consequent_max, defuzzify_method=defuzzify_method, consequent_num_mf=consequent_num_mf)
    
    # consequent_key='intrisic_reward',
    rule_list = []
    # names = ['extreme small', 'very small', 'small', 'medium', 'large', 'very large', 'extreme large']
    # Check to make sure this set of input values will activate at least one connected Term in each Antecedent via the current set of Rules.
    rule_list.append(ctrl.Rule(
        fuzzy_antecedents['lifelen_norm']['small'] & fuzzy_antecedents['recent_winrate']['small'] ,  int2consq(fuzzy_controller_param[0], fuzzy_consequent)
    ))
    rule_list.append(ctrl.Rule(
        fuzzy_antecedents['lifelen_norm']['medium'] & fuzzy_antecedents['recent_winrate']['small'],  int2consq(fuzzy_controller_param[1], fuzzy_consequent)
    ))
    rule_list.append(ctrl.Rule(
        fuzzy_antecedents['lifelen_norm']['large'] & fuzzy_antecedents['recent_winrate']['small'] ,  int2consq(fuzzy_controller_param[2], fuzzy_consequent)
    ))

    rule_list.append(ctrl.Rule(
        fuzzy_antecedents['lifelen_norm']['small'] & fuzzy_antecedents['recent_winrate']['medium'] ,  int2consq(fuzzy_controller_param[3], fuzzy_consequent)
    ))
    rule_list.append(ctrl.Rule(
        fuzzy_antecedents['lifelen_norm']['medium'] & fuzzy_antecedents['recent_winrate']['medium'],  int2consq(fuzzy_controller_param[4], fuzzy_consequent)
    ))
    rule_list.append(ctrl.Rule(
        fuzzy_antecedents['lifelen_norm']['large'] & fuzzy_antecedents['recent_winrate']['medium'] ,  int2consq(fuzzy_controller_param[5], fuzzy_consequent)
    ))

    rule_list.append(ctrl.Rule(
        fuzzy_antecedents['lifelen_norm']['small'] & fuzzy_antecedents['recent_winrate']['large'] ,  int2consq(fuzzy_controller_param[6], fuzzy_consequent)
    ))
    rule_list.append(ctrl.Rule(
        fuzzy_antecedents['lifelen_norm']['medium'] & fuzzy_antecedents['recent_winrate']['large'],  int2consq(fuzzy_controller_param[7], fuzzy_consequent)
    ))
    rule_list.append(ctrl.Rule(
        fuzzy_antecedents['lifelen_norm']['large'] & fuzzy_antecedents['recent_winrate']['large'] ,  int2consq(fuzzy_controller_param[8], fuzzy_consequent)
    ))

    controller = ctrl.ControlSystem(rule_list)
    feedback_sys = ctrl.ControlSystemSimulation(controller)
    feedback_sys.compute_fn = lambda x: compute_fn(feedback_sys, x)

    feedback_sys.register_input = fuzzy_antecedent
    feedback_sys.register_output = fuzzy_consequent

    return feedback_sys



def gen_feedback_sys_generic_multi_input(
        antecedent_list,
        consequent_key,
        consequent_min,
        consequent_max,
        consequent_num_mf,
        fuzzy_controller_param, 
        defuzzify_method='centroid',
        compute_fn=None
    ):
    fuzzy_antecedents = {}
    for antecedent_key, antecedent_min, antecedent_max in antecedent_list:
        fuzzy_antecedent = gen_antecedent(key=antecedent_key, min=antecedent_min, max=antecedent_max)
        fuzzy_antecedents[antecedent_key] = fuzzy_antecedent

 
    # defuzzy --> [-1.5, +1.5] --> [-31.6, 0.0316]
    fuzzy_consequent = gen_consequent(key=consequent_key, min=consequent_min, max=consequent_max, defuzzify_method=defuzzify_method, consequent_num_mf=consequent_num_mf)
    
    # consequent_key='intrisic_reward',
    rule_list = []
    # names = ['extreme small', 'very small', 'small', 'medium', 'large', 'very large', 'extreme large']
    # Check to make sure this set of input values will activate at least one connected Term in each Antecedent via the current set of Rules.

    for combo in itertools.combinations(fuzzy_antecedents.items(), 2):
        raise NotImplementedError
        rule_list.append(ctrl.Rule(
            fuzzy_antecedents['lifelen_norm']['small'] & fuzzy_antecedents['recent_winrate']['small'] ,  int2consq(fuzzy_controller_param[0], fuzzy_consequent)
        ))

    controller = ctrl.ControlSystem(rule_list)
    feedback_sys = ctrl.ControlSystemSimulation(controller)
    feedback_sys.compute_fn = lambda x: compute_fn(feedback_sys, x)

    feedback_sys.register_input = fuzzy_antecedent
    feedback_sys.register_output = fuzzy_consequent

    return feedback_sys



