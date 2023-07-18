



json_path = 'RESULT/random-formation-r1/raw_exp.jsonc'
ALG_NAME = "ALGORITHM.ppo_ma.foundation.py->AlgorithmConfig"



def validate_path():
    import os, sys
    dir_name = os.path.dirname(__file__)
    root_dir_assume = os.path.abspath(os.path.dirname(__file__) +  '/..')
    os.chdir(root_dir_assume)
    sys.path.append(root_dir_assume)

validate_path() # validate path so you can run from base directory
import commentjson as json
import shutil
import os
pj = os.path.join
bn = os.path.basename
dn = os.path.dirname
absp = os.path.abspath

with open(json_path, 'r', encoding='utf8') as f:
    json_struct = f.read()

json_struct = json.loads(json_struct)

note = json_struct["config.py->GlobalConfig"]["note"]
if "config.py->GlobalConfig" in json_struct:
    # 主框架参数修改
    json_struct["config.py->GlobalConfig"]["test_only"] = True
    json_struct["config.py->GlobalConfig"]["num_threads"] = 1
    json_struct["config.py->GlobalConfig"]["report_reward_interval"] = 1
    json_struct["config.py->GlobalConfig"]["test_interval"] = 1e10
    json_struct["config.py->GlobalConfig"]["fold"] = 1
    json_struct["config.py->GlobalConfig"]["backup_files"] = []
    json_struct["config.py->GlobalConfig"]["device"] = 'cuda'
    json_struct["config.py->GlobalConfig"]["note"] = json_struct["config.py->GlobalConfig"]["note"] + '-eval'

if "MISSION.uhmap.uhmap_env_wrapper.py->ScenarioConfig" in json_struct:
    # 虚幻umap的参数修改
    json_struct["MISSION.uhmap.uhmap_env_wrapper.py->ScenarioConfig"]["UElink2editor"] = False
    json_struct["MISSION.uhmap.uhmap_env_wrapper.py->ScenarioConfig"]["TimeDilation"] = 2

if ALG_NAME in json_struct:
    json_struct[ALG_NAME]["load_checkpoint"] = True

with open(pj(dn(json_path), 'auto_eval.jsonc'), 'w', encoding='utf8') as f:
    json.dump(json_struct, f, indent=4, ensure_ascii=False)

try: shutil.rmtree(absp(pj(dn(json_path), '..', note + '-eval')))
except: pass

shutil.copytree(src = absp(pj(dn(json_path), '..', note)), 
                dst = absp(pj(dn(json_path), '..', note + '-eval')))

fp = absp(pj(dn(json_path), '..', note + '-eval'))
json_fp = pj(fp, 'auto_eval.jsonc')
print(f'writing to {fp}')
print(f'run command: python main.py -c {json_fp}')


