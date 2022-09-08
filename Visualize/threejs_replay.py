import os, sys
import argparse
from Visualize.mcom import *
from Visualize.mcom_replay import RecallProcessThreejs
from Util.network import find_free_port


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HMP')
    parser.add_argument('-f', '--file', help='Directory of chosen file', default='temp/v2d_logger/backup.dp.gz')
    parser.add_argument('-p', '--port', help='The port for web server')
    args, unknown = parser.parse_known_args()
    if hasattr(args, 'file'):
        path = args.file
    else:
        assert False, (r"parser.add_argument('-f', '--file', help='The node name is?')")

    if hasattr(args, 'port') and args.port is not None:
        port = int(args.port)
    else:
        port = find_free_port()
        print('no --port arg, auto find:', port)

    load_via_json = (hasattr(args, 'cfg') and args.cfg is not None)
    
    rp = RecallProcessThreejs(path, port)
    rp.start()
    rp.join()


'''

note=RVE-drone1-fixaa-run2
cp -r ./checkpoint/$note ./checkpoint/$note-bk
cp -r ./checkpoint/$note/experiment.jsonc ./checkpoint/$note/experiment-bk.jsonc
cp -r ./checkpoint/$note/experiment.jsonc ./checkpoint/$note/train.jsonc
cp -r ./checkpoint/$note/experiment.jsonc ./checkpoint/$note/test.jsonc

python << __EOF__
import commentjson as json
file = "./checkpoint/$note/test.jsonc"
print(file)
with open(file, encoding='utf8') as f:
    json_data = json.load(f)
json_data["config.py->GlobalConfig"]["num_threads"] = 1
json_data["config.py->GlobalConfig"]["fold"] = 1
json_data["config.py->GlobalConfig"]["test_only"] = True
json_data["Mission.uhmap.uhmap_env_wrapper.py->ScenarioConfig"]["TimeDilation"] = 1
json_data["Algorithm.conc_4hist_hete.foundation.py->AlgorithmConfig"]["load_checkpoint"] = True
with open(file, 'w') as f:
    json.dump(json_data, f, indent=4)
__EOF__

python main.py -c ./checkpoint/$note/test.jsonc




'''