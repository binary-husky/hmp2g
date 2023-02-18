from UTIL.batch_exp import fetch_experiment_conclusion


res = fetch_experiment_conclusion(6, 
    [
        {
            'checkpoint': '/home/hmp/MultiServerMission/2023-02-17-15-35-51-Bo_AutoRL/src/ZHECKPOINT/parallel-0',
            'conclusion': '/home/hmp/MultiServerMission/2023-02-17-15-35-51-Bo_AutoRL/src/ZHECKPOINT/parallel-0/experiment_conclusion.pkl',
            'mark': '2023-02-17-15-35-51-Bo_AutoRL',
        },
        {
            'checkpoint': '/home/hmp/MultiServerMission/2023-02-17-15-35-51-Bo_AutoRL/src/ZHECKPOINT/parallel-1',
            'conclusion': '/home/hmp/MultiServerMission/2023-02-17-15-35-51-Bo_AutoRL/src/ZHECKPOINT/parallel-1/experiment_conclusion.pkl',
            'mark': '2023-02-17-15-35-51-Bo_AutoRL',
        },
    ], 
    [
        {
            "addr": "172.18.116.149:2266",
            "usr": "hmp",
            "pwd": "hmp"
        },
        {
            "addr": "172.18.116.149:2266",
            "usr": "hmp",
            "pwd": "hmp"
        },
    ]
)

def normalize_score(conclusion_list):
    score_list = []
    for c in conclusion_list:
        conclusion_parsed = {}
        # parse
        for name, line, time in zip(c['name_list'],c['line_list'],c['time_list']):
            conclusion_parsed[name] = line
        s = conclusion_parsed['acc win ratio of=team-0']
        score_list.append(s[-1])
    return score_list
print(res)