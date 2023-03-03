import numpy as np
import timeit, time
def clean_profile_folder():
    import shutil, os, glob, datetime
    if os.path.exists('PROFILE'):
        res = glob.glob('PROFILE/*')
        for exp in res:
            input_str = exp # "PROFILE/2023-03-03-09-56-59-Bo_AutoRL"
            # Extract the time substring
            time_str = input_str.split("/")[-1][:19]
            # Parse the time substring using datetime.strptime()
            time_then = datetime.datetime.strptime(time_str, "%Y-%m-%d-%H-%M-%S").timestamp()
            time_now = time.time()
            dt_hour = (time_now - time_then)/3600
            if dt_hour > 2:
                shutil.copytree(exp, f'TEMP/{time_str}')
                shutil.rmtree(exp)


clean_profile_folder()