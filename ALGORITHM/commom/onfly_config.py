import os, time, torch, traceback, shutil
from config import GlobalConfig
from UTIL.colorful import *

class ConfigOnFly():
    def _create_config_fly(self):
        logdir = GlobalConfig.logdir
        self.input_file_dir = '%s/cmd_io.txt' % logdir
        if not os.path.exists(self.input_file_dir):
            with open(self.input_file_dir, 'w+', encoding='utf8') as f: f.writelines(["# Write cmd at next line: ", ""])

    def _config_on_fly(self):
        if not os.path.exists(self.input_file_dir): return

        with open(self.input_file_dir, 'r', encoding='utf8') as f:
            cmdlines = f.readlines()

        cmdlines_writeback = []
        any_change = False

        for cmdline in cmdlines:
            if cmdline.startswith('#') or cmdline=="\n" or cmdline==" \n":
                cmdlines_writeback.append(cmdline)
            else:
                any_change = True
                try:
                    print亮绿('[foundation.py] ------- executing: %s ------'%cmdline)
                    exec(cmdline)
                    cmdlines_writeback.append('# [execute successfully]\t'+cmdline)
                except:
                    print红(traceback.format_exc())
                    cmdlines_writeback.append('# [execute failed]\t'+cmdline)

        if any_change:
            with open(self.input_file_dir, 'w+', encoding='utf8') as f:
                f.writelines(cmdlines_writeback)
