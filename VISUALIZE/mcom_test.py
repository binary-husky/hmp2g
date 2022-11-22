# ZHECKPOINT/mt_init_run2/logger/mcom_buffer_29____starting_session.txt
def validate_path():
    import os, sys
    dir_name = os.path.dirname(__file__)
    root_dir_assume = os.path.abspath(os.path.dirname(__file__) +  '/..')
    os.chdir(root_dir_assume)
    sys.path.append(root_dir_assume)
    
validate_path()

from VISUALIZE.mcom import mcom

mcv = mcom(
    path='./TEMP',
    draw_mode='Img',
    resume_mod=True,
    # figsize=(48,12),
    # resume_file='ZHECKPOINT/RVE-drone2-ppoma-run1/logger/mcom_buffer_0____starting_session.txt',
    resume_file='mcom_buffer_29____starting_session.txt',
    # resume_file='x.txt',
    image_path='./temp.jpg',
    # rec_exclude=["r*", "n*", 
    #     "*0.00*", 
    #     "*0.01*", 
    #     "*0.04*", 
    #     "*0.06*", 
    #     "*0.11*", 
    #     "*0.18*", 
    #     "*0.25*", 
    # ],
)

input('wait complete')

