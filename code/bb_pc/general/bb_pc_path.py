from os.path import isdir, realpath, dirname, split

BASE_DIR = split(split(split(dirname(realpath(__file__)))[0])[0])[0]

bb_pc_path = {
    'horse_points_file': BASE_DIR + '/demo_data/horse.ply',
    'reg_res_dir': BASE_DIR + '/data/registration_results/',
    'pickle_dir': BASE_DIR + '/data/pickles/',
}

