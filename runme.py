import os
import configparser
config=configparser.ConfigParser()
config.read('./config.ini')

# split curated into train data, validation data, and test data
# splited data will be saved in split_dir defined in config.ini
os.system('python3 split_curated_tr.py')





# stage 0
# WARM UP: train model on noisy data set, trained model will be saved in checkpoint_dir/stage0_model
os.system('python3 run_stage0.py')
# # test 0.285-> 0.282
# print('Testing after STAGE 0 ...')
os.system('python3 test.py {}'.format(os.path.join(config.get('DataPath','checkpoint_dir'),config.get('SaveModel','stage0_model'))))




# stage 1
# FINE TUNE: start with stage 0  trained model, train model on curated training data, trained model will be saved in checkpoint_dir/stage1_model
os.system('python3 run_stage1.py')
# test 0.828 -> 0.7915
print('Testing after STAGE 1 ...')
os.system('CUDA_VISIBLE_DEVICES=0 python3 test.py {}'.format(os.path.join(config.get('DataPath','checkpoint_dir'),config.get('SaveModel','stage1_model')))) #0.80 -> 0.77





# based on model trained in stage 1, filter data from noisy data to expand labeled data
os.system('python3 filterout_noisysamples.py')

# stage 2
# using semi-supervised learning train on curated train data and noisy data
os.system('CUDA_VISIBLE_DEVICES=0 python3 run_semisupervised.py')
# test 0.836 -> 0.816
print('Testing after STAGE 2 ...')
os.system('python3 test.py {}'.format(os.path.join(config.get('DataPath','checkpoint_dir'),config.get('SaveModel','stage2_model'))))
