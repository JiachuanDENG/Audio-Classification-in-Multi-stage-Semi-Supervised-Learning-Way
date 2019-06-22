import os
import configparser
"""
stage 1: use stage 0 model to start with,
train model on curated training data,
trained model will be saved in path defined in config ['SaveModel']['stage1_model']
"""
config=configparser.ConfigParser()
config.read('./config.ini')

datapath = os.path.join(config.get('DataPath','split_dir'),'mels_train_curated_split.pkl')
csvpath = os.path.join(config.get('DataPath','split_dir'),'train_curated_split.csv')

model_path = os.path.join(config.get('DataPath','checkpoint_dir'),config.get('SaveModel','stage0_model'))
loadModel = 'True'
save_model_filename = config.get('SaveModel','stage1_model')

os.system('CUDA_VISIBLE_DEVICES=3 python3 run_fat2019.py {} {} {} {} {}'.format(datapath,\
                                                                                csvpath,\
                                                                                model_path,\
                                                                                loadModel,\
                                                                                save_model_filename
                                                                                    ))
