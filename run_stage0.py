import os
import configparser
"""
stage 0: train model on noisy dataset to warm up
trained model will be save in path defined in config ['SaveModel']['stage0_model']
"""
config=configparser.ConfigParser()
config.read('./config.ini')

datapath = config.get('DataPath','stage0_noisy_data_path')
csvpath = config.get('DataPath','stage0_noisy_csv_path')
model_path = 'NAN'
loadModel = 'False'
save_model_filename = config.get('SaveModel','stage0_model')

os.system('CUDA_VISIBLE_DEVICES=0 python3 run_fat2019.py {} {} {} {} {}'.format(datapath,\
                                                                                csvpath,\
                                                                                model_path,\
                                                                                loadModel,\
                                                                                save_model_filename
                                                                                    ))
