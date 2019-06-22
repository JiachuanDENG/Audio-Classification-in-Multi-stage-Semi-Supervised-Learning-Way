import pandas as pd
import numpy as np
import pickle as pkl
import os
import copy
import configparser

config=configparser.ConfigParser()
config.read('./config.ini')

"""
since our model need to be trained in multiple steps, we need to split out validation data
before hand, so that we can have fair evaluation when training model
"""
np.random.seed(1994)
def main():
    save_dir = config.get('DataPath','split_dir')
    if not  os.path.exists(save_dir):
        os.system('mkdir {}'.format(save_dir))
    validation_ratio = 0.1
    test_ratio = 0.1
    curated_data_path = config.get('DataPath','curated_data_path')
    curated_csv_path = config.get('DataPath','curated_csv_path')
    with open(curated_data_path, 'rb') as f:
        curated_data= pkl.load(f)
    curated_csv_df = pd.read_csv(curated_csv_path)

    assert len(curated_data) == curated_csv_df.shape[0],'labels len not match data len'
    idx = np.random.permutation(len(curated_data))
    val_idx = copy.deepcopy(idx[:int(len(curated_data)*validation_ratio)])
    test_idx = copy.deepcopy(idx[int(len(curated_data)*validation_ratio):int(len(curated_data)*(validation_ratio+test_ratio))])
    train_idx = copy.deepcopy(idx[int(len(curated_data)*(validation_ratio+test_ratio)):])

    train_idx.sort()
    val_idx.sort()
    test_idx.sort()

    tr_data = [curated_data[int(i)] for  i in  train_idx]
    val_data = [curated_data[int(i)]for i in val_idx]
    test_data = [curated_data[int(i)] for i in test_idx]

    tr_csv_df = curated_csv_df[curated_csv_df.index.isin(train_idx)].reset_index()
    val_csv_df = curated_csv_df[curated_csv_df.index.isin(val_idx)].reset_index()
    test_csv_df = curated_csv_df[curated_csv_df.index.isin(test_idx)].reset_index()

    pkl.dump(tr_data,open(os.path.join(save_dir,'mels_train_curated_split.pkl'),'wb'))
    pkl.dump(val_data,open(os.path.join(save_dir,'mels_val_curated_split.pkl'),'wb'))
    pkl.dump(test_data,open(os.path.join(save_dir,'mels_test_curated_split.pkl'),'wb'))

    tr_csv_df.to_csv(os.path.join(save_dir,'train_curated_split.csv'),index=False)
    val_csv_df.to_csv(os.path.join(save_dir,'val_curated_split.csv'),index=False)
    test_csv_df.to_csv(os.path.join(save_dir,'test_curated_split.csv'),index=False)


    print ('train len:{},val len:{}, test len:{}'.format(tr_csv_df.shape[0],val_csv_df.shape[0],test_csv_df.shape[0]))




if __name__ == '__main__':
    main()
