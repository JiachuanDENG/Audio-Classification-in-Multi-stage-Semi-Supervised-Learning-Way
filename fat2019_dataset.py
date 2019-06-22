import pickle as pkl
import pandas as pd
import numpy as np
import math
import torch
np.random.seed(1994)

class FATDataset(object):
    def __init__(self,datapath,val_datapath,csvpath,val_csv_path,batch_size = 64,duration = 128,istraining=True,init_shuffle=True):
        self.istraining = istraining

        self.duration = duration
        self.batch_size = batch_size
        print ('loading data from pkl...')
        with open(datapath, 'rb') as f:
            self.data_tr= pkl.load(f)
        self.csv_df  = pd.read_csv(csvpath)

        with open(val_datapath,'rb') as f:
            self.data_val = pkl.load(f)
        self.val_csv_df = pd.read_csv(val_csv_path)

        # reference dataframe to get labels order
        self.ref_df = pd.read_csv('/data/jiachuan/freesound_kaggle/sample_submission.csv')
        self.labels_str = self.ref_df.columns[1:].tolist() # a  map label string -> index
        self.num_classes= len(self.labels_str)
        self.init_labels()
        if init_shuffle:
            self.shuffle_trainingdata()

    def get_class_num(self):
        return self.num_classes

    def select_datafromlist(self,data,idx):
        # data: list(np.array)
        # idx: np.array [N,]
        return [data[int(i)] for  i in  idx]



    def shuffle_trainingdata(self,):
        idx = np.random.permutation(len(self.data_tr))
        self.data_tr,self.labels_tr = self.select_datafromlist(self.data_tr,idx), self.labels_tr[idx]

    def get_numof_batch(self,istraindata=True):
        if istraindata:
            return int(math.ceil (len(self.data_tr)/self.batch_size))
        else:
            return int(math.ceil(len(self.data_val)/self.batch_size))
        # print (len(self.data),self.labels.shape)
    def get_numof_epoch_sample(self):
        return len(self.data_tr)

    def  init_labels(self):
        """
        return a np.array[N,num_classes]
        """
        # for training
        self.labels_tr = np.zeros((len(self.csv_df), self.num_classes)).astype(int)
        for i, row in enumerate(self.csv_df['labels'].str.split(',')):
            for label in row:
                idx = self.labels_str.index(label)
                self.labels_tr[i, idx] = 1

        # for validation
        self.labels_val = np.zeros((len(self.val_csv_df), self.num_classes)).astype(int)
        for i, row in enumerate(self.val_csv_df['labels'].str.split(',')):
            for label in row:
                idx = self.labels_str.index(label)
                self.labels_val[i, idx] = 1



    def cut_from_onedata(self,data):
        # data: np.array [128,x,3]
        # result: np.array [m,3,duration,128]
        #  m: int  how many samples cut from the original data

        m = int(math.ceil(data.shape[1]/self.duration))
        # padding
        res = np.zeros([data.shape[0],self.duration*m,data.shape[2]])
        res[:,:data.shape[1],:] = data
        res = np.swapaxes(res,0,2) #[3,duration*m,128]
        res = np.reshape(res,[res.shape[0],m,self.duration,res.shape[2]])#[3,m,duration,128]

        res = np.swapaxes(res,0,1) #[m,3,duration,128]
        return res,m


    def cut_from_batchdata(self,batch_data):
        # batch_data:  list of  np.array  batch_size * [128,x,3]
        # result : np.array [M, 3, duration, 128] , where  M could  be varied in different batch
        # subsamplenum_list: list(int) how many samples are cut from each original sample in one batch
        res = np.array([]).reshape(0,batch_data[0].shape[2],self.duration,batch_data[0].shape[0])
        subsamplenum_list = []
        for data in batch_data:
            cut_data,m = self.cut_from_onedata(data)
            res = np.vstack((res,cut_data))
            subsamplenum_list.append(m)
        return res,subsamplenum_list


    def get_data(self,batchidx,istraindata=True):
        if istraindata:
            data,labels = self.data_tr,self.labels_tr
        else:
            data,labels = self.data_val,self.labels_val

        batch_data = data[batchidx*self.batch_size:(batchidx+1)*self.batch_size] #list of np.array: batch_size * np.array[128,x,3]
        labels = labels[batchidx*self.batch_size:(batchidx+1)*self.batch_size,:] #np.array[batch_size,num_classes]
        cut_batch_data,subsamplenum_list = self.cut_from_batchdata(batch_data)
        return torch.from_numpy(cut_batch_data).float(),subsamplenum_list,torch.from_numpy(labels).float()
