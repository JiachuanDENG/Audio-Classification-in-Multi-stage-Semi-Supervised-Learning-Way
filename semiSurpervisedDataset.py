import numpy as np
import pandas as pd
import math
import pickle as pkl
from tqdm import tqdm
import torch
import copy
import configparser

config=configparser.ConfigParser()
config.read('./config.ini')

np.random.seed(0)

class SemiSupervisedDataset(object):
    def __init__(self,curated_data_path,curated_csv,noisy_data_path,noisy_csv,val_data_path,val_csv,batch_size = 64,duration = 128,masking_max_percentage=0.15):
        self.duration = duration
        self.batch_size = batch_size

        self.masking_max_percentage = masking_max_percentage

        print ('loading curated data from pkl...')
        with open(curated_data_path, 'rb') as f:
            self.curated_data= pkl.load(f)

        with open(val_data_path,'rb') as f:
            self.val_data = pkl.load(f)

        print ('loading noisy data from pkl...')
        with open(noisy_data_path, 'rb') as f:
            self.noisy_data= pkl.load(f)

        self.curated_csv_df  = pd.read_csv(curated_csv)
        self.val_csv_df = pd.read_csv(val_csv)
        self.noisy_csv_df  = pd.read_csv(noisy_csv)

        # reference dataframe to get labels order
        self.ref_df = pd.read_csv(config.get('DataPath','ref_csv_path'))
        self.labels_str = self.ref_df.columns[1:].tolist() # a  map label string -> index
        self.num_classes= len(self.labels_str)
        self.init_labeled_unlabeled_data()

    def get_class_num(self):
        return self.num_classes

    def get_numof_batch(self,istraindata=True):
        if istraindata:
            return int(math.ceil (len(self.labeled_data_epoch)/self.batch_size))
        else:
            return int(math.ceil(len(self.val_data)/self.batch_size))
    def get_samplenum_perEpoch(self):
        assert len(self.unlabeld_data_epoch ) == len(self.labeled_data_epoch),'labeled data and unlabeled data len not match'
        return len(self.labeled_data_epoch)*2 # containing N/2 labeled and N/2 unlabeled data

    def init_labeled_unlabeled_data(self):
        """
        initialize labeled and unlabled data
        will get:
            1.  self.val_labels: list of np.array read from pre-splited file
            2. self.labeled_data,self.labels: list of np.array consists of train data split from curated data + noisy data with guessing labels
            3. self.unlabeld_data: list of np.array, noisy data without guessing labels
        """
        self.labeled_data = []
        self.unlabeld_data = []
        self.labels = np.array([]).reshape(0,self.num_classes)

        # all curated data should be labeled data
        curated_labels = np.array([]).reshape(0,self.num_classes)
        for i, row in enumerate(self.curated_csv_df['labels'].str.split(',')):
            onesample_label = np.zeros([1,self.num_classes])
            for label in row:
                idx = self.labels_str.index(label)
                onesample_label[0,idx] = 1
            curated_labels = np.vstack((curated_labels,onesample_label))

        assert len(self.curated_data) == curated_labels.shape[0],'curated data and labels len not match'


        # get val labels
        self.val_labels = np.array([]).reshape(0,self.num_classes)
        for i, row in enumerate(self.val_csv_df['labels'].str.split(',')):
            onesample_label = np.zeros([1,self.num_classes])
            for label in row:
                idx = self.labels_str.index(label)
                onesample_label[0,idx] = 1
            self.val_labels = np.vstack((self.val_labels,onesample_label))


        # add split out training curated data into labeled_data
        self.labeled_data += self.curated_data
        self.labels = np.vstack((self.labels,curated_labels))

        # only parts of noisy data can be labeled data
        for i,row in enumerate(self.noisy_csv_df['labels'].str.split(',')):
            if type(row) != type([]):# which means row is NaN
                self.unlabeld_data.append(self.noisy_data[i])
            else:
                self.labeled_data.append(self.noisy_data[i])
                onesample_label = np.zeros([1,self.num_classes])
                for label in row:
                    idx = self.labels_str.index(label)
                    onesample_label[0,idx] = 1
                self.labels = np.vstack((self.labels,onesample_label))
        assert len(self.labeled_data) == self.labels.shape[0],'labeled data len {} != labels len {}'.format(len(self.labeled_data),self.labels.shape[0])
        print ('labeled data:{}, unlabeld_data:{}'.format(len(self.labeled_data),len(self.unlabeld_data)))

    def select_datafromlist(self,data,idx):
        # data: list(np.array)
        # idx: np.array [N,]
        return [data[int(i)] for  i in  idx]



    def shuffle_and_cut(self,):
        """
        since labeled and unlabeled data will be used in same batch half half,
        we need to make sure the number of labeled and unlabeled data are with same size in one epoch

        this function are required to be used before begining each epoch
        """
        # shuffle labeled data
        shuffle_idx = np.random.permutation(len(self.labeled_data))
        self.labeled_data,self.labels = self.select_datafromlist(self.labeled_data,shuffle_idx),self.labels[shuffle_idx]
        # shuffle unlabeled data
        shuffle_idx = np.random.permutation(len(self.unlabeld_data))
        self.unlabeld_data = self.select_datafromlist(self.unlabeld_data,shuffle_idx)

        oneEpochLen = min(len(self.labeled_data),len(self.unlabeld_data))

        self.labeled_data_epoch,self.labels_epoch = self.labeled_data[:oneEpochLen],self.labels[:oneEpochLen]
        self.unlabeld_data_epoch = self.unlabeld_data[:oneEpochLen]


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



    def spec_augment(self,data_batch):
        """
        doing stochastic data augmentation on each data sample
        reference: https://arxiv.org/pdf/1904.08779.pdf
        data_batch : list of np.array
        return :
            data_batch_cp: list of np.array, which has been data augmented
        """
        h_percentage = np.random.uniform(low=0., high=self.masking_max_percentage, size=len(data_batch))
        w_percentage = np.random.uniform(low=0., high=self.masking_max_percentage, size=len(data_batch))


        data_batch_cp = copy.deepcopy(data_batch) # to avoid change the original data, we have to use deepcopy [batch_size,128,x,3]

        for i in range(len(data_batch_cp)):
            height = data_batch_cp[i].shape[0]
            width = data_batch_cp[i].shape[1]

            h_mask = int(h_percentage[i] * height)
            h = int(np.random.uniform(0.,height-h_mask))
            data_batch_cp[i][h:h+h_mask,:,:] = 0

            w_mask = int(w_percentage[i]*width)
            w = int(np.random.uniform(0., width - w_mask))
            data_batch_cp[i][:,w:w+w_mask,:] = 0

        return data_batch_cp






    def get_data(self,batchidx,istraining=True):
        """
        get a batch of data consists of both labeled_data & unlabeld_data(half half) for training process
        and get a batch data consists of only curated data for validation process
        batchidx: int

        """
        if istraining:
            # labeled data part
            batch_labeled_data = self.spec_augment(self.labeled_data_epoch[batchidx*self.batch_size:(batchidx+1)*self.batch_size]) #list of np.array: batch_size * np.array[128,x,3]
            labels = self.labels_epoch[batchidx*self.batch_size:(batchidx+1)*self.batch_size,:] #np.array[batch_size,num_classes]
            cut_batch_labeled_data,labeled_data_subsamplenum_list = self.cut_from_batchdata(batch_labeled_data)

            # unlabeled data part
            batch_unlabeled_data = self.spec_augment(self.unlabeld_data_epoch[batchidx*self.batch_size:(batchidx+1)*self.batch_size])
            cut_batch_unlabeled_data,unlabeled_data_subsamplenum_list = self.cut_from_batchdata(batch_unlabeled_data)

            # labels for this batch, where unlabeled data's label will be all zeros
            assert len(batch_labeled_data) == len(batch_unlabeled_data), 'labeled data and unlabeled data size not equal'
            batch_labels = np.zeros([labels.shape[0]*2,labels.shape[1]])
            batch_labels[:labels.shape[0],:] = labels
            batch_labels[labels.shape[0]:,0] = -1 # mark the first element of labels as -1 to indicate it is unlabeled data

            # cuted data for this batch
            cut_batch_data = np.concatenate((cut_batch_labeled_data,cut_batch_unlabeled_data),axis=0)

            # samplenum list for this batch
            subsamplenum_list = labeled_data_subsamplenum_list + unlabeled_data_subsamplenum_list
        else:
            data,labels = self.val_data,self.val_labels

            batch_data = data[batchidx*self.batch_size:(batchidx+1)*self.batch_size] #list of np.array: batch_size * np.array[128,x,3]
            batch_labels = labels[batchidx*self.batch_size:(batchidx+1)*self.batch_size,:] #np.array[batch_size,num_classes]
            cut_batch_data,subsamplenum_list = self.cut_from_batchdata(batch_data)


        return torch.from_numpy(cut_batch_data).float(),subsamplenum_list,torch.from_numpy(batch_labels).float()
