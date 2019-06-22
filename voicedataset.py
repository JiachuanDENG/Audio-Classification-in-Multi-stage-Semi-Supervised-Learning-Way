import torch
from random import shuffle
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import scipy.io.wavfile as wave
import utils
import math

n_logfbank = 40
samplerate_=16000
winlen_=0.02
winstep_=0.01
#fel bank
def get_logfbank(f_path):
    fs,audio = wave.read(f_path) #int16
    # print (audaio.dtype)
    logmel_fea = utils.logfbank(audio,fs,winlen_,winstep_,nfilt=n_logfbank,nfft=512,lowfreq=0,highfreq=None,preemph=0.97)
    # delta1 = delta(logmel_fea, 2)
    # delta2 = delta(delta1, 2)
#     print ('logmel_fea:{},delta1:{},delta2:{}'.format(logmel_fea.shape,delta1.shape,delta2.shape))
#     final_fea=np.concatenate((logmel_fea[np.newaxis,:,:],delta1[np.newaxis,:,:],delta2[np.newaxis,:,:]),axis=0) # len = 3999
    return logmel_fea
#     return final_fea

class VoiceDataset(Dataset):

    def __init__(self, datapath,datacsv,onehotencoder,segmentframes=300,batch_size = 64, validation_ratio = 0.1):

        self.path = datapath
        self.datacsv_df = pd.read_csv(datacsv)
        self.segmentframes = segmentframes
        self.batch_size = batch_size
        self.onehotencoder =  onehotencoder
        self.samples = []
        print ('initialiate samples...')
        for i in tqdm(range(self.datacsv_df.shape[0])):
            fn = self.datacsv_df.loc[i]['fname']
            labels = self.datacsv_df.loc[i]['labels'].split(',')
            if os.path.exists(os.path.join(self.path,fn)):
                self.samples.append({'wavfilepath':os.path.join(self.path,fn),'labels':self.getlabels(labels)})
#         self.wavfiles = [os.path.join(self.path,f) for f in os.listdir(self.path) if '.wav' in f]
        shuffle(self.samples)
        self. training_samples_num = int(len(self.samples)*(1-validation_ratio))
        self.training_samples = self.samples[:self.training_samples_num]
        self.validation_samples = self.samples[self.training_samples_num:]
        # print('samples per epoch:{}'.format(len(self.samples)))

    def getlabels(self,labels):
        # get labels(str) in one hot(np.array)
        onehot = np.zeros(self.onehotencoder.get_numlabels())
        for l in labels:
            onehot += self.onehotencoder.label2onehot(l)
        return onehot
    def get_class_num(self):
        return self.onehotencoder.get_numlabels()

    def shufflefiles(self):
        shuffle(self.training_samples)

    def get_traininglen(self):
        return len(self.training_samples)

    def get_validationlen(self):
        return len(self.validation_samples)

    def get_numof_batch(self,istraining=True):
        if istraining:
            return len(self.training_samples)//self.batch_size
        else:
            return len(self.validation_samples)//self.batch_size

    def get_feature(self,wavfilename):
        logmel = get_logfbank(wavfilename) #[n,40]
        subsamplenum = int(math.ceil(logmel.shape[0]/self.segmentframes))
        logmel_pad = np.zeros([subsamplenum*self.segmentframes,logmel.shape[1]])
        logmel_pad[:logmel.shape[0],:] = logmel
        logmel_seg = logmel_pad.reshape([-1,self.segmentframes,logmel.shape[1]])
        return logmel_seg,subsamplenum

    def get_data(self, bidx,istraining=True):
        #bidx: batch idx
        if istraining:
            samples = self.training_samples[bidx*self.batch_size:(bidx+1)*self.batch_size]
        else:
            samples = self.validation_samples[bidx*self.batch_size:(bidx+1)*self.batch_size]
        logmel_batch = np.array([]).reshape(0,self.segmentframes,40)
        subsampleNumBatch = []
        label_batch = np.array([]).reshape(0,self.onehotencoder.get_numlabels())
        for sample in samples:
            logmel,subsamplenum = self.get_feature(sample['wavfilepath'])
            label = sample['labels']
#             print (logmel.shape)
            logmel_batch = np.vstack((logmel_batch,logmel))
            label_batch = np.vstack((label_batch,label))
            # label_batch = np.vstack((label_batch,np.tile(label,(subsamplenum,1))))
            subsampleNumBatch.append(subsamplenum)

        return torch.from_numpy(logmel_batch[:,np.newaxis,:,:]).float(),subsampleNumBatch,torch.from_numpy(label_batch).float()
