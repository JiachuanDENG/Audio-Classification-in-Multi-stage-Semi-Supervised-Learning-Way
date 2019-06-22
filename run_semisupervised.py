from semiSurpervisedDataset import SemiSupervisedDataset
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from pydub import AudioSegment
from tqdm import tqdm
import collections
import numpy as np
import matplotlib.pyplot as plt
import math
import utils
import glob
from random import shuffle
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from onehotencoder import One_hot_encoder
import torch.nn as nn
import torch.autograd as autograd
import models
import logging
from time import gmtime, strftime
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import oneSampleOutput
import torch.nn.functional as F
import configparser
import os
config=configparser.ConfigParser()
config.read('./config.ini')

currenttime = strftime("%Y-%m-%d-%H-%M-%S", gmtime())


logger = logging.getLogger('freesound_kaggle')
hdlr = logging.FileHandler('./log/semi_log_fat2019_{}.log'.format(currenttime))
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)
hdlr.setLevel(logging.INFO)


def loss_func(criterion,Z1,Z2,labels,alpha=0.01):
    """
    criterion: classification criterion, eg. BCELoss
    Z1: output of data augmented 1
    Z2: output of data augmented 2
    labels: element 0 == -1 indicate to be unlabeled data
    loss: 1/(N/2)*\sum log Z_l^1 y_i + 1/N*\sum ||Z_i^1 - Z_i^2||_2^2
    reference: https://ieeexplore.ieee.org/document/8354242
    """

    labeled_Z1,unlabeled_Z1 = Z1[labels[:,0]!=-1],Z1[labels[:,0]==-1]
    labeled_Z2,unlabeled_Z2 = Z2[labels[:,0]!=-1],Z2[labels[:,0]==-1]
    labeled_labels = labels[labels[:,0]!=-1]
    loss = criterion(labeled_Z1,labeled_labels) + alpha*torch.sum(torch.sum((F.softmax(Z1,dim=1)-F.softmax(Z2,dim=1))**2,dim=1),dim=0)/(Z1.shape[0])
    return loss

def train(device,model_path='',loadModel=False):
    EPOCH = 50
    printout_steps = 50
    eval_steps = 200
    lr_steps = 300
    lr = 3e-3
    t_max = 200
    eta_min = 3e-6

    print ('initialize dataset...')
    curated_data_path = os.path.join(config.get('DataPath','split_dir'),'mels_train_curated_split.pkl')
    curated_csv = os.path.join(config.get('DataPath','split_dir'),'train_curated_split.csv')
    noisy_data_path = config.get('DataPath','noisy_data_path')
    noisy_csv = config.get('DataPath','filtered_noisy_csv_path')

    val_datapath = os.path.join(config.get('DataPath','split_dir'),'mels_val_curated_split.pkl')
    val_csv_path = os.path.join(config.get('DataPath','split_dir'),'val_curated_split.csv')

    voiceDataset = SemiSupervisedDataset(curated_data_path,curated_csv,\
                                            noisy_data_path,noisy_csv,\
                                            val_datapath,val_csv_path,\
                                            batch_size=4)
    # print (voiceDataset.get_class_num())
    print ('create model ... ')

    cnnmodel = models.CNNModelv2(voiceDataset.get_class_num()).to(device)
    if loadModel:
        print ('loading model from {}...'.format(model_path))
        cnnmodel.load_state_dict(torch.load(model_path))

    optimizer=torch.optim.Adam(cnnmodel.parameters(),lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    criterion = nn.BCEWithLogitsLoss()
    bestlwlrap = -1
    for e in range(EPOCH):
        voiceDataset.shuffle_and_cut()
        num_step_per_epoch = voiceDataset.get_numof_batch(True)
        # print (num_step_per_epoch)
        for bidx in tqdm(range(num_step_per_epoch)):
            # Z1
            batch_data1,samplenumbatch1,label_batch1 = voiceDataset.get_data(bidx,True) #[M, 128, duration, 3]
            # Z2
            # Z1 and Z2 should be from two exactly  same batch, but use different data augmentation
            batch_data2,samplenumbatch2,label_batch2 = voiceDataset.get_data(bidx,True) #[M, 128, duration, 3]

            # print ('fingerprint got...')
            if batch_data1.shape[0] <= 1:
                continue
            # print (batch_data.shape,label_batch.shape)
            bx1 = batch_data1.to(device)
            bx2 = batch_data2.to(device)

            output1 = cnnmodel(bx1)
            output2 = cnnmodel(bx2)
            Z1 = oneSampleOutput(output1,samplenumbatch1).to(device)
            Z2 = oneSampleOutput(output2,samplenumbatch2).to(device)

            # print (Z1.shape,Z2.shape,label_batch1.shape)
            by = label_batch1.to(device) # label_batch1 should be exactly same as label_batch2
#
            loss = loss_func(criterion,Z1,Z2, by)
#             loss = autograd.Variable(loss, requires_grad = True)
            if bidx % printout_steps == 0:
                msg = '[TRAINING] Epoch:{}, step:{}/{}, loss:{}'.format(e,bidx,num_step_per_epoch,loss)
                print (msg)
                logger.info(msg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (bidx+1) % eval_steps == 0:
                # doing validation
                cnnmodel.eval()
                val_batches_num = voiceDataset.get_numof_batch(False)
                val_preds = np.array([]).reshape(0,voiceDataset.get_class_num())
                val_labels = np.array([]).reshape(0,voiceDataset.get_class_num())
                # val_loss = 0.
                for vbidx  in tqdm(range(val_batches_num)):

                    #
                    val_data,val_samplenumbatch,val_label = voiceDataset.get_data(vbidx,False)
                    # print ('val_data shape:',val_data.shape)
                    pred = oneSampleOutput(cnnmodel(val_data.to(device)).detach(),val_samplenumbatch).to(device)
                    val_preds = np.vstack((val_preds,pred.cpu().numpy()))
                    # print (pred.shape)
                    # print (criterion(pred,val_label.to(device)))
                    # val_loss += criterion(pred,val_label.to(device)).item()/val_label.shape[0]
                    val_labels = np.vstack((val_labels,val_label.cpu().numpy()))
                score, weight = utils.calculate_per_class_lwlrap(val_labels, val_preds)
                lwlrap = (score * weight).sum()
                msg = '[VALIDATION] Epoch:{}, step:{}:/{},  lwlrap:{}'.format(e,bidx,num_step_per_epoch,lwlrap)
                print (msg)
                logger.info(msg)
                if lwlrap > bestlwlrap :
                    bestlwlrap = lwlrap
                    #save model
                    save_model_filename =  config.get('SaveModel','stage2_model')
                    save_model_path = os.path.join(config.get('DataPath','checkpoint_dir'), save_model_filename)
                    torch.save(cnnmodel.state_dict(), save_model_path)
                    msg = 'save model to: {}'.format(save_model_path)
                    print (msg)
                    logger.info(msg)

                cnnmodel.train()
            if bidx % lr_steps == 0:
                scheduler.step()




if __name__ == '__main__':
    # if False: screeb 8914
    if torch.cuda.is_available():
        print ('use cuda ...')
        device = 'cuda'
    else:
        print ('use cpu...')
        device = 'cpu'

    # model_path = './checkpoint/fat_2019-06-19-07-06-26_epoch_74_bidx_400.model' # stage 2 0618
    model_path = os.path.join(config.get('DataPath','checkpoint_dir'),config.get('SaveModel','stage1_model'))
    loadModel =   True
    train(device,model_path,loadModel)
