from fat2019_dataset  import FATDataset
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
import sys
import configparser

currenttime = strftime("%Y-%m-%d-%H-%M-%S", gmtime())


logger = logging.getLogger('freesound_kaggle')
hdlr = logging.FileHandler('./log/log_fat2019_{}.log'.format(currenttime))
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)
hdlr.setLevel(logging.INFO)



config=configparser.ConfigParser()
config.read('./config.ini')

checkpoint_dir = config.get('DataPath','checkpoint_dir')
if not os.path.exists(checkpoint_dir):
    os.system('mkdir {}'.format(checkpoint_dir))



def train(traindatadir, valdatadir, traindatacsv,valdatacsv,device,model_path,save_model_filename,loadModel=False):
    EPOCH = 100
    printout_steps = 50
    eval_steps = 200
    lr_steps = 300
    lr = 3e-3
    t_max = 200
    eta_min = 3e-6

    print ('initialize dataset...')
    voiceDataset = FATDataset(traindatadir, valdatadir, traindatacsv,valdatacsv,batch_size=8)
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
        voiceDataset.shuffle_trainingdata()
        num_step_per_epoch = voiceDataset.get_numof_batch(istraindata=True)
        # print (num_step_per_epoch)
        for bidx in tqdm(range(num_step_per_epoch)):
            # print ('get fingerprint...')
            batch_data,samplenumbatch,label_batch = voiceDataset.get_data(bidx,True) #[M, 128, duration, 3]
            # print ('fingerprint got...')
            if batch_data.shape[0] <= 1:
                continue
            # print (batch_data.shape,label_batch.shape)
            bx = batch_data.to(device)

            output = cnnmodel(bx)
            output = oneSampleOutput(output,samplenumbatch).to(device)
            # print (output.shape,label_batch.shape)
            by = label_batch.to(device)
#             by = autograd.Variable(label_batch,requires_grad = True)
            loss = criterion(output, by)
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
                val_loss = 0.
                for vbidx  in tqdm(range(val_batches_num)):
                    # print ('generating validation fingerprint ... ')
                    val_data,val_samplenumbatch,val_label = voiceDataset.get_data(vbidx,False)
                    # print ('val_data shape:',val_data.shape)
                    pred = oneSampleOutput(cnnmodel(val_data.to(device)).detach(),val_samplenumbatch).to(device)
                    val_preds = np.vstack((val_preds,pred.cpu().numpy()))
                    # print (pred.shape)
                    # print (criterion(pred,val_label.to(device)))
                    val_loss += criterion(pred,val_label.to(device)).item()/val_label.shape[0]
                    val_labels = np.vstack((val_labels,val_label.cpu().numpy()))
                score, weight = utils.calculate_per_class_lwlrap(val_labels, val_preds)
                lwlrap = (score * weight).sum()
                msg = '[VALIDATION] Epoch:{}, step:{}:/{}, loss:{}, lwlrap:{}'.format(e,bidx,num_step_per_epoch,val_loss,lwlrap)
                print (msg)
                logger.info(msg)
                if lwlrap > bestlwlrap or bidx == num_step_per_epoch-1:
                    bestlwlrap = lwlrap
                    #save model
                    save_model_path = os.path.join(checkpoint_dir, save_model_filename)
                    torch.save(cnnmodel.state_dict(), save_model_path)
                    msg = 'save model to: {}'.format(save_model_path)
                    print (msg)
                    logger.info(msg)

                cnnmodel.train()
            if bidx % lr_steps == 0:
                scheduler.step()




if __name__ == '__main__':
    # if False:
    if torch.cuda.is_available():
        print ('use cuda ...')
        device = 'cuda'
    else:
        print ('use cpu...')
        device = 'cpu'

    datapath = sys.argv[1]
    csvpath = sys.argv[2]
    model_path = sys.argv[3]
    loadModel = sys.argv[4].lower() == 'true'
    save_model_filename = sys.argv[5]



    val_datapath = config.get('DataPath','val_data_path')
    val_csv_path = config.get('DataPath','val_csv_path')



    train(datapath,val_datapath,csvpath,val_csv_path,device,model_path,save_model_filename,loadModel)
