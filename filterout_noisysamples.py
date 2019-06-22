import torch
import models
from fat2019_dataset import FATDataset
from utils import oneSampleOutput
from tqdm import tqdm
import numpy as np
import pandas as pd
import configparser
import os
config=configparser.ConfigParser()
config.read('./config.ini')

def get_topNpred(model_output,N=5):
    """
    model_output: np.array [batch_size,num_classes]
    return: np.array [batch_size,N]
    """
    return model_output.argsort()[:,-N:][:,::-1]


def get_common_labels(model_output,noisyLabels):
    """
    model_output,noisyLabels: np.array[batch_size,num_classes]
    note that noisyLabels is in one-hot
    return :
        isempty_array: np.array [batch_size,] indicated whether each sample is filtered to have labels or not
        filtered_labels: np.array[batch_size,num_classes]
    """
    topNpred = get_topNpred(model_output,N=5)#topNpred: np.array [batch_size,N]
    isempty_array = np.full(noisyLabels.shape[0], False)
    filtered_labels = np.zeros(noisyLabels.shape)
    for i in range(topNpred.shape[0]):
        predictedlabels = topNpred[i,:] # in index

        for predictedlabel in predictedlabels:
            if noisyLabels[i,predictedlabel] == 1:
                filtered_labels[i,predictedlabel] = 1
                isempty_array[i] = True # means noisy sample i have filtered label(s)
    return isempty_array,filtered_labels


def save_filtered_results(ref_csv,filtered_labels,labels_str):
    """
    ref_csv: noisy data labels csv, filtered labels will be filled in same format
    isempty_array: array indicate whether each sample will have filtered label or not
    filtered_labels: filtered labels
    labers_str: a list of label str used in filtering process, which maps labelidx -> label str
    """

    ref_csv_df  = pd.read_csv(ref_csv)
    for i in range(ref_csv_df.shape[0]):
        label_str = ''
        for j in range(filtered_labels.shape[1]):
            l = filtered_labels[i,j]
            if l == 1:
                if len(label_str) != 0:
                    label_str = label_str+',' + labels_str[j]
                else:
                    label_str = labels_str[j]
        ref_csv_df.iloc[i].labels = label_str
    ref_csv_df.to_csv(config.get('DataPath','filtered_noisy_csv_path'),index=False)



def filter_noisysamples(noisy_traindatadir,val_datadir,noisy_traindatacsv,val_datacsv,device,model_path='',loadModel=False):

    voiceDataset = FATDataset(noisy_traindatadir,val_datadir,noisy_traindatacsv,val_datacsv,batch_size=8,istraining=False,init_shuffle=False)

    print ('create model ... ')

    cnnmodel = models.CNNModelv2(voiceDataset.get_class_num()).to(device)
    if loadModel:
        print ('loading model from {}...'.format(model_path))
        cnnmodel.load_state_dict(torch.load(model_path))

    num_step_per_epoch = voiceDataset.get_numof_batch(istraindata=True)
    # print (num_step_per_epoch)
    samplenum_total = 0
    isempty_array_total = np.array([]).reshape(0,)
    filtered_labels_total = np.array([]).reshape(0,voiceDataset.get_class_num())
    pass_filter = 0
    for bidx in tqdm(range(num_step_per_epoch)):

        batch_data,samplenumbatch,label_batch = voiceDataset.get_data(bidx,True) #[M, 128, duration, 3]
        # print ('fingerprint got...')
        # print (batch_data.shape,label_batch.shape)
        samplenum_total += len(samplenumbatch)
        bx = batch_data.to(device)

        output = cnnmodel(bx)
        model_output = oneSampleOutput(output,samplenumbatch).to('cpu').data.numpy() # [batch_size,num_classes]
        isempty_array,filtered_labels = get_common_labels(model_output, label_batch.to('cpu').numpy())
        # print (isempty_array.shape,filtered_labels.shape)
        isempty_array_total = np.concatenate ((isempty_array_total,isempty_array),axis=0)
        filtered_labels_total = np.vstack((filtered_labels_total,filtered_labels))


    print ('filtered out {}/{} labels from noisy data'.format(sum(isempty_array_total),isempty_array_total.shape[0]))


    assert samplenum_total == isempty_array_total.shape[0] and samplenum_total == filtered_labels_total.shape[0],'sample num error'


    assert samplenum_total == voiceDataset.get_numof_epoch_sample(),'sample number error,{},{}'.format(samplenum_total,voiceDataset.get_numof_epoch_sample())
    ## TODO: save filtered labels to csv
    save_filtered_results(ref_csv=config.get('DataPath','noisy_csv_path'),\
                            filtered_labels= filtered_labels_total,\
                            labels_str=voiceDataset.labels_str
                            )

if __name__ == '__main__':
    # if False:
    if torch.cuda.is_available():
        print ('use cuda ...')
        device = 'cuda'
    else:
        print ('use cpu...')
        device = 'cpu'

    datapath = config.get('DataPath','noisy_data_path')
    csvpath = config.get('DataPath','noisy_csv_path')


# actually val will not be used in this stage
    val_datapath = os.path.join(config.get('DataPath','split_dir'),'mels_val_curated_split.pkl')
    val_csv_path = os.path.join(config.get('DataPath','split_dir'),'val_curated_split.csv')

    model_path = os.path.join(config.get('DataPath','checkpoint_dir'),config.get('SaveModel','stage1_model'))
    loadModel =  True
    filter_noisysamples(datapath,val_datapath,csvpath,val_csv_path,device,model_path,loadModel)
