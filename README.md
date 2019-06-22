# Audio-Classification-in-Multi-stage-Semi-Supervised-Learning-Way
Audio Classification with Noisy Dataset with multi stage semi-supervised learning

## Problem Definition: 
Given a wav file (in variable length), predict its corresponding label(s), each wav could be in multiple classes
![task2_freesound_audio_tagging](https://user-images.githubusercontent.com/20760190/59892155-d9c85480-938c-11e9-8e64-65582cec6b32.png)

## Data Set:
Original Dataset can be found in Kaggle: https://www.kaggle.com/c/freesound-audio-tagging-2019/data.

To save time in data preprocessing, we also use the processed dataset (converting raw wav data to numpy matrix with Logmel transformation)https://www.kaggle.com/daisukelab/fat2019_prep_mels1

The dataset consists of both curated data (with accurate labels), and noisy data (with labels, but not sure whether accurate or not). Noisy data size is much larger than curated data size.

## Method
### Model
In our code, we have implemented multiple models(CNN, CNN+LSTM, ResNet), for simplicity, experiments are done based on CNN model by default.

Since CNN type model only allow fixed length input, while the data input length in our dataset is variable, we need to cut the long input audio into segments with fixed length (padding if necessary), 
and use the average of each segment's prediction as final prediction of the original audio data.


![Screen Shot 2019-06-20 at 7 20 22 PM](https://user-images.githubusercontent.com/20760190/59893091-8ce67d00-9390-11e9-92c4-5529ae6c0ff7.png)

### Multi Stage Training

![Screen Shot 2019-06-20 at 11 50 31 PM](https://user-images.githubusercontent.com/20760190/59903592-368c3500-93b6-11e9-9a98-06f471af0539.png)

#### Stage 0. WARM UP
Train the model on roughly selected noisy data (i.e. mels_trn_noisy_best50s.pkl in https://www.kaggle.com/daisukelab/fat2019_prep_mels1), details for how to roughly select from noisy data can be found in 
https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data

#### Stage 1. Fine Tune
Started from Model 0 which is trained in Stage 0, we train the model again on the curated dataset.

#### Stage 1.5. Filter Noisy Data
Using Model 1 which is trained in Stage 1, we can filter out parts of noisy data which we are confident that its corresponding labels are correct. At the end of this operation, we will get: ***1***. labeled data 
(consists of curated data and noisy data we are confident on its labels) and ***2***. unlabeled data (noisy data that we are inconfident on its labels)

#### Stage 2. Semi-Supervised Learning
Both ***labeled data*** {x<sub>l</sub>, y<sub>l</sub>} and ***unlabeled data*** {x<sub>u</sub>} will be used in this stage. Before the input data feeding into classifier, a stochastic data augmentation is required. Here we use [SpecAugment](https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html) as the augmentation <br/>
***Loss*** function consists of two parts: </br>
***1.*** For {x<sub>l</sub>, y<sub>l</sub>} BCELoss will be calculated</br>
***2.*** For both {x<sub>l</sub>} and {x<sub>u</sub>} will do stochastic augmentation by 2 times: Take x<sub>l</sub> for example ![CodeCogsEqn (2)](https://user-images.githubusercontent.com/20760190/59935231-36b22200-9402-11e9-927e-d559d68d6f68.gif)
, where f<sub>&theta;</sub> refers to the classifier and ***g*** refers to data augmentation function. Then the squared difference loss will be calculated on the model outputs: ![CodeCogsEqn (1)](https://user-images.githubusercontent.com/20760190/59935230-36b22200-9402-11e9-8479-21ce5af04dce.gif).  The main idea of this loss is to regularize the network such that it generates about the same outputs for the same data input that undergoes data augmentation.

### Experiment Results
Since only curated data (i.e. mels_train_curated.pkl in FAT2019 dataset) have correct labels, the evaluation is done based on this data. We split the mels_train_curated.pkl into three parts: curated training data, curated validation data and curated testing data in 8:1:1.

Evaluation Metrics we use is [***label-weighted label-ranking average precision***](https://scikit-learn.org/stable/modules/model_evaluation.html#label-ranking-average-precision)

And here is the results for each stage:

| Stage | Validation | Testing
| ------------- | ------------- | -----------
| Stage 0 |  0.285 | 0.282
| Stage 1  | 0.828 | 0.791
| Stage 2 | 0.836 | 0.816

