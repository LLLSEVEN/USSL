import torch
from torch.utils.data.dataset import Dataset
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader
from util import BackgroundGenerator
import numpy as np


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class CustomDataSet(Dataset):
    def __init__(self, images, texts, labels,ml_labels):
        self.images = images
        self.texts = texts
        self.labels = labels
        self.ml_labels=ml_labels

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        label = self.labels[index]
        ml_labels=self.ml_labels[index]
        return img, text, label,ml_labels

    def __len__(self):
        count = len(self.images)
        return count


class SingleModalDataSet(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        count = len(self.data)
        return count


def get_loader(path, batch_size):
    img_train = loadmat(path + "train_img.mat")['train_img']  
    img_test = loadmat(path + "test_img.mat")['test_img']  
    text_train = loadmat(path + "train_txt.mat")['train_txt']
    text_test = loadmat(path + "test_txt.mat")['test_txt']
    label_train = loadmat(path + "train_lab.mat")['train_lab'] 
    label_test = loadmat(path + "test_lab.mat")['test_lab']

    ml_train=np.unique(label_train,axis=0,return_inverse=True)
    ml_train_index=ml_train[1]

    ml_test=np.unique(label_test,axis=0,return_inverse=True)#
    ml_test_index=ml_test[1]

    imgs = {'train': img_train, 'test': img_test}
    texts = {'train': text_train, 'test': text_test}
    labels = {'train': label_train, 'test': label_test}
    ml_labels={'train': ml_train_index, 'test': ml_test_index} 

    shuffle = {'train': True, 'test': False}
    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x] ,ml_labels=ml_labels[x])
                for x in ['train', 'test']}
    dataloader = {x: DataLoaderX(dataset[x], batch_size=batch_size,
                                    shuffle=shuffle[x], num_workers=0) for x in ['train', 'test']}

    img_dim = img_train.shape[1]  
    text_dim = text_train.shape[1]  
    num_class = label_train.shape[1]  
 
    input_data_par = {}
    input_data_par['img_test'] = img_test
    input_data_par['text_test'] = text_test
    input_data_par['label_test'] = label_test
    input_data_par['img_train'] = img_train 
    input_data_par['text_train'] = text_train 
    input_data_par['label_train'] = label_train 
    input_data_par['img_dim'] = img_dim   
    input_data_par['text_dim'] = text_dim
    input_data_par['num_class'] = num_class
    input_data_par['ml_train']=ml_train[0]  
    input_data_par['ml_train_index']=ml_train_index   
    return dataloader, input_data_par








