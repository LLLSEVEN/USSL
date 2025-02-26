import os
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from scipy.io import loadmat
from model import USSL
from train_model import train_model
from load_data import get_loader
from evaluate import fx_calc_map_label
from scipy.io import loadmat

import torch, gc
gc.collect()  
torch.cuda.empty_cache()
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset = 'mirflickr'  # 'mirflickr' or 'NUS-WIDE-TC21' or 'MS-COCO'
    embedding = 'glove'  
    DATA_DIR = 'data/' + dataset + '/'
    EVAL = False   
    
    training_results = []  
    if dataset == 'mirflickr':
        alpha1=0.8
        alpha2=1 
        alpha3=0.1
        alpha4=0.5
        max_epoch =5
        gamma1=0.1
        gamma2=0.5
        jieta=0.5   
        batch_size = 100
        lr = 5e-5  
        betas = (0.5, 0.999)  
        t = 0.1  
        n_layers = 5   
        k = 8
        temp = 0.35  
        gamma = 0.7  
        num_groups=10
    elif dataset == 'NUS-WIDE-TC21':
        alpha1=0.8
        alpha2=1 
        alpha3=0.1
        alpha4=0.5
        max_epoch =5
        gamma1=0.1
        gamma2=0.5
        jieta=0.5   
        batch_size = 100
        lr = 5e-5  
        betas = (0.5, 0.999)  
        t = 0.1  
        n_layers = 5   
        k = 8
        temp = 0.35  
        gamma = 0.7 
        num_groups=10 
    elif dataset == 'MS-COCO':
        alpha1=0.8
        alpha2=1 
        alpha3=0.1
        alpha4=0.5
        max_epoch =5
        gamma1=0.1
        gamma2=0.5
        jieta=0.5   
        batch_size = 100
        lr = 5e-5   
        betas = (0.5, 0.999)  
        t = 0.1  
        n_layers = 5   
        k = 8
        temp = 0.35  
        gamma = 0.7  
        num_groups=10
    else:
        raise NameError("Invalid dataset name!")

    seed = 103
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if embedding == 'glove':
        inp = loadmat('embedding/' + dataset + '-inp-glove6B.mat')['inp']  
        inp = torch.FloatTensor(inp)
        ml_adj = loadmat('embedding/' + dataset + '-inp-glove6B_ml_adj.mat')['ml_adj'] 
        lml_adj = loadmat('embedding/' + dataset + '-inp-glove6B_lml_adj.mat')["lml_adj"]
        ml_adj= torch.FloatTensor(ml_adj)
        lml_adj = torch.FloatTensor(lml_adj)
    elif embedding == 'googlenews':
        inp = loadmat('embedding/' + dataset + '-inp-googlenews.mat')['inp']
        inp = torch.FloatTensor(inp)
        ml_adj = loadmat('embedding/' + dataset + '-inp-googlenews_ml_adj.mat')['ml_adj'] 
        lml_adj = loadmat('embedding/' + dataset + '-inp-googlenews_lml_adj.mat')["lml_adj"]
        ml_adj= torch.FloatTensor(ml_adj)
        lml_adj = torch.FloatTensor(lml_adj)
    elif embedding == 'fasttext':
        inp = loadmat('embedding/' + dataset + '-inp-fasttext.mat')['inp']
        inp = torch.FloatTensor(inp)
        ml_adj = loadmat('embedding/' + dataset + '-inp-fasttext_ml_adj.mat')['ml_adj'] 
        lml_adj = loadmat('embedding/' + dataset + '-inp-fasttext_lml_adj.mat')["lml_adj"]
        ml_adj= torch.FloatTensor(ml_adj)
        lml_adj = torch.FloatTensor(lml_adj)
    else:
        inp = None

    print('...Data loading is beginning...')
    data_loader, input_data_par = get_loader(DATA_DIR, batch_size)
    print('...Data loading is completed...')

    model_ft = USSL(img_input_dim=input_data_par['img_dim'], text_input_dim=input_data_par['text_dim'],
                        num_classes=input_data_par['num_class'] , t=t,jieta=jieta, adj_file='data/' + dataset + '/adj.mat',
                        inp=inp,ml_adj=ml_adj,lml_adj=lml_adj,
                        n_layers=n_layers,num_groups=num_groups,ml_train=input_data_par['ml_train']).cuda()

    params_to_update = list(model_ft.parameters())
    optimizer = optim.Adam(params_to_update, lr=lr, betas=betas)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    if EVAL:
        model_ft.load_state_dict(torch.load('model/USSL' + dataset + '.pth'))  
    else:
        print('...Training is beginning...')
        # Train and evaluate
        model_ft, img_acc_hist, txt_acc_hist, loss_hist = train_model(model_ft, data_loader, optimizer,alpha1,alpha2,alpha3,alpha4,
                                                                        temp, gamma1,gamma2, max_epoch)
        print('...Training is completed...')

        torch.save(model_ft.state_dict(), 'model/USSL.pth')

    print('...Evaluation on testing data...')
    model_ft.load_state_dict(torch.load('model/USSL.pth'))
    model_ft.eval()          
    view1_feature, view2_feature, view1_predict, view2_predict, classifiers, x_ml, _, _, _ = model_ft(
        torch.tensor(input_data_par['img_test']).cuda(), torch.tensor(input_data_par['text_test']).cuda())
    label = input_data_par['label_test']
    ml_test_index=input_data_par['ml_train_index']  
    view1_feature = view1_feature.detach().cpu().numpy()
    view2_feature = view2_feature.detach().cpu().numpy()

    img_to_txt = fx_calc_map_label(view1_feature, view2_feature, label)
    print('...Image to Text MAP = {}'.format(img_to_txt))
    txt_to_img = fx_calc_map_label(view2_feature, view1_feature, label)
    print('...Text to Image MAP = {}'.format(txt_to_img))
    print('...Average MAP = {}'.format(((img_to_txt + txt_to_img) / 2.)))
    print('...num_groups = {}'.format(num_groups))
   






            