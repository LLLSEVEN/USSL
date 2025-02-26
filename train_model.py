from __future__ import print_function
from __future__ import division
import time
import copy
from util import *
import numpy as np
import torch
import torch.nn as nn
import datetime
from evaluate import fx_calc_map_label
from loss import cla_loss, gan_loss, soft_con_loss,new_loss
                      



def train_model(model, data_loaders, optimizer,alpha1,alpha2,alpha3,alpha4,  temp, gamma1,gamma2, num_epochs=500):
    since = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_img_acc_history = []
    test_txt_acc_history = []
    epoch_loss_history = []

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 20)

        for phase in ['train', 'test']:
            if phase == 'train':

                model.train()
            else:

                model.eval()

            running_loss = 0.0
            for imgs, txts, labels,ml_labels in data_loaders[phase]:
                if torch.sum(imgs != imgs) > 1 or torch.sum(txts != txts) > 1:
                    print("Data contains Nan.")
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        txts = txts.cuda()
                        labels = labels.float().cuda()

                    view1_feature, view2_feature, view1_predict, view2_predict,x_l, x_ml,y_l,y_ml,Al_ml_adj= model(imgs, txts)
                  
                    labels = labels.detach()
                    Al_ml_adj = Al_ml_adj.detach()
                    ml_adj = Al_ml_adj[ml_labels]  
                    feature_ml=x_ml[ml_labels] 
                    c_loss = cla_loss(view1_predict, view2_predict, labels, labels)  
                    con_loss= soft_con_loss(view1_feature, view2_feature, labels,Al_ml_adj,x_ml,ml_adj,feature_ml,  temp, gamma1,gamma2)  
                    g_loss = gan_loss(y_l,y_ml) 
                    sample_mlabel_loss=new_loss(view1_feature, view2_feature,feature_ml)
                    loss = alpha1 * c_loss + alpha2*con_loss + alpha3* g_loss +alpha4*sample_mlabel_loss 
                    if torch.isnan(loss).item()==False:
                        if phase == 'train':
                            loss.backward(retain_graph=True)
                            optimizer.step()
                    
                if torch.isnan(loss).item()==False:
                    running_loss += loss.item()
            time2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            if phase == 'train':
                print(time2)
                print('Train Loss: {:.7f}'.format(epoch_loss))
            if phase == 'test':
                t_imgs, t_txts, t_labels = [], [], []
                with torch.no_grad():
                    for imgs, txts, labels,mltest_labels in data_loaders['test']:
                        if torch.cuda.is_available():
                            imgs = imgs.cuda()
                            txts = txts.cuda()
                            labels = labels.float().cuda()
                        t_view1_feature, t_view2_feature, _, _, _,  _, _, _, _, = model(imgs, txts)
                        t_imgs.append(t_view1_feature.cpu().numpy())
                        t_txts.append(t_view2_feature.cpu().numpy())
                        t_labels.append(labels.cpu().numpy())
                t_imgs = np.concatenate(t_imgs)
                t_txts = np.concatenate(t_txts)
                t_labels = np.concatenate(t_labels)

                img2text = fx_calc_map_label(t_imgs, t_txts, t_labels)
                txt2img = fx_calc_map_label(t_txts, t_imgs, t_labels)
                print('{} Loss: {:.7f} Img2Txt: {:.4f}  Txt2Img: {:.4f}'.format(phase, epoch_loss, img2text, txt2img))

            if phase == 'test' and (img2text + txt2img) / 2. > best_acc:
                best_acc = (img2text + txt2img) / 2.
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'test':
                test_img_acc_history.append(img2text)
                test_txt_acc_history.append(txt2img)
                epoch_loss_history.append(epoch_loss)
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best average ACC: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, test_img_acc_history, test_txt_acc_history, epoch_loss_history

