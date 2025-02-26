import torch
import torch.nn as nn
import torch.nn.functional as F
from util import *
criterion_md = nn.CrossEntropyLoss()


def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t())
    return Sim


def cla_loss(view1_predict, view2_predict, labels_1, labels_2):  
    cla_loss1 = ((view1_predict - labels_1.float()) ** 2).sum(1).sqrt().mean()
    cla_loss2 = ((view2_predict - labels_2.float()) ** 2).sum(1).sqrt().mean()


    return cla_loss1 + cla_loss2

def new_loss(view1_feature, view2_feature,feature_ml):   
    ml_loss = ((view1_feature - feature_ml) ** 2).sum(1).sqrt().mean()+((view2_feature - feature_ml) ** 2).sum(1).sqrt().mean()
    batch=view1_feature.shape[0]
    loss=ml_loss/batch
    return loss

def gan_loss(y_l,y_ml):  
    bs_l = y_l.size()[0]  
    bs_ml=y_ml.size()[0]   
    l_md = torch.ones(bs_l, dtype=torch.long).cuda()
    ml_md = torch.zeros(bs_ml, dtype=torch.long).cuda()
    return criterion_md(y_l, l_md) + criterion_md(y_ml, ml_md)     


def soft_con_loss(view1_feature, view2_feature, labels,Al_ml_adj,x_ml,ml_adj,feature_ml, t, gamma1,gamma2):  
    view1_feature = F.normalize(view1_feature, dim=1)
    view2_feature = F.normalize(view2_feature, dim=1)
    adj=Al_ml_adj
    feature_ml = F.normalize(feature_ml, dim=1)
    feature_ml=feature_ml.to(device)
    adj = F.normalize(adj, dim=1)
    x_ml = F.normalize(x_ml, dim=1)
    ml_adj = F.normalize(ml_adj, dim=1) 
    adj=adj.to(device)
    ml_adj=ml_adj.to(device)
    den_sim_img=torch.matmul(view1_feature,feature_ml.T) 
    den_sim_img=torch.exp(torch.diag(den_sim_img)/t)   
    w_vm=torch.matmul(ml_adj,adj.T) 
    nume_sim_img=torch.matmul(view1_feature,x_ml.T)   
    nume_img=torch.exp(torch.mul(w_vm,nume_sim_img)/t)
    nume_img=nume_img.sum(dim=1,keepdim=False)-den_sim_img
    iso_img=torch.mean(-torch.log(torch.div(den_sim_img,nume_img)))  
    den_sim_txt=torch.matmul(view2_feature,feature_ml.T) 
    den_sim_txt=torch.exp(torch.diag(den_sim_txt)/t)    
    w_tm=torch.matmul(ml_adj,adj.T)  
    nume_sim_txt=torch.matmul(view2_feature,x_ml.T)  
    nume_txt=torch.exp(torch.mul(w_tm,nume_sim_txt)/t)
    nume_txt=nume_txt.sum(dim=1,keepdim=False)-den_sim_txt
    iso_txt=torch.mean(-torch.log(torch.div(den_sim_txt,nume_txt)))  

    den_sim_label_img=torch.matmul(feature_ml,view1_feature.T) 
    den_sim_label_img=torch.exp(torch.diag(den_sim_label_img)/t)   
    w_mv=torch.matmul(adj,ml_adj.T)  
    nume_sim_ml_image=torch.matmul(x_ml,view1_feature.T)   
    nume_ml_img=torch.exp(torch.mul(w_mv,nume_sim_ml_image)/t)

    nume_ml_img=nume_ml_img.sum(dim=0,keepdim=False)-den_sim_label_img
    iso_ml_img=torch.mean(-torch.log(torch.div(den_sim_label_img,nume_ml_img)))  

    den_sim_label_txt=torch.matmul(feature_ml,view2_feature.T) 
    den_sim_label_txt=torch.exp(torch.diag(den_sim_label_txt)/t)   
    w_mt=torch.matmul(adj,ml_adj.T)  
    nume_sim_ml_text=torch.matmul(x_ml,view2_feature.T)  
    nume_ml_txt=torch.exp(torch.mul(w_mt,nume_sim_ml_text)/t)

    nume_ml_txt=nume_ml_txt.sum(dim=0,keepdim=False)-den_sim_label_txt
    iso_ml_txt=torch.mean(-torch.log(torch.div(den_sim_label_txt,nume_ml_txt)))  


    ISO_loss=(iso_img+iso_txt+iso_ml_img+iso_ml_txt)*0.01


    batch=view1_feature.shape[0]
    l_adj=torch.zeros(batch,batch)
    l_adj=l_adj.to(device)
    for i in range(batch):
            for j in range(batch):
                if    labels[i].equal(labels[j])==True:
                    l_adj[i][j]=1
                

    den_sim_img_img=torch.exp(torch.matmul(view1_feature,view1_feature.T)/t)  
    w_vv=torch.matmul(ml_adj,ml_adj.T)  
    nume_sim_img_img=torch.matmul(view1_feature,view1_feature.T)  
    nume_img_img=torch.exp(torch.mul(w_vv,nume_sim_img_img)/t)    
    nume_img_img=nume_img_img.sum(dim=1,keepdim=False)

    nume_img_img=nume_img_img.repeat(batch,1).T  
    l_adj=l_adj-torch.diag_embed(torch.diag(l_adj))  
    n_img_img=l_adj.sum(dim=1,keepdim=False).clamp(min=1.0) 
    n_img_img1=l_adj.sum(dim=1,keepdim=False).clamp(max=1.0)
    n_img_img1=n_img_img1.sum(dim=0,keepdim=False).clamp(min=1.0)  
    intra_image_loss=-torch.log(torch.div(den_sim_img_img,(nume_img_img-den_sim_img_img)))
    intra_image_loss=torch.mul(l_adj,intra_image_loss)  
    intra_image_loss=torch.div(intra_image_loss.sum(dim=1,keepdim=False) ,n_img_img) 
    intra_image=0.01*(intra_image_loss.sum(dim=0,keepdim=False)/n_img_img1)



    den_sim_txt_txt=torch.exp(torch.matmul(view2_feature,view2_feature.T)/t)  
    w_tt=torch.matmul(ml_adj,ml_adj.T)  
    nume_sim_txt_txt=torch.matmul(view2_feature,view2_feature.T)  
    nume_txt_txt=torch.exp(torch.mul(w_tt,nume_sim_txt_txt)/t)   
    nume_txt_txt=nume_txt_txt.sum(dim=1,keepdim=False) 

    nume_txt_txt=nume_txt_txt.repeat(batch,1).T  
    l_adj=l_adj-torch.diag_embed(torch.diag(l_adj))  
    n_txt_txt=l_adj.sum(dim=1,keepdim=False).clamp(min=1.0) 
    n_txt_txt1=l_adj.sum(dim=1,keepdim=False).clamp(max=1.0)
    n_txt_txt1=n_txt_txt1.sum(dim=0,keepdim=False).clamp(min=1.0) 
    intra_txt_loss=-torch.log(torch.div(den_sim_txt_txt,(nume_txt_txt-den_sim_txt_txt)))
    intra_txt_loss=torch.mul(l_adj,intra_txt_loss)  
    intra_txt_loss=torch.div(intra_txt_loss.sum(dim=1,keepdim=False) ,n_txt_txt) 
    intra_txt=0.01*(intra_txt_loss.sum(dim=0,keepdim=False)/n_txt_txt1)

    den_sim_img_txt=torch.exp(torch.matmul(view1_feature,view2_feature.T)/t)  
    w_vt=torch.matmul(ml_adj,ml_adj.T)  
    nume_sim_img_txt=torch.matmul(view1_feature,view2_feature.T)  
    nume_img_txt=torch.exp(torch.mul(w_vt,nume_sim_img_txt)/t)    
    nume_img_txt=nume_img_txt.sum(dim=1,keepdim=False) 
    nume_img_txt=nume_img_txt.repeat(batch,1).T  
    n_img_txt=l_adj.sum(dim=1,keepdim=False).clamp(min=1.0) 
    n_img_txt1=l_adj.sum(dim=1,keepdim=False).clamp(max=1.0)
    n_img_txt1=n_img_txt1.sum(dim=0,keepdim=False).clamp(min=1.0)  
    inter_image_text_loss=-torch.log(torch.div(den_sim_img_txt,(nume_img_txt-den_sim_img_txt)))
    inter_image_text_loss=torch.mul(l_adj,inter_image_text_loss)  
    inter_image_text_loss=torch.div(inter_image_text_loss.sum(dim=1,keepdim=False) ,n_img_txt) 
    inter_image_text=0.01*(inter_image_text_loss.sum(dim=0,keepdim=False)/n_img_txt1)
    den_sim_txt_img=torch.exp(torch.matmul(view2_feature,view1_feature.T)/t)  
    w_tv=torch.matmul(ml_adj,ml_adj.T) 
    nume_sim_txt_img=torch.matmul(view2_feature,view1_feature.T)  
    nume_txt_img=torch.exp(torch.mul(w_tv,nume_sim_txt_img)/t)    
    nume_txt_img=nume_txt_img.sum(dim=1,keepdim=False) 

    nume_txt_img=nume_txt_img.repeat(batch,1).T  
    n_txt_img=l_adj.sum(dim=1,keepdim=False).clamp(min=1.0) 
    n_txt_img1=l_adj.sum(dim=1,keepdim=False).clamp(max=1.0)
    n_txt_img1=n_txt_img1.sum(dim=0,keepdim=False).clamp(min=1.0) 
    if n_txt_img1==0:
        n_txt_img1=1
    inter_text_image_loss=-torch.log(torch.div(den_sim_txt_img,(nume_txt_img-den_sim_txt_img)))
    inter_text_image_loss=torch.mul(l_adj,inter_text_image_loss)  
    inter_text_image_loss=torch.div(inter_text_image_loss.sum(dim=1,keepdim=False) ,n_txt_img) 
    inter_text_image=0.01*(inter_text_image_loss.sum(dim=0,keepdim=False)/n_txt_img1)

    loss =gamma1*(intra_image+intra_txt+inter_image_text+inter_text_image)+  gamma2*ISO_loss

    return loss









