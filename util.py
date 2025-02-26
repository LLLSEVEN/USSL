import torch
from scipy.io import loadmat, savemat
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def gen_A(num_classes, t, adj_file):
    result = loadmat(adj_file)
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, None]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj.squeeze()
    _adj = _adj * 0.5
    _adj = _adj + np.identity(num_classes, np.float64)
    return _adj


def gen_adj(A):
    D = torch.pow(A.sum(axis=1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

def sim_ml_adj(ml_adj): 
    dim=ml_adj.shape[0] 
    ml_train=torch.tensor(ml_adj)
    A=torch.zeros(dim,dim)
    for i in range(dim):
        for j in range(dim):
            if i !=j :
                sim=torch.dot(ml_train[i],ml_train[j])  
                L1=ml_train[i].sum(dim=0)
                L2=ml_train[j].sum(dim=0)
                A[i][j]=sim/(L1+L2-sim)
    return A


def sim_l_ml_adj(ml_adj,num_classes):   
    dim=ml_adj.shape[0]
    ml_label=torch.tensor(ml_adj)  
    l_label=torch.eye(num_classes)  
    ml_label = ml_label.float()
    l_label = l_label.float()
    A=torch.zeros(num_classes,dim)
    for i in range(num_classes):
        for j in range(dim):
            if i !=j :
                sim=torch.dot(l_label[i],ml_label[j]) 
                L1=l_label[i].sum(dim=0)
                L2=ml_label[j].sum(dim=0)
                A[i][j]=sim/(L1+L2-sim)
    return A

def l_ml_adj(A,ml_adj,lml_adj): 

    A = A.to(device)
    lml_adj = lml_adj.to(device)
    ml_adj = ml_adj.to(device)

    R0 = torch.cat([A, lml_adj.T], dim=0)   
    R1 = torch.cat([lml_adj, ml_adj], dim=0) 
    R=torch.cat([R0, R1], dim=1)
 
    return R
              

def l_mlfeature(inp ,mlfeature):  
    dim1=inp.shape[0]  
    dim2=mlfeature.shape[0]  
    d=inp.shape[1]   
    A=torch.zeros(dim1+dim2,d)
    for i in range(dim1):
        A[i]=inp[i]
    for j in range(dim2):
        A[j+dim1]=mlfeature[j]
    return A

def finding(labels,Al_ml_adj,ml_train,x_ml):
    batch=labels.shape[0]  
    ml_dim=x_ml.shape[0] 
    f_dim=x_ml.shape[1]
    adj=Al_ml_adj[24:24+ml_dim,:]
    A=torch.zeros(batch,ml_dim+24)
    feature_ml=torch.zeros(batch,f_dim)

    labels=torch.tensor(labels)
    Al_ml_adj=torch.tensor(Al_ml_adj)
    ml_train=torch.tensor(ml_train)
    
    labels = labels.to(device)
    ml_train=ml_train.to(device)

    a=x_ml.shape[0]
    if a ==1024:
        x_ml=x_ml.T

    for i in range(batch):
        for j in range(ml_dim):
            if labels[i].equal(ml_train[j])==True:
                A[i]=adj[j]
                feature_ml[i]=x_ml[j]
    return A,feature_ml 

  
def pad_and_split_matrices(A, B, C, num_groups):  
    A = A.to(device)
    B = B.to(device)
    C = C.to(device)
    num_rows, num_cols_A = A.size()    
    _, num_cols_B = B.size()   
    num_rows_C, _ = C.size()  
    padded_rows = (num_rows + num_groups - 1) // num_groups * num_groups    

    padding = torch.zeros(padded_rows - num_rows, num_cols_A, dtype=A.dtype) 

    padding=padding.to(device)
    A_padded = torch.cat((A, padding), dim=0)  
    A_padded = torch.cat((A_padded, torch.zeros(padded_rows, padded_rows - num_cols_A).to(device)), dim=1)   
    B_padded = torch.cat((B, torch.randn(padded_rows - num_rows, num_cols_B).to(device)), dim=0)  
    C_padded = torch.cat((C, torch.zeros(num_rows_C, padded_rows - num_rows).to(device)), dim=1)  
    group_size = padded_rows // num_groups  
    A_groups = torch.zeros(num_groups, group_size, group_size, dtype=A.dtype)   
    B_groups = torch.zeros(num_groups, group_size, num_cols_B, dtype=B.dtype)  
    C_groups = torch.zeros(num_groups, num_rows_C, group_size, dtype=C.dtype)  

    for i in range(num_groups):  
        
        start_idx = i * group_size  
        end_idx = start_idx + group_size  
        A_groups[i] = A_padded[start_idx:end_idx, start_idx:end_idx]  
        B_groups[i] = B_padded[start_idx:end_idx]  
        C_groups[i] = C_padded[:, start_idx:end_idx]  

    return A_groups, B_groups, C_groups  


def cat_group_l_ml_adj(A,ml_adj,lml_adj):  
    part=ml_adj.shape[0]
    num_l=A.shape[0]  
    num_ml=ml_adj.shape[1]  
    A = A.to(device)
    ml_adj = ml_adj.to(device)
    lml_adj= lml_adj.to(device)
    adj=torch.rand(part,num_l+num_ml,num_l+num_ml)
    for i in range(part):
        R0 = torch.cat([A,lml_adj[i]],dim=1)   
        R1 = torch.cat([lml_adj[i].T,ml_adj[i]],dim=1)  
        adj[i]=torch.cat([R0, R1], dim=0) 
    return adj  

def  reduction(ml_groups_f,ml_groups_a,num_label,num_ml_labels): 
    num_groups=ml_groups_f.shape[0]
    num_ml_embedding_label=ml_groups_a.shape[1]-num_label  
    dim_l_label=ml_groups_f.shape[2] 
    cropped_adj_tensors = torch.empty(num_groups,num_ml_embedding_label,num_ml_embedding_label)
    cropped_feature_tensors = torch.empty(num_groups,num_ml_embedding_label,dim_l_label)
    final_adj=torch.empty(num_ml_embedding_label*num_groups,num_ml_embedding_label*num_groups)
    final_feature=torch.empty(num_ml_embedding_label*num_groups,dim_l_label)
    for i in range(num_groups):  
        cropped_adj_tensors[i] = ml_groups_a[i, num_label:ml_groups_a.shape[1],num_label:ml_groups_a.shape[1]] 
        cropped_feature_tensors[i] = ml_groups_f[i,num_label:ml_groups_a.shape[1], :]
        final_adj[i*num_ml_embedding_label:(i+1)*num_ml_embedding_label,i*num_ml_embedding_label:(i+1)*num_ml_embedding_label]=cropped_adj_tensors[i]
        final_feature[i*num_ml_embedding_label:(i+1)*num_ml_embedding_label]=cropped_feature_tensors[i]

    final_feature=final_feature[0:num_ml_labels,:]
    final_adj=final_adj[0:num_ml_labels,0:num_ml_labels]
    return final_feature,final_adj

def split_and_keep_diagonal_blocks(matrix, num_groups,l_ml_A):  
    rows, cols = matrix.size()  
    group_size = rows // num_groups  
    extra_rows = rows % num_groups  
    ml_A = torch.zeros_like(matrix) 
    ada_A= torch.zeros_like(matrix) 
    for i in range(num_groups):  
        start_row = i * group_size  
        end_row = start_row + group_size  
        if i == num_groups - 1 and extra_rows > 0:  
            end_row += extra_rows  
        ml_A[start_row:end_row, start_row:end_row] = matrix[start_row:end_row, start_row:end_row]  
        ada_A[start_row:end_row, :] =1
        ada_A=ada_A.T
        ada_A[start_row:end_row, :]=1
        ada_A=ada_A.T
    l_ml_A=torch.cat([ml_A.to(device),l_ml_A.to(device)],dim=1)
    return l_ml_A  ,ada_A

