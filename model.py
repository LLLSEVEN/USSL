import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import Parameter
from util import *


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input=input.to(device)
        support = torch.matmul(input.to(device), self.weight.to(device))
        # print(input.shape)
        # print(adj.shape)
        # print(support.shape)
        output = torch.matmul(adj.to(device), support.to(device))
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class ImgNN(nn.Module):
    """Network to learn image representations"""

    def __init__(self, input_dim=4096, output_dim=1024):
        super(ImgNN, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.bn(x)
        out = F.relu(self.denseL1(x))
        return out

class TextNN(nn.Module):
    """Network to learn text representations"""

    def __init__(self, input_dim=1024, output_dim=1024):
        super(TextNN, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.bn(x)
        out = F.relu(self.denseL1(x))
        return out

class ReverseLayerF(Function):   
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class ModalClassifier(nn.Module): 
    """Network to discriminate modalities"""

    def __init__(self, input_dim=40):
        super(ModalClassifier, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.denseL1 = nn.Linear(input_dim, input_dim // 4)
        self.denseL2 = nn.Linear(input_dim // 4, input_dim // 16)
        self.denseL3 = nn.Linear(input_dim // 16, 2)

    def forward(self, x):
        x = ReverseLayerF.apply(x, 1.0)
        x = self.bn(x)
        out = F.relu(self.denseL1(x),inplace=False)
        out = F.relu(self.denseL2(out),inplace=False)
        out = self.denseL3(out)
        return out



class USSL(nn.Module):  
    def __init__(self, img_input_dim=4096, text_input_dim=1024, minus_one_dim=1024, num_classes=10, in_channel=300, t=0,jieta=0.5,
                 adj_file=None, inp=None,ml_adj=None,lml_adj=None, n_layers=4,num_groups=10,ml_train=None):
        super(USSL, self).__init__()
        self.img_net = ImgNN(img_input_dim, minus_one_dim)
        self.text_net = TextNN(text_input_dim, minus_one_dim)
        self.num_classes = num_classes
        self.ml_adj=ml_adj  
        self.lml_adj=lml_adj  
        self.num_groups=num_groups
        self.inp=inp
        self.jieta=jieta
        self.ml_train=torch.tensor(ml_train).to(torch.float32) 
        self.gnn = GraphConvolution
        self.n_layers = n_layers

        self.l_num=self.lml_adj.shape[0]   
        self.ml_num=self.ml_adj.shape[0]  
        self.inp_dim=inp.shape[1]  
        self.relu = nn.ReLU(inplace=False)
        self.lrn = self.gnn(in_channel, 300) 

        self.hypo = nn.Linear(300, minus_one_dim)  
        self.l_D_net = ModalClassifier(minus_one_dim)   
        self.ml_D_net = ModalClassifier(minus_one_dim)  

        self.W_l=Parameter(torch.Tensor(self.l_num,self.inp_dim))  

        self._adj = torch.FloatTensor(gen_A(num_classes, t, adj_file))  
        self.l_ada = Parameter(0.02 * torch.rand(self.l_num,self.l_num))
        self.ml_ada=Parameter(0.02 * torch.rand(self.ml_num,self.ml_num)) 
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature_img, feature_text):
        view1_feature = self.img_net(feature_img)  
        view2_feature = self.text_net(feature_text)
        lael_ada=self.relu(self._adj.to(device)+self.beta * self.l_ada.to(device)) 
        ml_lael_ada=self.relu(self.ml_adj.to(device)+self.jieta * self.ml_ada.to(device)) 
        inp_new=self.inp.cpu()*self.W_l.cpu()
        inp_ml=torch.mm(self.ml_train,inp_new)
        l_feature=torch.cat([self.inp,inp_ml],dim=0) 
        l_ml_adj=torch.cat([lael_ada,self.lml_adj.to(device)],dim=1)   
        paddings_matrix=torch.ones(self.ml_num,self.ml_num+self.l_num)  
        label_adj=gen_adj(torch.cat([l_ml_adj,paddings_matrix.to(device)],dim=0)) 
        label_A =label_adj[0:self.num_classes,:] 
        layers=[]
        final_l_feature = self.lrn(l_feature, label_A)
        layers.append(final_l_feature)
        group_ml_adj,group_ml_ada=split_and_keep_diagonal_blocks(ml_lael_ada,self.num_groups,self.lml_adj.T)  
        group_ml_feature=torch.cat([inp_ml.to(device),self.inp.to(device)], dim=0)
        group_ml_ada=torch.cat([group_ml_ada.to(device),torch.ones(self.ml_num,self.l_num).to(device)],dim=1) 
        paddings_matrix1=torch.rand(self.l_num,self.ml_num+self.l_num)  
        ml_label_adj=gen_adj(torch.cat([group_ml_adj.to(device),paddings_matrix1.to(device)],dim=0)) 
        ml_label_A =ml_label_adj[0:self.ml_num,:]
        final_ml_feature = self.lrn(group_ml_feature, ml_label_A)
        layers.append(final_ml_feature)

        final_ml_adj=ml_label_A[:,0:self.ml_num]
        final_linear_ml_feature = self.hypo(final_ml_feature.to(device))  
        final_linear_l_feature=self.hypo(final_l_feature.to(device))  
        norm_img = torch.norm(view1_feature, dim=1)[:, None] * torch.norm(final_linear_l_feature, dim=1)[None, :] + 1e-6  
        norm_txt = torch.norm(view2_feature, dim=1)[:, None] * torch.norm(final_linear_l_feature, dim=1)[None, :] + 1e-6 

        final_linear_l_feature = final_linear_l_feature.transpose(0, 1) 
        y_img = torch.matmul(view1_feature, final_linear_l_feature) 
        y_text = torch.matmul(view2_feature, final_linear_l_feature)
        y_img = y_img / norm_img  
        y_text = y_text / norm_txt  
        final_linear_l_feature=final_linear_l_feature.T
        y_l = self.l_D_net(final_linear_l_feature)  
        y_ml = self.ml_D_net(final_linear_ml_feature)        
        y_l = F.softmax(y_l, dim=1)  
        y_ml = F.softmax(y_ml, dim=1)
   
        return view1_feature, view2_feature, y_img, y_text, final_linear_l_feature, final_linear_ml_feature,y_l,y_ml,final_ml_adj
    
  