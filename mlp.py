
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, ReLU
import torch 
def build_sequential(in_dim,hidden_dims,out_dim,activation=None,
                    complex_params=False,output_activation=None,batch_norm=False,affine=True,dropout=None,**kwargs):
    if activation is None: activation='ReLU'
    layers=[nn.Linear(in_dim,hidden_dims[0])]
    for i in range(len(hidden_dims)-1):
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dims[i-1],affine=affine))
        if activation is not None:
            layers.append(getattr(nn,activation)())
        
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dims[i],hidden_dims[i+1]))
    if batch_norm:
        layers.append(nn.BatchNorm1d(hidden_dims[-1],affine=affine))
        
    if activation is not None:
        layers.append(getattr(nn,activation)())
    if dropout is not None:
        layers.append(nn.Dropout(dropout))
        
        
        
    layers.append(nn.Linear(hidden_dims[-1],out_dim))
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_dim,affine=affine))
    if output_activation is not None : 
        if output_activation==nn.Softmax:
            layers.append(output_activation(dim=-1))
        elif output_activation.lower() =='none' or output_activation.lower=='linear': pass 
        else:layers.append(getattr(nn,output_activation)()) 
    if 'dtype' in kwargs:
        dtype=kwargs.get('dtype')
        for i,item in enumerate(layers):
            print (i,item,item.__class__.__name__)
            if item.__class__.__name__=='Linear':
                layers[i]=item.type(dtype) 
    return Sequential(*layers)


MLP=build_sequential
        
        
        
        
        
        
