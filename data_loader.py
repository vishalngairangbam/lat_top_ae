import torch
from torch.utils.data import TensorDataset 
import numpy as np
import pandas as pd 
from scipy.stats import uniform_direction
import sys , os 
import vector
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from utils import S2,rotational_embedding
from itertools import combinations
from utils import print_events,Unpickle,nested_to_vector,read_dataframe
from utils import convert_to_vector as convert_to_vector_parton
seeds={'sphere':9283923718,'distorted_sphere':18292730,
        'planes':38272919,'sphere2x2':28273817,'embed_9D_sphere2x2':28739109,
        'embed_9D_S5':3748298,
        'hypersphere':6718278,'top':None,'rpvg':None,
       'distorted_torus':82937192,'torus':27464710}







def datasetdecorator(geometry_generator):
    def wrapper(num_points,random_seed=None,**kwargs):  
        print ('Dataset func:',geometry_generator,'Random seed:',random_seed,'Num points:',num_points) 
        
        if random_seed is not None: 
            rng=torch.Generator()
            rng.manual_seed(random_seed)
        else: rng=None
        toy_data = geometry_generator(rng,num_points,**kwargs)[:num_points]    

        val_frac=0.2
        val_split=int(round((1-val_frac)*len(toy_data),0)) 
        train_data=toy_data[:val_split]
        val_data=toy_data[val_split:]
        train_data=TensorDataset(train_data,train_data)
        val_data=TensorDataset(val_data,val_data) 
        print ('Train size:',len(train_data),'Validation size:',len(val_data))
        return train_data,val_data
    return wrapper

@datasetdecorator
def distorted_sphere(rng, num_points,**kwargs):
    coor,th,phi=S2(num_points,generator=rng,return_angles=True) 
    r = 2 + torch.sin(3*th)/4 + torch.sin(5*phi)/4 
    toy_data= r.unsqueeze(1)*coor
    return toy_data


@datasetdecorator
def parton_topW(rng,num_points,background=None,**kwargs):
    array= _load_parton(filename='semilep_ttx_wnlv_seq')
    if background is None or background == 'parton_topW':
        scale_array=None
    else:
        assert background=='parton_top4f'
        scale_array=_load_parton(filename='semilep_ttx_wnlv_seq_t4f')
    #print (array,scale_array)
    array=_scale(array,background_features=scale_array)
    return array
@datasetdecorator
def parton_top4f(rng,num_points,background='parton_topW',**kwargs):
    array= _load_parton(filename='semilep_ttx_wnlv_seq_t4f')
    if background is None or background == 'parton_top4f':
        scale_array=None
    else:
        assert background=='parton_topW'
        scale_array=_load_parton(filename='semilep_ttx_wnlv_seq')
    array=_scale(array,background_features=scale_array)
    return array 
def _load_parton(filename):
    events=read_dataframe(os.path.join("data",filename+'.h5'),"parton_level")
    print_events(events)
    w=events['quarks']
    b=events['bquark-top']
    top_candidate=b+w.sum(1)
    w[:,0]=w[:,0].boostCM_of(top_candidate)
    w[:,1]=w[:,1].boostCM_of(top_candidate)
    b=b.boostCM_of(top_candidate)
    
    
    features=np.vstack([b.px,b.py,b.pz,w[:,0].px,w[:,0].py,w[:,0].pz,w[:,1].px,w[:,1].py,w[:,1].pz]).swapaxes(0,1)
    return features






def _load_reco(filename):
    events=read_dataframe(os.path.join("data",filename+'.h5'),"reco_level")
    print_events(events)
    
    w=events['jets']
    b=events['bjet-top']
    top_candidate=b+w.sum(1)
    w[:,0]=w[:,0].boostCM_of(top_candidate)
    w[:,1]=w[:,1].boostCM_of(top_candidate)
    b=b.boostCM_of(top_candidate)
    
    
    features=np.vstack([b.px,b.py,b.pz,w[:,0].px,w[:,0].py,w[:,0].pz,w[:,1].px,w[:,1].py,w[:,1].pz]).swapaxes(0,1)
    return features
    
@datasetdecorator
def reco_topW(rng,num_points,background=None,**kwargs):
    array= _load_reco(filename='semilep_ttx_wnlv_seq') 
    if background is None or background == 'reco_topW':
        scale_array=None
    else:
        assert background=='reco_top4f'
        scale_array=_load_reco(filename='semilep_ttx_wnlv_seq_t4f')
    array=_scale(array,background_features=scale_array)
    return array

@datasetdecorator
def reco_top4f(rng,num_points,background=None,**kwargs):
    array= _load_reco(filename='semilep_ttx_wnlv_seq_t4f') 
    if background is None or background == 'reco_top4f':
        scale_array=None
    else:
        assert background=='reco_topW'
        scale_array=_load_reco(filename='semilep_ttx_wnlv_seq')
    array=_scale(array,background_features=scale_array)
    return array
    
def embed_9D(array,rng,embedding_angles):
    rotation_axes=[(0,6),(1,7),(2,8),(3,8),(4,7),(5,6)]
    if embedding_angles=='single':
        size=1
    elif embedding_angles=='per_axis':
        size=(len(rotation_axes))
    elif embedding_angles=='per_sample':
        size=len(array)
    elif embedding_angles=='per_sample_per_axis':
        size=(len(rotation_axes),len(array))
    angles=torch.rand(size,generator=rng)*2*np.pi
    print (angles,array.shape,array)
    
    array=rotational_embedding(array,rotation_axes,angles)
    return array 
@datasetdecorator
def embed_9D_sphere2x2(rng,num_points,embedding_angles='per_axis',**kwargs):
    toy_data1=S2(num_points,generator=rng)
    toy_data2=S2(num_points,generator=rng)
    embed_9d=torch.cat((toy_data1, toy_data2,torch.zeros((len(toy_data2),3),dtype=torch.float)),dim=1)
    embed_9d=embed_9D(embed_9d,rng,embedding_angles)
    return embed_9d


@datasetdecorator
def sphere(rng, num_points,**kwargs):     
    toy_data=S2(num_points,generator=rng)   
    return toy_data
@datasetdecorator
def sphere2x2(rng, num_points,**kwargs):     
    toy_data1=S2(num_points,generator=rng)
    toy_data2=S2(num_points,generator=rng)
    toy_data = torch.cat((toy_data1, toy_data2),dim=1)  
    return toy_data


def _scale(features,background_features=None):
    if background_features is not None:
        scale_features=background_features
    else: 
        scale_features=features
    scaler = StandardScaler()
    scaler.fit(scale_features)
    
    features=scaler.transform(features)
    return torch.tensor(features,dtype=torch.float) 



