#! /usr/bin/env python

from argparse import ArgumentParser
parser=ArgumentParser(description='select options to train quantum autoencoder')
parser.add_argument('--latent-scale-factor',default=None,type=float,help='scale factor for latent variables')
parser.add_argument('--train',default=False,action='store_true',help='train network!')
parser.add_argument('--status-bar',default=False,action='store_true') 
parser.add_argument('--scatter',default=False,action='store_true')
parser.add_argument('--topology',default='S2xS2',type=str) 
parser.add_argument('--no-decay-on-plateau',dest='decay_on_plateau',default=True,action='store_false') 
parser.add_argument('--data',default='parton_topW',type=str,
                    help='function name for generating pseudodata, should be in the global namespace of the module') 
parser.add_argument('--num-samples',default=500_000,type=int) 
parser.add_argument('--num-runs',default=1,type=int) 
parser.add_argument('-b','--batch-size',default=256,type=int)
parser.add_argument('-e','--epochs',default=300,type=int)
parser.add_argument('--encoder-dims',type=int,nargs='+',default=[1024,512,256,128,64]) 
parser.add_argument('--decoder-dims',type=int,nargs='+',default=None) 
parser.add_argument('--lr',default=0.001,type=float,help='learning rate of optimizer during training')
parser.add_argument('--save',default=False,action='store_true')
parser.add_argument('--save-dir',default='network_runs',
                  help='directory to save, creates <tag>_# folder and saves it there')
parser.add_argument('--test',default=False,action='store_true')
parser.add_argument('--path',default='',type=str,help='required for test, path to run directory e.g ./network_runs/run_1')
parser.add_argument('--run-tag',default='run',type=str,help='tag for run, will create <run_tag>_# directory ar <save_dir>')
parser.add_argument('--no-gpu',dest='gpu',default=True,action='store_false') 
parser.add_argument('--cuda-id',type=int,default=0) 
parser.add_argument('--latent-activation',type=str,default='Tanh') 
parser.add_argument('--decoder-activation',type=str,default='ReLU') 
parser.add_argument('--latent-dim', type=int, default=6) 
parser.add_argument('--no-early-stop',dest='early_stop',default=True,action='store_false')
parser.add_argument('--tolerance',default=20,type=int) 
parser.add_argument('--quantum',dest='classical_head',default=True,action='store_false') 
parser.add_argument('--trivial-latent-topology',default=False,action='store_true',
                    help='assume $\mathbb{R}^l$ topology in usual autoencoders without a topological head') 
parser.add_argument('--signal-data',default=[],metavar='N',
                    type=str,nargs='+',help='Additional signal data for testing anomalous signal') 
args=parser.parse_args()

import matplotlib.pyplot as plt
try:
    plt.style.use('mystyle.mplstyle')
except OSError:
    pass 
from utils import check_dir,print_events,Pickle,Unpickle,set_up_run_folder,nice_verbose_args
import data_loader
from data_loader import seeds
import os,sys
prefix=os.getcwd()
import numpy as np 
if args.latent_scale_factor is None: args.latent_scale_factor = np.pi
import torch
from torch_hybrid_autoenc import HybridAutoEncoder
import pennylane as qml
from trainers import supervised_train,classifier_test
from itertools import combinations
from torch.utils.data import DataLoader as torch_dataloader
from deep_hep.plotter import Plotter
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as skm
import matplotlib.pyplot as plt 
import warnings
from math import comb

hierarchy=['data','topology'] if not args.trivial_latent_topology else ['data','trivial_latent_top']

latent_dict={"S2xR2":5,"RP2xR2":5,"S2":3,'S2xS2':6,'S5':6,'S4':5,'RP2':3,'RP2xRP2':6,'S2xRP2':6}     
if not args.trivial_latent_topology:
    args.latent_dim=latent_dict[args.topology]   

if args.gpu:                                                                                      #
    device = torch.device("cuda:{}".format(args.cuda_id) if torch.cuda.is_available() else "cpu")  #
else:
    device=torch.device("cpu")

def log_mse(inf_dict):
    return np.log10(np.mean((inf_dict['true_labels']-inf_dict['predicted'])**2,axis=1))

def test(current_args):
    if current_args.path =='':
        with open('last_path.txt','r') as f:
            current_args.path=f.readlines()[0] 
    args=Unpickle('args',path=current_args.path) 
    added_keys=[]
    for key in vars(current_args):
        if key not in vars(args):
            setattr(args,key,getattr(current_args,key))
            added_keys.append(key)
    
    if len(added_keys)>0:
        print ('Added keys for argument namespace compatibility: ',added_keys)
        warnings.warn('Ensure default value in current arguments do not change the behaviour of old initialsation variables!')  
    
    
    _,background_data=getattr(data_loader,args.data)(args.num_samples,random_seed=seeds.get(args.data,None),
                                                     ) 

    autoenc=HybridAutoEncoder(background_data[0][0].shape[0],args.encoder_dims,args.latent_dim,
                              latent_scale_factor=args.latent_scale_factor,
                              latent_activation=args.latent_activation,
                              topology=args.topology,
                           trivial_latent_topology=args.trivial_latent_topology,
                          without_decoder=False,decoder_dims=args.decoder_dims,decoder_activation=args.decoder_activation,
                         ) 
    
    assert 'best_model.pkl' in os.listdir(current_args.path)
    model_path=os.path.join(current_args.path,'best_model.pkl') 
    autoenc.load_state_dict(torch.load(model_path,map_location=device))
    background_inf,_=classifier_test(autoenc,background_data,status_bar=True,device=device,
                 batch_size=args.batch_size,graph_train=False,
                 regression=True)
    
    inference_dictionary={args.data+'(BG)':background_inf}
    inference_dictionary['topo_name']=autoenc.topo_name
    inference_dictionary['encoder_dims']=tuple(args.encoder_dims)
    bg_mse=log_mse(background_inf) 
    
    inference_dictionary['log_mse_'+args.data]=bg_mse
    print (bg_mse.shape,bg_mse)
    print (current_args.signal_data) 
    if args.data in current_args.signal_data:
        current_args.signal_data.remove(args.data) 
    if current_args.signal_data is not None:
        for signal in current_args.signal_data:
            _,data=getattr(data_loader,signal)(args.num_samples,random_seed=seeds.get(signal,None),background=args.data,
                                               )    
            signal_inf,_=classifier_test(autoenc,data,status_bar=True,device=device,
                 batch_size=args.batch_size,graph_train=False,
                 regression=True)
            print_events(signal_inf,signal) 
            inference_dictionary[signal]=signal_inf
            signal_mse=log_mse(signal_inf) 
            inference_dictionary['log_mse_'+signal]=signal_mse
            print (signal,signal_mse)
            y_true=np.concatenate((np.zeros(len(bg_mse)),np.ones(len(signal_mse))))
            y_pred=np.expand_dims(np.concatenate((bg_mse,signal_mse)),1)
            scaler = MinMaxScaler()
            scaler.fit(y_pred)
            y_pred=scaler.transform(y_pred)
            
            auc=skm.roc_auc_score(y_true,y_pred.squeeze())
            inference_dictionary[f'auc_{signal}']=auc
            print ('AUC:',auc) 
    
    inference_dictionary['model_path']=model_path
    print_events(inference_dictionary) 
    Pickle(inference_dictionary,'autoenc_predict',path=current_args.path) 





def train(args):
    nice_verbose_args(args) 
    train_data,val_data=getattr(data_loader,args.data)(args.num_samples,random_seed=seeds.get(args.data,None),
                                )  
    if args.save:
        args.save_dir=check_dir(os.path.join(prefix,args.save_dir))
        args.save_dir=set_up_run_folder(args,hierarchy,parent_path=args.save_dir) 
        all_dirs=[item for item in os.listdir(args.save_dir) if os.path.isdir(os.path.join(args.save_dir,item))]
        if args.num_runs>1:
            args.save_dir=check_dir(os.path.join(args.save_dir,'multi_'+args.run_tag+f'_{len(all_dirs)+1}'))
        save_prefix=args.save_dir
    for _ in range(args.num_runs):
       
        autoenc=HybridAutoEncoder(train_data[0][0].shape[0],args.encoder_dims,
                           args.latent_dim,latent_scale_factor=args.latent_scale_factor,
                           latent_activation=args.latent_activation,
                           trivial_latent_topology=args.trivial_latent_topology, topology=args.topology,
                          without_decoder=False,decoder_dims=args.decoder_dims,decoder_activation=args.decoder_activation,
                          )          
        if args.save:
            all_dirs=[item for item in os.listdir(save_prefix) if os.path.isdir(os.path.join(save_prefix,item))]
            args.save_dir=check_dir(os.path.join(save_prefix,f'run_{len(all_dirs)+1}'))
            Pickle(args,'args',path=args.save_dir) 
            args_repr=nice_verbose_args(args,return_as_string=True) 
            num_params=sum(p.numel() for p in autoenc.parameters() if p.requires_grad)
            with open(os.path.join(args.save_dir,'args.txt'),'w+') as f:
                f.write(args_repr) 
            with open(os.path.join(args.save_dir,'model_repr.txt'),'w+') as f:
                f.write(repr(autoenc))
                f.write('\nNum params:'+str(num_params))
            with open('last_path.txt','w+') as f:
                f.write(args.save_dir) 
        supervised_train(autoenc,train_data,val_data,graph_train=False,save_dir=args.save_dir,device=device,
                 batch_size=args.batch_size,epochs=args.epochs,lr=args.lr,decay_on_plateau=args.decay_on_plateau,
                 tolerance=args.tolerance if args.early_stop else None,plot_logscale=True,
                 regression=True,status_bar=args.status_bar,save=args.save ) 
if __name__=='__main__':
    if args.train:
        train(args) 
    elif args.test:
        test(args) 
        







