
try:
    from torch_geometric.loader import DataLoader as pyg_dataloader
except ModuleNotFoundError:
    print ('torch_geometric not found, needed for graph dataset training...')
    pyg_dataloader=None
from torch.utils.data import DataLoader as torch_dataloader
from tqdm import tqdm
import numpy as np
import sys,os

import torch 
from trainer_classes import SupervisedTrainer
from utils import print_events,Unpickle,nice_verbose_args,check_dir
irc_models=None


import warnings
from time import time
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
def supervised_train(model,train_data,val_data,
     save_dir='network_runs',epochs=10,early_stop_epoch=30,ddp=False,rank=0,status_bar=False,
                     save=False,debug=False,batch_size=128,run_tag='run_',device=None,
                     collate_fn=None,
                     **trainer_kwargs
                     ): 
    graph_train = trainer_kwargs.get('graph_train',True)    
    if not graph_train or collate_fn is not None:
        dataloader_cls=torch_dataloader
    else:
        dataloader_cls=pyg_dataloader
    train_loader=dataloader_cls(train_data,shuffle=not ddp,batch_size=batch_size,
                                collate_fn=collate_fn,sampler=DistributedSampler(train_data) if ddp else None)
    val_loader=dataloader_cls(val_data,shuffle=False,batch_size=batch_size,
                              collate_fn=collate_fn,sampler=DistributedSampler(val_data) if ddp else None)
    if device is None:
        device=torch.device('cpu')     
    decay_on_plateau=trainer_kwargs.get('decay_on_plateau',False)
    model.to(device)
    if ddp:
        model=DDP(model)  
    if save: 
        assert os.path.exists(save_dir),f"Could not find save path {save_dir}, make sure it exists..."                       
    trainer=SupervisedTrainer(model,rank=rank,**trainer_kwargs)
    train_losses, val_losses = [], []
    if rank==0:
        print('Training Start')
    end_flag=torch.zeros(1).to(device) 
    try:
        for epoch in range(epochs+1):
            epoch_start=time() 
            if epoch !=0:
                train_loss=0.
                step=0
                model.train() 
                if ddp:
                    train_loader.sampler.set_epoch(epoch)
                if rank==0  and status_bar: it=tqdm(train_loader)
                else: it=train_loader
                for batch in it:
                    if type(batch)==list:
                        batch[0],batch[1]=batch[0].to(device),batch[1].to(device) 
                    else:
                        batch.to(device)
                    train_loss += trainer.iteration(batch,start_epoch=step==0)
            
            
                    step+=1
        
            
            else:
                train_loss = float('nan') 
            train_loss /= len(train_loader)
            #train_losses.append(train_loss)
            model.eval()
            val_loss=0.
            if rank==0 and status_bar: 
                print ('Validating...')
                it=tqdm(val_loader)
                
            else: it=val_loader
            for batch in it:
                if type(batch)==list:
                    batch[0],batch[1]=batch[0].to(device),batch[1].to(device) 
                else:
                    batch.to(device)
                val_loss += trainer.iteration(batch,train=False)
            val_loss /= len(val_loader)
            #val_losses.append(val_loss)
            if rank==0:
                print('Epoch: {:02d} | Train Loss: {:.8f} | Validation Losses: {:.8f}'.format(epoch,train_loss,val_loss))
            #early_stop=False 
            
            if rank==0:
                trainer.scheduler_step(epoch,val_loss=val_loss) 
                early_stop=trainer.check_end(epoch,val_loss) 
                if early_stop:
                    end_flag +=1 
                if save:        
                    trainer.save(epoch, save_dir,ddp=ddp,train_loss=train_loss,val_loss=val_loss)
            if ddp:
                torch.distributed.all_reduce(end_flag)
               
                        
                

            print (f'Rank: {rank} Epoch run time: {time()-epoch_start:.3f} seconds\n')
            if end_flag.item()>0:
                print ('Loss not improved for :',trainer.tolerance,' epochs, exiting training loop ...') 
                break 
    except KeyboardInterrupt:
        if save: 
            if rank==0:
                trainer.finish_training(save_dir) 
        sys.exit()
    else:
        if save: 
            if rank==0:
                trainer.finish_training(save_dir)
        return 
     
    
def classifier_test(model,test_data,batch_size=128,device=None,debug=False,get_graph_rep=False,
                    print_model=False,collate_fn=None,**trainer_kwargs):
    graph_train = trainer_kwargs.get('graph_train',False)                 
    if not graph_train or collate_fn is not None:
        dataloader_cls=torch_dataloader
    else:
        dataloader_cls=pyg_dataloader
       
    test_loader=dataloader_cls(test_data,shuffle=False,batch_size=batch_size,collate_fn=collate_fn)
    if device is None:
        device=torch.device('cpu') 
    model.to(device) 
    if print_model:
        print (model) 
    trainer=SupervisedTrainer(model,**trainer_kwargs) 
    true_labels,predicted_probs=[],[]
    print ('Testing ...')
    all_graph_rep={}
    for data in tqdm(test_loader):
        if type(data)==list:
            data[0],data[1]=data[0].to(device),data[1].to(device) 
            labels=data[1]
        else:
            data.to(device)
            labels=data.y
        #data (print.y,data)  
        if get_graph_rep:
            predicted,graph_reps=trainer.iteration(data,test=True,get_graph_rep=get_graph_rep)
            if not all_graph_rep:
                all_graph_rep={key:val.cpu().detach().numpy() for key,val in graph_reps.items()}
            else:
                for key,val in all_graph_rep.items():
                    #print (key,all_graph_rep[key][-2:]) 
                    all_graph_rep[key]=np.concatenate((val,graph_reps[key].cpu().detach().numpy()),axis=0)
                    #print (all_graph_rep[key][-2:]) 
                #print_events(all_graph_rep)
                
                #sys.exit()  
        else:
            predicted=trainer.iteration(data,test=True,get_graph_rep=get_graph_rep)
        predicted_probs.append(predicted)
        true_labels.append(labels.detach().cpu().squeeze().numpy())
    predicted_probs=np.concatenate(predicted_probs,axis=0)
    true_labels=np.concatenate(true_labels,axis=0) 
    if all_graph_rep:print_events(all_graph_rep,'all graph rep')
    #sys.exit() 
    #print (predicted_probs,true_labels)
    #print (predicted_probs.shape,true_labels.shape) 
    return {'predicted':predicted_probs,'true_labels':true_labels} ,all_graph_rep
     
        
def load_test_model(current_args=None,path=None,device=torch.device('cpu'),return_epoch=False,model_module=None):
    if current_args is None: path=path
    else: path=current_args.path
    args=Unpickle('args',path=path)
    data_kwargs=Unpickle('data_kwargs',path=path)
    model_kwargs=Unpickle('model_kwargs',path=path)
    trainer_kwargs=Unpickle('trainer_kwargs',path=path)
    nice_verbose_args(args) 
    added_keys=[]
    for key in vars(current_args):
        if key not in vars(args):
            setattr(args,key,getattr(current_args,key))
            added_keys.append(key)
    
    if len(added_keys)>0:
        print ('Added keys for argument namespace compatibility: ',added_keys)
        warnings.warn('Ensure default value in current arguments do not change the behaviour of old initialsation variables!')  
    if 'history.pickle' in os.listdir(path):
        history=Unpickle('history',path=path)
        print_events(history) 
        min_epoch=np.argmin(history['val_loss'])
        min_loss= history['val_loss'][min_epoch]
        print ('Min loss:',min_loss,'Min Epoch:',min_epoch) 
        args.min_loss=min_loss
    else:
        args.min_loss = None
        min_epoch=None
    if args.best_only and 'best_model.pkl' in os.listdir(path):
        filename='best_model.pkl'
        with open(os.path.join(path,'epoch_index.txt'),'r') as f:
            lines=f.readlines()
        best_epoch=int(lines[0])
        if min_epoch is not None: 
            assert best_epoch==min_epoch 
    else:
        if args.best_only:
            warnings.warn('Could not find best_model.pkl in path, falling back to best epoch load...') 
        all_epochs=[item for item in os.listdir(path) if item.startswith('ep')]
        if min_epoch is None:
            min_epoch=int(input('history file not found, enter min_epoch: '))
        else:
            assert len(history['val_loss'])==len(all_epochs) 
        filename=f'ep{min_epoch:03}.pkl' if min_epoch>99 else f'ep{min_epoch:02}.pkl'
    #print (all_epochs,filename) 
    print ('Filename: ',filename) 
    if model_module is None:
        model_module=irc_models
    model_cls=getattr(model_module,args.model_name)  
    #print (model_cls) 
    model=model_cls(**model_kwargs) 
    model_path=os.path.join(path,filename)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model.eval()  
    if current_args is None: return model
    else: 
        if return_epoch:
            return args,model,data_kwargs,trainer_kwargs,model_path,min_epoch
        else:
            return args,model,data_kwargs,trainer_kwargs,model_path
    
    
    
    
    
    
    
    
        
        
    
    
