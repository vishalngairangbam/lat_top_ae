import torch
import torch.nn.functional as F
from utils import Plotter
import os,sys
import numpy as np
import torch.nn as nn
from utils import Pickle
import operator
from copy import deepcopy
_major_version=str(torch.__version__).split('.')[0]




def weighted_loss(logits,labels,weights,loss_fn=None):
    unweighted_loss=loss_fn(logits,labels,reduction='none')
    #print (unweighted_loss,weights,weights*unweighted_loss)
    to_sum=weights*unweighted_loss/weights.sum()
    return to_sum.sum()
    
    
    
class SupervisedTrainer:
    measure_funcs={'max':(max,operator.gt),'min':(min,operator.lt)}
    reduce_on_plateau_defaults={'factor':0.5, 'patience':3, 'cooldown':0}
    def __init__(self, model, binary=True,logits=True,lr=0.001,rank=0,best_only=True,
                  scheduler=None,optim='Adam',tolerance=None, 
                       best_measure='min',metric='val_loss',regression=False,
                       plot_logscale=False,scheduler_kwargs=None,exponential_decay=False,
                       decay_start_epoch=None,exponential_decay_kwargs={},warmup_period=None,init_lr=0.01,progression='linspace',
                       decay_on_plateau=False,graph_train=True,scheduler_key="val_loss",optim_kwargs=None,**kwargs):
        if scheduler_kwargs is not None:
            scheduler_kwargs=deepcopy(scheduler_kwargs) 
            print ('Scheduler kwargs:',scheduler_kwargs)
        else:
            scheduler_kwargs={} 
        if optim_kwargs is not None:
            optim_kwargs=deepcopy(optim_kwargs)
            print ('Optimiser kwargs:',optim_kwargs) 
        else:
            optim_kwargs={} 
        self.tolerance=tolerance
        self.model = model
        self.best_only=best_only
        self._best_epoch=None
        self.best_measure=best_measure
        self.metric=metric
        self._end_tup=None
        #self.weighted=args.weighted
        #self.args=args
        if warmup_period is not None and warmup_period != 0:
            self.warmup_lrs=getattr(np,progression)(init_lr,lr,warmup_period) 
            print ('Initialised warmup learning rates: ',self.warmup_lrs) 
            lr=self.warmup_lrs[0]
        else:
            self.warmup_lrs=[]
        self.warmup_period=warmup_period
        self.optim = getattr(torch.optim,optim)(self.model.parameters(), lr=lr,**optim_kwargs) 
        if rank==0:
            print('Total Parameters:', sum([p.nelement() for p in self.model.parameters()]),'Optimizer: ',self.optim)
        
        if decay_on_plateau:
            self.track_loss=True
            self.scheduler_key=scheduler_key
            if not scheduler_kwargs: scheduler_kwargs=self.reduce_on_plateau_defaults
            if _major_version=='1': scheduler_kwargs['verbose']=True
            print ('Decay on plateau kwargs:',scheduler_kwargs) 
            self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, **scheduler_kwargs)
            self.exponential_decay=False
            self.decay_start_epoch=None
            self.exponential_decay_kwargs={} 
        elif scheduler is not None:
            if _major_version=='1': scheduler_kwargs['verbose']=True
            self.track_loss=scheduler_kwargs.pop('track_loss') 
            self.scheduler_key=scheduler_key
            self.scheduler=getattr(torch.optim.lr_scheduler,scheduler)(self.optim,**scheduler_kwargs)
            self.exponential_decay=exponential_decay
            self.decay_start_epoch=decay_start_epoch
            self.exponential_decay_kwargs=exponential_decay_kwargs
        else:
            self.track_loss=False
            self.scheduler= None
        print ('Scheduler: ',self.scheduler) 
        if graph_train:
            self.pre_iter=self.graph_pre
        else:
            self.pre_iter=self.mlp_pre 
        if regression:
            self.do_softmax=False
            self.loss=F.mse_loss
        else:
            self.do_softmax=True
            if binary:
                if logits:
                    self.loss=F.binary_cross_entropy_with_logits
                else:
                    self.loss=F.binary_cross_entropy
            else:
                assert logits
                self.loss=F.cross_entropy 
        if rank==0:
            print ('Loss fn: ',self.loss) 
        self.plot_keys=[] 
        self.plot_logscale=plot_logscale
        self.scheduler_epoch=None
    def graph_pre(self,inputs):
        if self.do_softmax: return inputs,inputs.y.unsqueeze(-1).to(torch.float)
        else: return inputs,inputs.y
    def mlp_pre(self,inputs):
        if self.do_softmax: return inputs[0],inputs[1].unsqueeze(-1).to(torch.float)
        else: return inputs[0],inputs[1] 
    def iteration(self,inputs,train=True,test=False,start_epoch=False,get_graph_rep=False):
        data,labels=self.pre_iter(inputs)#data.y.unsqueeze(-1).to(torch.float)
        if not test:
            if train:self.model.train()
            else: self.model.eval()
            logits=self.model(data)
            
            loss = self.loss(logits, labels)
            assert not torch.isnan(loss).item() 
            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            return loss.item()
        else:
            self.model.eval()
            with torch.no_grad():
                if get_graph_rep:logits,batch_graph_rep=self.model(data,get_graph_rep=get_graph_rep)
                else: logits=self.model(data)
            if self.do_softmax:
                if logits.shape[-1]>1:
                    return_array=F.softmax(logits,dim=-1).cpu().squeeze().detach().numpy()
                else:
                    return_array=torch.sigmoid(logits).cpu().squeeze().detach().numpy() 
            else:
                return_array=logits.cpu().squeeze().detach().numpy() 
            if get_graph_rep: return return_array,batch_graph_rep
            else: return return_array
    
    def finish_training(self,save_path,loss_name='Loss',plot_acc=False,acc_linestyles={},start_ind=1 ): 
        p=Plotter(x_name='epochs',y_name=loss_name,size=(16,10))
        if self.plot_logscale:
            p.axes.set_yscale('log') 
        train_x=np.arange(len(self.train_loss))
        val_x=np.arange(len(self.val_loss))  
        p.axes.plot(train_x,self.train_loss,'-b',linewidth=2.5,label='Training')
        p.axes.plot(val_x[start_ind:],self.val_loss[start_ind:],'-g',linewidth=2.5,label='Validation')
        p.axes.scatter(train_x,self.train_loss,marker='o',s=50,color='k')
        p.axes.scatter(val_x[start_ind:],self.val_loss[start_ind:],marker='o',s=50,color='k')
        if len(train_x)<20: p.axes.set_xticks(train_x)
        else: p.axes.set_xticks(train_x[1::int(len(train_x)/10)])
        if plot_acc:
            p_acc=Plotter(x_name='epochs',y_name='Accuracy',size=(16,10))
            
        history_dict={}
        for key in self.plot_keys:
            history_dict[key]=getattr(self,key)
            if plot_acc and 'acc' in key:
                acc_line_key=key.split('_')[1]
                linestyle=acc_linestyles.get(acc_line_key,'-')
                if key.startswith('train'):
                    p_acc.axes.plot(train_x,history_dict[key],linestyle+'b',label=key,) 
                    p_acc.axes.scatter(train_x,history_dict[key],marker='o',s=50,color='k')
                elif key.startswith('val'):
                    p_acc.axes.plot(val_x,history_dict[key],linestyle+'g',label=key) 
                    p_acc.axes.scatter(train_x,history_dict[key],marker='o',s=50,color='k')
                else:
                    raise ValueError
        Pickle(history_dict,'history',path=save_path) 
        p.save_fig('history',save_path=save_path,set_legends=True)
        if plot_acc:
            p_acc.save_fig('acc_history',save_path=save_path,set_legends=True)
        return history_dict
    def scheduler_step(self,epoch,verbose=True,**kwargs):
        self.scheduler_epoch=epoch
        if self.scheduler is None or epoch == 0:
            return 
        if epoch<len(self.warmup_lrs):
            print ('Within warm up period with learning rates:',self.warmup_lrs) 
            new_lr=self.warmup_lrs[epoch] 
            for param_group in self.optim.param_groups:
                 param_group['lr'] = new_lr
            print(' new lr: ',new_lr) 
            return 
        
        if self.track_loss:
            self.scheduler.step(kwargs.get(self.scheduler_key))
        else:
            self.scheduler.step()
        if verbose and _major_version=='2':
            print ('Last Learning rate:',self.scheduler.get_last_lr())  
        if self.exponential_decay and epoch+1 ==self.decay_start_epoch:
            print (self.exponential_decay_kwargs) 
            self.scheduler=torch.optim.lr_scheduler.ExponentialLR(self.optim,**self.exponential_decay_kwargs) 
            print ('Changing to exponential decay from next epoch...',self.scheduler)
    def check_end(self,epoch,val_loss):
        '''Stops training if loss has not improved for self.tolerance epochs, 
        this is independent of saving and lr schedulers for backwards compatibility and easy debugging
        '''
        if self.tolerance is None: 
            return False
        if self._end_tup is None:
            self._end_tup=epoch,val_loss
            return False
        best_epoch,best_loss=self._end_tup
        if val_loss<best_loss:
            self._end_tup=epoch,val_loss
            return False
        return (epoch-best_epoch)>self.tolerance
    
    def save(self, epoch, save_dir,ddp=False,verbose=1,**kwargs):     
        if self.scheduler is not None and self.scheduler_epoch != epoch:
            self.scheduler_step(epoch,**kwargs) 
        if self.best_only:
            if epoch==0:
                output_path=os.path.join(save_dir,'untrained_model.pkl')
            else:
                all_metrics=getattr(self,self.metric)
                #print (all_metrics)
                func,op=self.measure_funcs[self.best_measure]
                best_metric=func(all_metrics)
                #print (best_metric) 
                current_value=kwargs.get(self.metric)
                 
                if op(current_value,best_metric):
                    self._best_epoch=epoch
                    output_path=os.path.join(save_dir,'best_model.pkl') 
                    with open(os.path.join(save_dir,'epoch_index.txt'),'w+') as f:
                        f.write(f'{epoch}')  
                    print (f'New best {self.metric}:',current_value,f'previous best:',best_metric,) 
                    
                else:
                    output_path=None
                    print (f'Current value {self.metric}:',current_value,f'Best value:',best_metric,'Best epoch:',self._best_epoch) 
        else:
            if epoch>99: output_path = os.path.join(save_dir, 'ep{:03}.pkl'.format(epoch))
            else: output_path = os.path.join(save_dir, 'ep{:02}.pkl'.format(epoch))
        
        for key,val in kwargs.items():
            if key not in dir(self):
                setattr(self,key,[])
                self.plot_keys.append(key) 
            getattr(self,key).append(val) 
        
        if output_path is not None:
            
            if ddp: torch.save(self.model.module.state_dict(), output_path)
            else: torch.save(self.model.state_dict(), output_path)
        
        

 
 
     
        
        
