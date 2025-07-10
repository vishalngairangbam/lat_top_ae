import os
import multiprocessing
import sys
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
import vector
pwd= os.getcwd()
import pandas as pd 




'''
aucs=np.array([0.893780643875,0.8934346269750001,0.8920418727999999,0.8937112891500001,0.8929389597124999])
# run2,3,4,5,6,7
'''



def read_dataframe(filename,key):
    df=pd.read_hdf(filename,key)
    keys=df.keys()
    #print (keys)
    unique_keys={}
    for item in keys:
        key,index=item.split('_')[:2]
        if key not in unique_keys: unique_keys[key]={int(index)}
        else: unique_keys[key].add(int(index))
    #unique_keys=set([item.split("_")[0] for item in keys])
    events={}
    for key,indices in unique_keys.items():
        indices=list(indices)
        indices.sort()
        #print (indices)
        vec=vector.array({attr:np.array([df[key+f'_{idx}_{attr}'].values for idx in indices]).swapaxes(0,1).squeeze() for attr in ("px",'py','pz','e')})
        #print (vec.shape)
        events[key]=vec
    return events


def S2(num_points,generator=None,return_angles=False):
    th_angle=torch.arccos(torch.rand(num_points,generator=generator)*2-1)
    ph_angle=torch.rand(num_points,generator=generator)*2*np.pi
    sph_data=torch.stack([torch.cos(ph_angle)*torch.sin(th_angle), \
                          torch.sin(ph_angle)*torch.sin(th_angle),torch.cos(th_angle)], dim=1)
    if return_angles: return sph_data,th_angle,ph_angle
    else: return sph_data 
def rotate_vectorized(i,j,theta,array):
    x,y=array[...,i].clone(),array[...,j].clone()
    array[...,i]=x*torch.cos(theta) + y*torch.sin(theta)
    array[...,j]=-x*torch.sin(theta) + y*torch.cos(theta)
    return array

def rotational_embedding(array,rotation_axes,angles):
    if len(angles)==len(rotation_axes):
        it=zip(rotation_axes,angles)
    else:
        if hasattr(angles,'__len__'):
            assert len(angles) in {1,len(array)} 
        it=zip(rotation_axes,[angles]*len(rotation_axes))
    for axes,angles in it:
        #print (axes) 
        array=rotate_vectorized(*axes,angles,array) 
    return array

def _convert_to_vector(array,attributes,axis=-1):
    
    assert len(attributes)==array.shape[axis]
    attributes=[item.lower() for item in attributes]
    #print (array.shape,attributes)
    dictionary={}
    for key in ['px','py','pz','e']:
        dictionary[key]=array[...,attributes.index(key)]
    #dictionary['E']=dictionary.pop('e') 
    #print (dictionary) 
    array=vector.array(dictionary)
    return array

def convert_to_vector(events):
    attributes=events['attributes']
    new_events={} 
    for key,val in events.items():
        if type(val) != np.ndarray or val.ndim<2: continue 
        new_events[key]=_convert_to_vector(val,attributes)
    #for key,val in new_events.items():
    #    print (key,val.shape) 
    return new_events
def nice_verbose_args(args,return_as_string=False):
    if return_as_string:
        s='Selected argument namespace: \n'
        for key,item in vars(args).items():
            s+=f'\t {key:16} : {repr(item):50}\n'
        return s    
    print ('Selected argument namespace: ')
    for key,item in vars(args).items():
        print (f'\t {key:16} : {repr(item):50}')
    print ('\n')

def set_up_run_folder(namespace,hierarchy=[''],parent_path='./network_runs'): 
    '''Utility function to set up hierarchical directory for final multirun training...
    '''
    print (f'Creating hierarchical save path : {hierarchy}') 
    current_path=check_dir(parent_path)
    for item in hierarchy:
        if item not in vars(namespace):
            current_path=check_dir(current_path,item)
        else:
            value=getattr(namespace,item)
            if type(value)==bool:
                name=item if value else 'no_'+item
            else:
                assert type(value)==str
                name=value
            current_path=check_dir(current_path,name)
    return current_path
class Plotter:
    '''
    '''
    def __init__(self,x_name=None,y_name=None,range=None,size=(10,10),projection=None,title=None,set_range=False):
        
        self.image_range={"x":(-5,5),"y":(-5,5)}
        if range !="image" and not range: range=self.image_range
        self.title=title
        self.projection=projection
        if projection != "subplots":
            if projection=='3d': subplot_kw={'projection': '3d'}
            else: subplot_kw={}  
            fig,axes=plt.subplots(figsize=size,subplot_kw=subplot_kw)#,plt.axes()
            #if self.title is not None:fig.suptitle(title)
            #if projection=="3d": axes=fig.add_subplot(111, projection="3d") 
            #elif projection=="image":pass
            #else: fig.add_subplot(axes)
            if projection ==None and range != None and set_range:
                axes.set_xlim(range["x"]),axes.set_ylim(range["y"])
            if projection not in { "image", '3d' }: 
                if x_name is not None:
                    axes.set_xlabel(x_name,fontsize=40)
                if y_name is not None:
                    axes.set_ylabel(y_name,fontsize=40)
            self.fig,self.axes=fig,axes
        else:self.fig,self.axes=None,None
        self.marker,self.cmap,self.markersize="s","Blues",10
    def save_fig(self,title,extension="png",dpi=100,legend_axes=[0],save_path="plots",set_legends=False,**kwargs):
        pwd=os.getcwd()
        plot_axes=kwargs.get('plot_axes',False)
        if plot_axes:
            if self.projection=='subplots': [self.plot_axes(item) for item in self.axes]
            else: self.plot_axes(self.axes)
        if save_path=="plots":
            dirs = os.listdir(os.getcwd())
            if "plots" not in dirs:
                os.mkdir("plots")
            os.chdir(pwd+'/plots')
        else:
            os.chdir(save_path) 
        
        if 'legend_cols' in kwargs:
            if int(v_mpl.split('.')[1])<6: key='ncol'
            else: key='ncols'
            ext_kwargs={key:kwargs.pop('legend_cols')}
        else:
            ext_kwargs={}  
        if self.projection=='subplots':
            for i,item in enumerate(self.axes):
                if i in legend_axes and set_legends: item.legend(loc=kwargs.get('legend_loc',"best"),
                      prop={'size':kwargs.get('legend_size',25)},
                                                                 title=kwargs.get('legend_title'),
                                                                 title_fontsize=kwargs.get('legend_title_fontsize',30),**ext_kwargs)
                item.tick_params(axis='both',which='major',labelsize=kwargs.get('tick_label_size',28))
        else:
            self.axes.tick_params(axis='both',which='major',labelsize=kwargs.get('tick_label_size',28))
            if set_legends: self.axes.legend(loc=kwargs.get('legend_loc',"best"),prop={'size':kwargs.get('legend_size',25)},
                                             markerscale=kwargs.get('markerscale',1),
                                             title=kwargs.get('legend_title'),title_fontsize=kwargs.get('legend_title_fontsize',30),
                                             **ext_kwargs)
        
        if self.title is not None:
            x_bound=self.axes.get_xbound()
            y_bound=self.axes.get_ybound() 
            
            self.axes.set_title(self.title,size=30,rotation=0,
                            va='center',ha='center')
        self.fig.savefig(title+'.'+extension, format=extension, dpi=dpi,
                         bbox_inches=kwargs.get("bbox_inches","tight"),pad_inches=kwargs.get('pad',0.4),
                         transparent=kwargs.get('transparent',False))
        #plt.show(block=False)
        plt.close()
        print ("Output plot saved at ",os.getcwd(),title+"."+extension)
        os.chdir(pwd)
        return 

def nested_to_vector(array):
    struct_types=array.dtype
    array=array.view(np.float64).reshape(tuple(list(array.shape)+[-1])).view(struct_types).squeeze()\
               .view(vector.MomentumNumpy4D)      
    return array     




def check_dir(*args):
    '''check if <path> to dir exists or not. If it doesn't, create the <dir> returns the absolute path to the created dir'''
    pwd=os.getcwd()
    if len(args)==1:
        path=args[0]
    else:
        path=os.path.join(*args) 
    try:
        os.chdir(path)
    except OSError:
        os.mkdir(path)
        os.chdir(path)
    path=os.getcwd()
    os.chdir(pwd)
    return path
    
def print_events(events,name=None):
    '''Function for printing nested dictionary with at most 3 levels, with final value being a numpy.ndarry, prints the shape of the array'''
    if name: print (name)
    for channel in events:
        if type(events[channel]) == np.ndarray or type(events[channel]) == list:
            if type(events[channel]) == np.ndarray or channel=='EventAttribute': print ("    Final State:", channel,np.array(events[channel]).shape)
            else: 
                print ("    Final State:", channel,len(events[channel]))
            continue
        print ("Channel: ",channel)
        if type(events[channel]) != dict: continue
        for topology in events[channel]:
            if type(events[channel][topology])!= dict: continue
            if type(events[channel][topology])==np.ndarray  or type(events[channel]) == list:
                print ("    Final State: ",topology, np.array(events[channel][topology]).shape)
                continue
            print ("Topology: ",topology)
            for final_state in events[channel][topology]:
                print ("    Final State: ",final_state," Shape: ",events[channel][topology][final_state].shape)
    return





def Unpickle(filename,path=None,load_path=".",verbose=True,keys=None,extension='.pickle'):
    '''load <python_object> from <filename> at location <load_path>'''
    if '.' not in filename: filename=filename+extension
    if path is not None: load_path=path
    pwd=os.getcwd()
    if load_path != ".": os.chdir(load_path)
    if filename[-4:]==".npy":
        ret=np.load(filename,allow_pickle=True)
        if verbose: print (filename," loaded from ",os.getcwd())
        os.chdir(pwd)
        return ret
    try:
        with open(filename,"rb") as File:
            return_object=pickle.load(File)
    except Exception as e:
        print (e," checking if folder with ",filename.split(".")[0]," exists..")
        try: os.chdir(filename.split(".")[0])   
        except Exception as e: 
            os.chdir(pwd)
            raise e     
        print ("exists! loading...")
        return_object=folder_load(keys=keys)
    if verbose: print (filename," loaded from ",os.getcwd())
    os.chdir(pwd)
    return return_object
def Pickle(python_object,filename,path=None,save_path=".",verbose=True,overwrite=True,append=False,extension='.pickle'):
    '''save <python_object> to <filename> at location <save_path>'''
    if '.' not in filename: filename=filename+extension
    if path is not None: save_path=path
    pwd=os.getcwd()
    if save_path != "." :
        os.chdir(save_path)
    if not overwrite:
        if filename in os.listdir("."): 
            raise IOError("File already exists!")
    if append: 
        assert type(python_object)==dict
        prev=Unpickle(filename)
        print_events(prev,name="old")
        python_object=merge_flat_dict(prev,python_object)
        print_events(python_object,name="appended")
    if type(python_object)==np.ndarray:
        np.save(filename,python_object)
        suffix=".npy"
    else:
        try:
            File=open(filename,"wb")
            pickle.dump(python_object,File)
        except OverflowError as e:
            File.close()
            os.system("rm "+filename)
            os.chdir(pwd)
            print (e,"trying to save as numpy arrays in folder...")
            folder_save(python_object,filename.split(".")[0],save_path)
            return
        suffix=""
    if verbose: print (filename+suffix, " saved at ", os.getcwd())
    os.chdir(pwd)
    return
def folder_save(events,folder_name,save_path,append=False):
    pwd=os.getcwd()
    os.chdir(save_path) 
    try: os.mkdir(folder_name)
    except FileExistsError as e: 
        print (e,"Overwriting...")
    finally:os.chdir(folder_name)                      
    for item in events: 
        if append:
            print ("appending...") 
            events[item]=np.concatenate((np.load(item+".npy",allow_pickle=True),events[item]),axis=0)
        if type(events[item]) ==list:
            print("list type found as val, creating directory...")
            os.mkdir(item)
            os.chdir(item)
            for i,array in enumerate(events[item]):
                np.save(item+str(i),array,allow_pickle=True)
                print (array.shape,"saved at ",os.getcwd())
            os.chdir("..")
        else: 
            np.save(item,events[item],allow_pickle=True)
            print (item+".npy saved at ",os.getcwd(), "shape = ",events[item].shape)
    os.chdir(pwd)
    return

def folder_load(keys=None,length=None):
    events=dict()
    pwd=os.getcwd()
    for filename in os.listdir("."):
        if os.path.isdir(filename):
            os.chdir(filename)
            events[filename]=[np.load(array_files,allow_pickle=True) for array_files in os.listdir(".")]
            os.chdir("..")  
            continue          
        if keys is not None:
            if filename[:-4] not in keys: continue
        try:
            events[filename[:-4]]=np.load(filename,allow_pickle=True)[:length]
        except IOError as e:
            os.chdir(pwd)
            raise e
        else:
            print (filename[:-4]," loaded to python dictionary...")
    return events
    
    


    
    
    
    



