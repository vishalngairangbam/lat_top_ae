from mlp import MLP 
import torch
import sys 













top_key_dict={'S2xS2':'$S^{2}\\otimes S^2$','RP2xRP2':'$\mathbb{R}\mathbb{P}^2\\otimes \mathbb{R}\mathbb{P}^2$',
              'S2xRP2':'$S^2\\otimes\mathbb{R}\mathbb{P}^2$','S5':'$S^5$','S4':'$S^4$',
              'S2xR2':'$S^2\\otimes\mathbb{R}^2$','S2':'$S^{2}$',
              'RP2xR2':'$\mathbb{R}\mathbb{P}^2\\otimes\mathbb{R}^2$'}
def embed_rp2(inputs):
    xx, yy, zz = inputs[:,0], inputs[:,1], inputs[:,2]
    r=torch.sqrt(xx**2+yy**2+zz**2)
    xx,yy,zz=xx/r,yy/r,zz/r
    out=torch.vstack([xx**2-yy**2,xx*yy,yy*zz,zz*xx]).swapaxes(0,1) 
    return out
def embed_s2(inputs):
    xx, yy, zz = inputs[:,0], inputs[:,1], inputs[:,2]
    r=torch.sqrt(xx**2+yy**2+zz**2)
    xx,yy,zz=xx/r,yy/r,zz/r
    out=torch.vstack([xx,yy,zz]).swapaxes(0,1)  
    return out 
class ClassicalLatentHead(torch.nn.Module):
    def __init__(self,topology='S2',with_radius=False):
        super(ClassicalLatentHead,self).__init__()
        self.topology=topology 
        
    def forward(self,inputs):
        if self.topology=='S2' or self.topology=='S2xS2':
            xx, yy, zz = inputs[:,0], inputs[:,1], inputs[:,2]
            r=torch.sqrt(xx**2+yy**2+zz**2)
            xx,yy,zz=xx/r,yy/r,zz/r
            out=torch.vstack([xx,yy,zz]).swapaxes(0,1) 
            return out
        elif self.topology=='S2xRP2':
            s2_embed=embed_s2(inputs[:,:3])
            rp2_embed=embed_rp2(inputs[:,3:])
            #print (s2_embed.shape,rp2_embed.shape)
            out=torch.cat([s2_embed,rp2_embed],dim=-1)
            return out
        elif self.topology=='RP2':
            
            return embed_rp2(inputs)
        elif self.topology=='S2xR2':
            s2=embed_s2(inputs[:,:3])
            out=torch.cat([s2,inputs[:,3:]],dim=-1)
            return out     
        elif self.topology=='RP2xR2':
            rp2=embed_rp2(inputs[:,:3])
            out=torch.cat([rp2,inputs[:,3:]],dim=-1)
            #print (out.shape)
            #sys.exit()
            return out         
        elif self.topology=='S2xRP2':
            out=torch.cat([embed_rp2(inputs[:,:3]),embed_rp2(inputs[:,3:])],dim=-1)
            return out    
        elif self.topology=='RP2xRP2':
            out=torch.cat([embed_rp2(inputs[:,:3]),embed_rp2(inputs[:,3:])],dim=-1)
            return out
        elif self.topology=='S4':
            x1,x2,x3,x4,x5 = inputs[:,0], inputs[:,1], inputs[:,2],inputs[:,3],inputs[:,4]
            r=torch.sqrt(x1**2+x2**2+x3**2+x4**2+x5**2)
            x1,x2,x3,x4,x5=x1/r,x2/r,x3/r,x4/r,x5/r
            out=torch.vstack([x1,x2,x3,x4,x5]).swapaxes(0,1)
            return out

        elif self.topology=='S5':
            x1,x2,x3,x4,x5,x6=inputs[:,0],inputs[:,1],inputs[:,2],inputs[:,3],inputs[:,4],inputs[:,5]
            r=torch.sqrt(x1**2+x2**2+x3**2+x4**2+x5**2+x6**2)
            x1,x2,x3,x4,x5,x6=x1/r,x2/r,x3/r,x4/r,x5/r,x6/r
            
            out=torch.vstack([x1,x2,x3,x4,x5,x6]).swapaxes(0,1)
            return out 
        else: raise ValueError(f"{self.topology} not recognised!")
        
class HybridAutoEncoder(torch.nn.Module):
    def __init__(self,input_dim,encoder_dims,latent_dim,latent_activation=None,decoder_inp=None,
                 without_decoder=False,decoder_activation=None,
                 trivial_latent_topology=False,topology='S2',dropout=None,
                 latent_scale_factor=1,decoder_dims=None):
        super(HybridAutoEncoder,self).__init__() 
        print ('Hybrid AutoEncoder:',input_dim,encoder_dims)
        print ('Topology:',topology)
        self.topology=topology
        self.classical_enc=MLP(input_dim,encoder_dims,latent_dim,output_activation=latent_activation,dropout=dropout)
        if trivial_latent_topology:
            self.latent_head=None
        else:
            self.latent_head=ClassicalLatentHead(topology=self.topology) 
        
        self.latent_scale_factor=latent_scale_factor
        self.latent_dim=latent_dim
        if not without_decoder:
            if trivial_latent_topology: 
                decoder_inp=latent_dim
            else:
                decoder_inp=self.infer_decoder_inp(input_dim) 
            #print (decoder_inp) 
            #sys.exit() 
            if decoder_dims is None:
                decoder_dims=encoder_dims+[]
                decoder_dims.reverse() 
            self.classical_dec=MLP(decoder_inp,decoder_dims,input_dim,dropout=dropout,
                                   activation=decoder_activation)  
        else:
            self.classical_dec=None
        if self.latent_head is None:
            self.topo_name='$\mathbb{R}^'+f'{latent_dim}$'
        else:
            self.topo_name=top_key_dict[self.topology]  
        print (self) 
    def infer_decoder_inp(self,inp_dim):
        with torch.no_grad():
            random_inp=torch.rand((1,inp_dim))
            lat=self.classical_enc(random_inp) 
            out_dim=lat.shape[-1]
            lat=self.latent_head(lat)
            #print (lat.shape)
            if self.topology=='S2xS2':
                print (lat.shape,out_dim) 
                assert out_dim==6
                dim=lat.shape[-1]*2
            elif self.topology=='RP2':
                assert out_dim==3
                dim=lat.shape[-1]
            elif self.topology=='RP2xRP2':
                assert out_dim==6
                dim=lat.shape[-1]
            elif self.topology=='S2xR2' or self.topology=='RP2xR2':
                assert out_dim==5
                dim=lat.shape[-1]
            elif self.topology=='S2xRP2':
                assert out_dim==6
                dim=lat.shape[-1]
            elif self.topology=='S4':
                assert out_dim==5
                dim=lat.shape[-1]    
            elif self.topology=='S2':
                assert out_dim==3 
                dim=lat.shape[-1]
            elif self.topology=='S5':
                assert out_dim==6
                dim=lat.shape[-1]
            else:
                raise RuntimeError("unrecognized topology in decoder input inference")       
        return dim 
            
    def forward(self,x):
        latent_rep=self.classical_enc(x)
        if self.latent_head is not None:
            if self.topology == 'S2xS2' :
                latent_rep=torch.cat([self.latent_head(self.latent_scale_factor*latent_rep[:,:3]),
                                      self.latent_head(self.latent_scale_factor*latent_rep[:,3:])],
                                      dim=-1)
            else:
                latent_rep=self.latent_head(self.latent_scale_factor*latent_rep) 
        if self.classical_dec is None: return latent_rep 
        decoded=self.classical_dec(latent_rep) 
        return decoded 
    def __str__(self):
        s= 'LatentTopologyAE('+repr(self.classical_enc)+'\nLatent Topology:='
        if self.latent_head is None:
            s += f"R^{self.latent_dim}\n"  
        else:
             s+= self.topology+'\n'+repr(self.latent_head)+'\n'
        s+= repr(self.classical_dec)+')\n'
        return s
    def __repr__(self):
        return str(self)     
        
        
        
        
