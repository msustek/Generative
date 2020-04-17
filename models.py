import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import math
from abc import ABCMeta, abstractmethod
import os



class EncDec(nn.Module, metaclass=ABCMeta):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        @abstractmethod
        def mean_f(self,x, ind): return NotImplemented
        @abstractmethod
        def lv_f(self,x, ind):   return NotImplemented
        @abstractmethod
        def hid_f(self,x):  return NotImplemented

    
    def forward(self, x, ind):
        h = self.hid_f(x)
        if self.conf.USE_IDEAL_M:
            mean = self.conf.IDEAL_FUNC(x)
        else:
            mean = self.mean_f(h, ind)
            
        if self.conf.FIX_LV:
            return mean, torch.zeros(len(x),self.conf.out_dim_m).double() + self.conf.FIX_LV_CONST 
        
        if self.conf.SHARED_COV:    
            h = torch.ones_like(h)
        logvar = self.lv_f(h, ind)

        if self.conf.out_dim_lv == 1: # Isotropic covariance matrix
            logvar = torch.cat(self.conf.out_dim_m*[logvar], dim=-1)
            
        return mean, logvar

    
    # Do nothing by default
    def prepare_for_eval(self, data_iter, model):
        pass

class FakeEncoder(EncDec):
    def __init__(self, data_count, conf):
        super().__init__(conf)
        self.mean_z = nn.Parameter(torch.zeros(data_count, conf.out_dim_m), requires_grad = True)#.double()
        self.logvar_z = nn.Parameter(torch.zeros(data_count, conf.out_dim_lv) + conf.LV_INIT_V, requires_grad = True) #.double()

    def mean_f(self,x, ind):
        return self.mean_z[ind]
    
    def lv_f(self,x, ind):
        return self.logvar_z[ind]

    def hid_f(self,x):
        return x     
    
    def prepare_for_eval(self, data_iter, model, n_epochs = 250):
        # reinit to default values
        #with torch.no_grad():
        #    self.mean_z.data[model.conf.TRAIN_DATA_C:].mul_(0)
        #    self.logvar_z.data[model.conf.TRAIN_DATA_C:].mul_(0)
        #    self.logvar_z.data[model.conf.TRAIN_DATA_C:].add_(self.conf.LV_INIT_V)
        
        def reset_nans(train_d_c):
            with torch.no_grad():
                val_m = self.mean_z.data[train_d_c:]
                i = val_m != val_m # finds NANs
                self.mean_z.data[train_d_c:][i] = 0#.put_(i,torch.tensor(len(i)*[0]))  

                val_lv = self.logvar_z.data[model.conf.TRAIN_DATA_C:]
                i = val_lv != val_lv # finds NANs
                self.logvar_z.data[train_d_c:][i] = 0
                #self.logvar_z.data[train_d_c:][i].add_(self.conf.LV_INIT_V / 2) # Divide it by 2 so it starts closer to prior which is 0
        
        with torch.enable_grad():
            #opt = model.conf.OPTIMIZER(self.parameters())
            #opt = optim.SGD(self.parameters(), lr = 1e-4)
            opt = optim.Adam(self.parameters())
            # TODO: is 250 epochs reasonable?
            for epoch in range(n_epochs):
                reset_nans(model.conf.TRAIN_DATA_C)
                model.train_one_epoch(opt, data_iter)
            #    print(model.train_one_epoch(opt, data_iter))
            #print("\n")


    
class EncDecNN(EncDec):
    def __init__(self, conf):
        super().__init__(conf)
        self.fc_m  = nn.Linear(conf.hid_dim, conf.out_dim_m)
        self.fc_lv = nn.Linear(conf.hid_dim, conf.out_dim_lv)
        
       # TODO: should it help? 
       # if conf.LV_INIT_V:
       #     with torch.no_grad():
       #         self.fc_lv.bias.data.add_(conf.LV_INIT_V)
       #     print(self.fc_lv.bias)
        

    def mean_f(self,x, ind):
        return self.fc_m(x)

    def lv_f(self,x, ind):
        return self.fc_lv(x)
    

class EncDecFeedForw(EncDecNN):    
    def __init__(self, conf, layer_c, act_fun):
        super().__init__(conf)
        self.layer_c = layer_c
        self.act_fun = act_fun
        
        self.fcs = [nn.Linear(conf.in_dim, conf.hid_dim)]
        for i in range(1, layer_c):
            self.fcs.append(nn.Linear(conf.hid_dim, conf.hid_dim))
        self.fcs = nn.ModuleList(self.fcs)
        
    def hid_f(self,x):
        h = x
        for i in range(self.layer_c):
            h = self.act_fun(self.fcs[i](h))
        return h

class EncDecExpLog(EncDecNN):
    def __init__(self, conf, lay_w, fast_upd):
        super().__init__(conf)
        self.epsilon = 0.0000001
        self.fast_upd = fast_upd

        # TODO: Only one layer support so far
        self.explog_coeffs = [] 
        for i in range(2*lay_w):
            self.explog_coeffs.append(nn.Parameter(torch.Tensor(conf.in_dim)))
        self.explog_coeffs = nn.ParameterList(self.explog_coeffs)
        self.explog_red = nn.Linear(conf.in_dim + (2*lay_w)*3*conf.in_dim, conf.hid_dim)

        with torch.no_grad():
            step = 1.3
            for i in range(lay_w):
                self.explog_coeffs[2*i].data.fill_(math.pow(step,i+1) / self.fast_upd)
                self.explog_coeffs[2*i+1].data.fill_(math.pow(step,-(i+1)) / self.fast_upd)

                
    def explog_trans(self, x, coeff):
        transf = torch.exp(self.fast_upd * coeff * torch.log(torch.abs(x) + self.epsilon)) - self.epsilon        
        return torch.cat([transf, transf * (torch.sign(x)+1)/2, -1 * transf * (torch.sign(x) - 1)/2], dim=-1)
        
            
    def hid_f(self, x):
        transformed = []
        # Could be done in parallel - do not need ATM
        for coeff in self.explog_coeffs:
            transformed.append(self.explog_trans(x,coeff))
        return self.explog_red(torch.cat([x,*transformed],dim=-1))
    
    
class AE_Model(nn.Module):
    def __init__(self, conf, device):
        super().__init__()
        self.conf = conf
        self.device = device
        ## TODO: can you enforce that it is abstract class and must have enc, dec?:
        self.encoder = None
        if conf.USE_EXPLOG:
            self.decoder = EncDecExpLog(conf.dec, conf.EXPLOG_L_WIDTH, conf.EXPLOG_FAST_UPD)
        else:
            self.decoder = EncDecFeedForw(conf.dec, conf.NUM_OF_LAYERS, conf.ACT_FUN)   
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x, ind):
        mu_z, logvar_z = self.encoder(x, ind)
        z = self.reparameterize(mu_z, logvar_z)

        mu_x, logvar_x = self.decoder(z, ind)
        out = self.reparameterize(mu_x, logvar_x)
        return out, mu_z, logvar_z, mu_x, logvar_x   

        #mu_z, logvar_z = self.encoder(x, ind)
        #z = self.reparameterize(mu_z, logvar_z)

        #outs = mus_x = logvars_x = []
        #for dec_sample in range(posterior_samples):
        #    mu_x, logvar_x = self.decoder(z, ind)
        #    out = self.reparameterize(mu_x, logvar_x)
        #    mus_x.append(mu_x)
        #    logvars_x.append(logvar_x)
        #    outs.append(out)
        ##if posterior_samples == 1:
        #    
        #return out, mu_z, logvar_z, torch.tensor(mus_x), torch.tensor(logvars_x)
        #return out, mu_z, logvar_z, mu_x, logvar_x   # if there are more samples from posterior, mu_x and logvar_x correspond to the last one

    
    
    # Reconstruction + KL divergence losses summed over all elements and batch
    def compute_loss(self, recon_x, x, mu_z, logvar_z, mu_x, logvar_x):    
        if self.conf.SAMPLE_FROM_DECODER:
            reconstruct_l = self.conf.RECON_MULT*F.mse_loss(recon_x, x, reduction='sum')
        else:
            reconstruct_l = self.conf.RECON_MULT*0.5*(torch.sum((x-mu_x).pow(2) * torch.exp(-logvar_x)) + torch.sum(logvar_x) + x.numel()*np.log(2*np.pi))

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.sum(1 + logvar_z - self.conf.MU_MULT*mu_z.pow(2) - logvar_z.exp())
        return reconstruct_l + kld, reconstruct_l, kld

    def train_one_epoch(self, optimizer, data_iter):
        self.train()
        tot_rec_l = 0
        tot_kld_l = 0

        for data, ind in data_iter:
            #data, ind = data
            data = torch.from_numpy(data).to(self.device)
            optimizer.zero_grad()
            recon_batch, mu_z, logvar_z, mu_x, logvar_x = self(data, ind)
            #loss = sum(loss_function(recon_batch, data, mu, logvar))
            loss, rec_l, kld_l = self.compute_loss(recon_batch, data, mu_z, logvar_z, mu_x, logvar_x)
            loss.backward()
            if self.conf.CLIPPING:
                torch.nn.utils.clip_grad_value_(self.parameters(),self.conf.CLIP_VALUE)
            tot_rec_l += rec_l.item()
            tot_kld_l += kld_l.item()
            optimizer.step()
        l = data_iter.size
        return (tot_rec_l + tot_kld_l) / l , tot_rec_l / l, tot_kld_l / l
    
    def eval_model(self, data_iter, prepare_for_eval = True):
        self.eval()
        
        tot_rec_l = 0
        tot_kld_l = 0
        if prepare_for_eval:
            # If we have FakeEncoder, it needs to optimize encoder to get reasonable mean and lv
            self.encoder.prepare_for_eval(data_iter, self)
            self.decoder.prepare_for_eval(data_iter, self)
        with torch.no_grad():    
            for data, ind in data_iter:
                data = torch.from_numpy(data).to(self.device)
                recon_batch, mu_z, logvar_z, mu_x, logvar_x = self(data, ind)          
                _, rec_l, kld_l = self.compute_loss(recon_batch, data, mu_z, logvar_z, mu_x, logvar_x)
                tot_rec_l += rec_l.item()
                tot_kld_l += kld_l.item()
        l = data_iter.size
        return (tot_rec_l + tot_kld_l) / l , tot_rec_l / l, tot_kld_l / l
    
    def train_model(self, n_epochs, enc_cycles_c, dec_cycles_c, optimizer_type, train_iter, val_iter = None, modify_during_training = lambda *args: None):
        optimizer_all = optimizer_type(self.parameters())
        optimizer_enc = optimizer_type(self.encoder.parameters())
        optimizer_dec = optimizer_type(self.decoder.parameters())

        train_loss = []   
        val_loss = []
        for epoch in range(n_epochs):
            modify_during_training(self, epoch, train_iter)

            if (enc_cycles_c == 0 and dec_cycles_c == 0): # Train jointly
                train_loss.append(self.train_one_epoch(optimizer_all, train_iter))
            else:
                # TODO: should we store the loss per each cycle or once per epoch?
                for enc_cycle in range(enc_cycles_c):
                    #optimizer_enc = optimizer_type(self.encoder.parameters())
                    train_loss.append(self.train_one_epoch(optimizer_enc, train_iter))
                    #model.train_one_epoch(optimizer_enc, data)
                for dec_cycle in range(dec_cycles_c):
                    #optimizer_dec = optimizer_type(self.decoder.parameters())
                    train_loss.append(self.train_one_epoch(optimizer_dec, train_iter))
            
            # Get validation loss
            if val_iter is not None and epoch % self.conf.EVAL_PER_EPOCH == self.conf.EVAL_PER_EPOCH - 1:
                val_loss.append(self.eval_model(val_iter))
                
            if epoch % self.conf.SAVE_PER_EPOCH == self.conf.SAVE_PER_EPOCH - 1:
                self.save(f"epoch{epoch}")
                # TODO: save model        
        return train_loss, (None if val_loss == [] else val_loss)
    
    def save(self, name):
        torch.save(self, os.path.join(self.conf["MODEL_SAVE_DIR"], name + '.pt'))
    
    
class GauDenNet(AE_Model):        
    def __init__(self, conf, device):
        super().__init__(conf, device)            
        self.encoder = FakeEncoder(conf.ALL_DATA_C, conf.enc)    

        
class VaAuEn(AE_Model):
    def __init__(self, conf, device):
        super().__init__(conf, device)    
        
        if conf.USE_EXPLOG:
            self.encoder = EncDecExpLog(conf.enc, conf.EXPLOG_L_WIDTH, conf.EXPLOG_FAST_UPD)
        else:
            self.encoder = EncDecFeedForw(conf.enc, conf.NUM_OF_LAYERS, conf.ACT_FUN)

