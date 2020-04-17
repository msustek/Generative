import math
import torch
from dotmap import DotMap
from torch import nn, optim
import os
from pathlib import Path
import pickle
from exp_definition import perform_experiment
from toydata import load_dataset


## COV:
#fixxed 
#shared / per_example_different
#diag / isotropic / full (not supported; we would need different loss function)

conf = DotMap()
conf["enc"] = conf_enc = DotMap()
conf["dec"] = conf_dec = DotMap()



#conf["VAE"] = False # True --> VAE, False --> DensityNetwork (FakeEncoder)
#conf["N_EPOCHS"] = 150
#conf["N_ENC_CYCLES"] = 100
#conf["N_DEC_CYCLES"] = 100
conf["TRAIN_DATA_C"] = 40
conf["ALL_DATA_C"] = 400
conf["BATCH_SIZE"] = 20

def adam(params):
    return optim.Adam(params, lr=1e-3)
def sgd(params):
    return optim.SGD(params, lr = 1e-4)
conf["OPTIMIZER"] = adam
#conf["OPTIMIZER"] = sgd

conf["ACT_FUN"] = torch.relu
#conf["ACT_FUN"] = torch.tanh

conf["SAMPLE_FROM_DECODER"] = False
#conf["SAMPLE_FROM_DECODER"] = True

conf_enc["LV_INIT_V"] = -5 # ONLY for Fake Encoder (Gaussian Density Network)
conf_enc["FIX_LV"]= False
#conf_enc["FIX_LOGVAR"]= True; conf_enc["LV_CONST"] = -5
conf_dec["FIX_LV"]= False
#conf_dec["FIX_LOGVAR"]= True; conf_dec["LV_CONST"] = -1


#conf_enc["SHARED_COV"] = False
#conf_enc["SHARED_COV"] = True
#conf_dec["SHARED_COV"] = False
#conf_dec["SHARED_COV"] = True


conf["INPUT_DIM"] = 2
conf["LATENT_DIM_M"] = conf["OUTPUT_DIM_M"] = conf["INPUT_DIM"]
# isotropic = 1, diagonal = 2 (or anything else), full not supported
#conf["LATENT_DIM_LV"] = 2
#conf["OUTPUT_DIM_LV"] = 1 
conf["HID_EN"] = conf["HID_DE"] = 20
conf["NUM_OF_LAYERS"] = 2

## EXPLOG params
conf["USE_EXPLOG"] = False
# How many lower and higher explog transformations will be available per layer
conf["EXPLOG_L_WIDTH"] = 2
# Faster learning for adam while using EXPLOG
conf["EXPLOG_FAST_UPD"] = 1 

## Clipping
conf["CLIPPING"] = False
conf["CLIP_VALUE"] = 0.2

## If we use both parts and variance would be 0, we get ideal autoencoder (generating from prior also gives exactly p(x))
def id_ban_enc_m(x):
    return torch.stack([x[:,0]/math.sqrt(2),math.sqrt(25/2)*(x[:,1] - 0.5*x[:,0].pow(2))], dim=-1)
def id_ban_dec_m(z):
    z_x_0_cor_var = math.sqrt(2) * z[:,0]
    return torch.stack([z_x_0_cor_var, z[:,1]*math.sqrt(2/25) + 0.5*z_x_0_cor_var.pow(2)], dim=-1)
conf_enc["USE_IDEAL_M"] = False
conf_enc["IDEAL_FUNC"] = id_ban_enc_m
conf_dec["USE_IDEAL_M"] = False 
conf_dec["IDEAL_FUNC"] = id_ban_dec_m

## Experimental, should not be used ATM
# Encourage batch var and mean to be close to standard normal -> $p(z|x)$ close to prior p(z); p(z|x) is computed ONLY per batch
conf["BATCH_L"] = False
# Use KLD between $p(z_i|x_i)$ and prior p(z)
conf["KLD"] = True
conf["MU_MULT"] = 1.0
conf["RECON_MULT"] = 1.0

## Do not touch
conf_enc["in_dim"] = conf["INPUT_DIM"]
conf_enc["hid_dim"] = conf["HID_EN"]
conf_enc["out_dim_m"] = conf["LATENT_DIM_M"] 
#conf_enc["out_dim_lv"] = conf["LATENT_DIM_LV"]

conf_dec["in_dim"] = conf["LATENT_DIM_M"]
conf_dec["hid_dim"] = conf["HID_EN"]
conf_dec["out_dim_m"] = conf["OUTPUT_DIM_M"] 
#conf_dec["out_dim_lv"] = conf["OUTPUT_DIM_LV"]


     
def modify_during_training(model, epoch, dataset):
#    if (epoch == 200):
#        show_model(model, dataset, model.device)
#        #optimizer = optim.Adam(model.decoder.parameters(), lr=1e-3)
#        #model.IDEAL_DEC = False
#        optimizer = optim.Adam(model.encoder.parameters(), lr=1e-3)
#        model.IDEAL_ENC = False
#    if (epoch == 400):
#        show_model(model, dataset, model.device)
#        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    pass

def b(bool_v):
    if bool_v:
        return "T"
    return "F"



if __name__ == "__main__":
    for DS_TYPE in ["square", "banana"]:
        #DS_TYPE = "gauss"
        #DS_TYPE = "square"
        #DS_TYPE = "banana"
        
        data = load_dataset(DS_TYPE).T

        #from toydata import create_all_datasets
        #create_all_datasets()


        # Done: SHARED / NOT ; ISOTR/DIAG (enc/dec) 
        # Not done: LR?; ADAM/SGD; 
        
        
        for update_type in ["joint", "alter"]: 
            if update_type == "joint":
                conf["N_EPOCHS"] = 30000
                conf["N_ENC_CYCLES"] = 0
                conf["N_DEC_CYCLES"] = 0
                conf["EVAL_PER_EPOCH"] = 1000
                conf["SAVE_PER_EPOCH"] = 1000
                
                #conf["N_EPOCHS"] = 100
                #conf["N_ENC_CYCLES"] = 0
                #conf["N_DEC_CYCLES"] = 0
                #conf["EVAL_PER_EPOCH"] = 10
                #conf["SAVE_PER_EPOCH"] = 10
            else:
                conf["N_EPOCHS"] = 150
                conf["N_ENC_CYCLES"] = 100
                conf["N_DEC_CYCLES"] = 100
                conf["EVAL_PER_EPOCH"] = 10
                conf["SAVE_PER_EPOCH"] = 10
                
            for opt_type in ["sgd", "adam"]:
                conf["OPTIMIZER"] = eval(opt_type)
            
                for sce in [True, False]:
                    conf_enc["SHARED_COV"] = sce

                    for scd in [True, False]:
                        conf_dec["SHARED_COV"] = scd

                        for vde in[1,2]:
                            conf_enc["out_dim_lv"] = vde

                            for vdd in [1,2]:
                                conf_dec["out_dim_lv"] = vdd

                                for model_t in [True, False]:
                                    conf["VAE"] = model_t
                                    save_results_cond = [opt_type, update_type, f'{DS_TYPE[:3]}{conf["TRAIN_DATA_C"]}'] 
                                    perform_experiment(conf, modify_during_training, f'{DS_TYPE[:3]}{conf["TRAIN_DATA_C"]}_{update_type}_OPT={opt_type}_SCE={b(sce)}_SCD={b(scd)}_VDE={vde}_VDD={vdd}', data, toy_data_introspect = True, script_path = os.path.abspath(__file__), save_results_cond = save_results_cond)
                                    #exit()

