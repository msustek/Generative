import math
import torch
from dotmap import DotMap
from torch import nn, optim
import pickle
import os
from pathlib import Path
import numpy as np
from bokeh.layouts import grid 
from shutil import copyfile
from datetime import datetime
from models import GauDenNet, VaAuEn
from data_iterator import MyBatchDataIter
from model_visual import get_detailed_loss, get_model_introspect, save_fig

LOSS_FILE_HEADER = "Model;Full loss - Train;Reconstruction loss - Train;KLD - Train;Full loss - Dev;Reconstruction loss - Dev;KLD - Dev;\
Start date;Dataset (training examples);Trained jointly/separately;Optimizer;Shared covariance - encoder;Shared covariance - decoder;\
Dimensionality of covariance matrix 1 = isotropic -> otherwise diagonal - encoder;Dimensionality of covariance matrix - decoder\n"


def adam(params):
    return optim.Adam(params, lr=1e-3)
def sgd(params):
    return optim.SGD(params, lr = 1e-4)
def id_ban_enc_m(x):
    return torch.stack([x[:,0]/math.sqrt(2),math.sqrt(25/2)*(x[:,1] - 0.5*x[:,0].pow(2))], dim=-1)
def id_ban_dec_m(z):
    z_x_0_cor_var = math.sqrt(2) * z[:,0]
    return torch.stack([z_x_0_cor_var, z[:,1]*math.sqrt(2/25) + 0.5*z_x_0_cor_var.pow(2)], dim=-1)


def save_losses(results_dir, fname, conf, train_l, val_l, start_time, exp_name):
    path_to_file = os.path.join(results_dir, fname)
    file_exists = os.path.exists(path_to_file)
    with open(path_to_file, 'a') as f: 
        if not file_exists:
            f.write(LOSS_FILE_HEADER)
        f.write(f'{"VAE" if conf["VAE"] else "GDN"};{train_l[0,-1]:.6f};{train_l[1,-1]:.6f};{train_l[2,-1]:.6f};{val_l[0,-1]:.6f};{val_l[1,-1]:.6f};{val_l[2,-1]:.6f};{start_time};{exp_name.replace("_", ";")}\n')

def repair_experiment(conf, modify_during_training, exp_name, dataset, script_path, toy_data_introspect, save_results_cond = None):
    start_time = datetime.now()
    print(f"{exp_name} STARTED at  {start_time}")
    
    main_exp_dir = "experiments"
    results_dir = exper_dir = os.path.join(main_exp_dir, "results")
    exper_dir = os.path.join(main_exp_dir, exp_name[:5] + exp_name[6:], "VAE" if conf["VAE"] else "GDN")        
    fig_dir = os.path.join(exper_dir, "figures")
    models_dir = os.path.join(exper_dir, "saved_models")
    
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(exper_dir).mkdir(parents=True, exist_ok=True)
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    conf["MODEL_SAVE_DIR"] = models_dir
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # TODO: if I do more experiments at once, do not create new data iterators each time for bigger datasets?
    tr_d = dataset[:conf.TRAIN_DATA_C,:]
    val_d = dataset[conf.TRAIN_DATA_C:,:]
    tr_DI = MyBatchDataIter(tr_d, conf["BATCH_SIZE"], True)
    val_DI = MyBatchDataIter(val_d, conf["BATCH_SIZE"], False, start_ind = conf.TRAIN_DATA_C)


    model = torch.load(os.path.join(models_dir, 'final_r.pt'))
    #if conf["VAE"]:
    #    model = VaAuEn(conf, device).to(device)
    #else:
    #    model = GauDenNet(conf, device).to(device);
    #model.double()
    
    train_l = np.load(os.path.join(exper_dir, "train_loss") + ".npy")
    #show_detailed_loss(train_loss)
    
    val = model.eval_model(val_DI, False)
    print(val)
    val = val[0]
    if True:
#    if math.isinf(val) or math.isnan(val):
        val_losses = []
#        if not conf["VAE"]:
#            for i in range(10):
#                # Give extra iterations to optimize logvar and mean for GDN - we had nans somewhere
#                model.eval_model(val_DI) # repair Nans and more samples from posterior

        for i in range(32): 
            # 64 samples from posterior
            l = model.eval_model(val_DI, False)
            val_losses.append(l)

        val_l = np.array(val_losses).mean(axis=0)
        val_l = np.array([val_l]).T
        #print(val_l.shape)
        #model.save("final_r")
        # SAVE e) pictures showing introspection - only for 2D toy dataset
        if toy_data_introspect:
            model_intros_figs = get_model_introspect(model, tr_d, np.arange(conf.TRAIN_DATA_C), val_d, np.arange(conf.TRAIN_DATA_C, conf.ALL_DATA_C), device, ret_col = False, show_var_at_start = True)
            model_intros_figs_grid = grid([[model_intros_figs[0],model_intros_figs[1]], [model_intros_figs[2], model_intros_figs[3]], [model_intros_figs[4], model_intros_figs[5]]])
            #TODO: val error - what to measure -> ELBO / only neg_log_like (many samples?)? Generate (10000?) datapoints from prior and evaluate sum_log_like of real distribution? (Lucas and Katka promised literature survey), 
            #TODO: val error - how often - save once per few epochs - how often?, can we decide what would be things to try?
            #TODO: should we also consider init. bias of NN to be the same as for fake encoder? -> I tried but does not seem to help (mb gradients from logvar much higher and gets it close to prior soon?.. or something else...)
            #TODO: how often save model? 10 epochs?

            #save_fig(model_intros_figs_grid, fig_dir, "model_introspect", model_intros_figs)


            save_losses(results_dir, 'losses.csv', conf, train_l, val_l, start_time, exp_name)
            #if save_results_cond is not None:
            #    for cond in save_results_cond:
            #        file_name = cond + '.csv'
            #        save_losses(results_dir, file_name, conf, train_l, val_l, start_time, exp_name)


        print(f"{exp_name} FINISHED at {datetime.now()}\n")

### LOAD:
#with open(os.path.join(exper_dir, 'conf.p'), 'rb') as f:
#    conf = pickle.load(f)
#model = torch.load(os.path.join(exper_dir, 'model.pt'))
#train_loss = np.load(os.path.join(exper_dir, "train_loss") + ".npy")
#
#show_detailed_loss(train_loss)
#show_model_introspect(model, tr_d, np.arange(len(tr_d)), device)

