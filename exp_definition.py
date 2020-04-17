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

def save_losses(results_dir, fname, conf, train_l, val_l, start_time, exp_name):
    path_to_file = os.path.join(results_dir, fname)
    file_exists = os.path.exists(path_to_file)
    with open(path_to_file, 'a') as f: 
        if not file_exists:
            f.write(LOSS_FILE_HEADER)
        f.write(f'{"VAE" if conf["VAE"] else "GDN"};{train_l[0,-1]:.6f};{train_l[1,-1]:.6f};{train_l[2,-1]:.6f};{val_l[0,-1]:.6f};{val_l[1,-1]:.6f};{val_l[2,-1]:.6f};{start_time};{exp_name.replace("_", ";")}\n')

def perform_experiment(conf, modify_during_training, exp_name, dataset, script_path, toy_data_introspect, save_results_cond = None):
    start_time = datetime.now()
    print(f"{exp_name} STARTED at  {start_time}")
    
    main_exp_dir = "experiments"
    results_dir = exper_dir = os.path.join(main_exp_dir, "results")
    exper_dir = os.path.join(main_exp_dir, exp_name, "VAE" if conf["VAE"] else "GDN")        
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


    if conf["VAE"]:
        model = VaAuEn(conf, device).to(device)
    else:
        model = GauDenNet(conf, device).to(device);
    model.double()
    
    
    # SAVE a) config
    with open(os.path.join(exper_dir, 'conf.txt'), 'w') as f:
        for k,v in conf.iteritems():
            f.write(f'{k} : {v}\n')
    with open(os.path.join(exper_dir, 'conf.p'), 'wb') as f:
        pickle.dump(conf, f)
    
    # SAVE b) script for running the experiment
    copyfile(script_path, os.path.join(exper_dir, 'experiment_run.py'))
    
    # TRAIN
    print(f'\tTraining {"VAE" if conf["VAE"] else "GDN"} STARTED  at {datetime.now()}')
    train_l, val_l = model.train_model(conf["N_EPOCHS"], conf["N_ENC_CYCLES"], conf["N_DEC_CYCLES"], conf["OPTIMIZER"], tr_DI, val_DI, modify_during_training)
    print(f'\tTraining {"VAE" if conf["VAE"] else "GDN"} FINISHED at {datetime.now()}')
    
    # SAVE c) final model
    model.save("final")
    
    # SAVE d) train/val loss (data + picture)
    train_l = np.array(train_l).T
    np.save(os.path.join(exper_dir, "train_loss"), train_l)
    fig_tr_l = get_detailed_loss(train_l, "Training loss")
    save_fig(fig_tr_l, fig_dir, "tr_loss")
    
    if val_l is not None:
        val_l = np.array(val_l).T
        np.save(os.path.join(exper_dir, "val_loss"), val_l)
        fig_val_l = get_detailed_loss(val_l, "Validation loss")
        save_fig(fig_val_l, fig_dir, "val_loss")
    
    # SAVE e) pictures showing introspection - only for 2D toy dataset
    if toy_data_introspect:
        model_intros_figs = get_model_introspect(model, tr_d, np.arange(conf.TRAIN_DATA_C), val_d, np.arange(conf.TRAIN_DATA_C, conf.ALL_DATA_C), device, ret_col = False, show_var_at_start = True)
        model_intros_figs_grid = grid([[model_intros_figs[0],model_intros_figs[1]], [model_intros_figs[2], model_intros_figs[3]], [model_intros_figs[4], model_intros_figs[5]]])
        #TODO: val error - what to measure -> ELBO / only neg_log_like (many samples?)? Generate (10000?) datapoints from prior and evaluate sum_log_like of real distribution? (Lucas and Katka promised literature survey), 
        #TODO: val error - how often - save once per few epochs - how often?, can we decide what would be things to try?
        #TODO: should we also consider init. bias of NN to be the same as for fake encoder? -> I tried but does not seem to help (mb gradients from logvar much higher and gets it close to prior soon?.. or something else...)
        #TODO: how often save model? 10 epochs?
    
        save_fig(model_intros_figs_grid, fig_dir, "model_introspect", model_intros_figs)
        
    # SAVE f) RESULTS
        # I) final training loss
        
#    with open(os.path.join(results_dir, 'train_loss.txt'), 'a') as f:
#        f.write(f'{"VAE" if conf["VAE"] else "GDN"}\tFull: {train_l[0,-1]:.6f}   \tRec: {train_l[1,-1]:.6f}\t KLD: {train_l[2,-1]:.6f}\t ST: {start_time}\t{exp_name}\n')
    with open(os.path.join(results_dir, 'train_loss.csv'), 'a') as f:
        f.write(f'{"VAE" if conf["VAE"] else "GDN"};{train_l[0,-1]:.6f};{train_l[1,-1]:.6f};{train_l[2,-1]:.6f};{start_time};{exp_name}\n')
        
        # II) final validation loss
    if val_l is not None:
#        with open(os.path.join(results_dir, 'test_loss.txt'), 'a') as f:
#            f.write(f'{"VAE" if conf["VAE"] else "GDN"}\tFull: {val_l[0,-1]:.6f}   \tRec: {val_l[1,-1]:.6f}\tKLD: {val_l[2,-1]:.6f}\tST: {start_time}\t{exp_name}\n')
        with open(os.path.join(results_dir, 'test_loss.csv'), 'a') as f:
            f.write(f'{"VAE" if conf["VAE"] else "GDN"};{val_l[0,-1]:.6f};{val_l[1,-1]:.6f};{val_l[2,-1]:.6f};{start_time};{exp_name.replace("_", ";")}\n')
        
        save_losses(results_dir, 'losses.csv', conf, train_l, val_l, start_time, exp_name)
        if save_results_cond is not None:
            for cond in save_results_cond:
                file_name = cond + '.csv'
                save_losses(results_dir, file_name, conf, train_l, val_l, start_time, exp_name)
        
        
    print(f"{exp_name} FINISHED at {datetime.now()}\n")

### LOAD:
#with open(os.path.join(exper_dir, 'conf.p'), 'rb') as f:
#    conf = pickle.load(f)
#model = torch.load(os.path.join(exper_dir, 'model.pt'))
#train_loss = np.load(os.path.join(exper_dir, "train_loss") + ".npy")
#
#show_detailed_loss(train_loss)
#show_model_introspect(model, tr_d, np.arange(len(tr_d)), device)

