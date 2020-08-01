from __future__ import print_function
import argparse
import os
import sys
from importlib.machinery import SourceFileLoader
from algorithms import VAE_Algorithm_v2 as alg
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import architectures.VAE_TinyResNet as vae
import cv2
import pickle
import random
import math
from scipy.stats import pearsonr
import datetime
import causal_utils as caus_utils
from lib.eval.hinton import hinton
from lib.eval.regression import *
from sklearn.linear_model import Lasso
from sklearn.ensemble.forest import RandomForestRegressor
from matplotlib.transforms import offset_copy
from lib.eval.regression import normalize
from sklearn.neural_network import MLPRegressor
#from pygraphviz import *


import matplotlib
matplotlib.use('Qt5Agg')


def obtain_representation(test_set,config_file,checkpoint_file,dsprite=True):

    #load Vae
    vae_config_file = os.path.join('.', 'configs', config_file + '.py')
    vae_directory = os.path.join('.', 'models', checkpoint_file)
    vae_config = SourceFileLoader(config_file, vae_config_file).load_module().config 
    vae_config['exp_name'] = config_file
    vae_config['vae_opt']['exp_dir'] = vae_directory # the place where logs, models, and other stuff will be stored
    #print(' *- Loading config %s from file: %s' % (config_file, vae_config_file))   
    vae_algorithm = getattr(alg, vae_config['algorithm_type'])(vae_config['vae_opt'])
    #print(' *- Loaded {0}'.format(vae_config['algorithm_type']))
    vae_algorithm.load_checkpoint('models/'+config_file+"/"+checkpoint_file)
    vae_algorithm.model.eval()
    print("loaded vae")
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #load the dataset
    #f= open(test_set, 'rb')
    input_data = test_set
    decoded_dim=[]
    for i in range(len(input_data)):

        img=input_data[i]
        img_in=img
        if dsprite:
            img=np.expand_dims(img,axis=-1)
        #toch the img and encode it
        x=torch.tensor(img).float().permute(2, 0, 1)
        x=x.unsqueeze(0)
        x = Variable(x).to(device)
        #dec_mean1, dec_logvar1, z, enc_logvar1=vae_algorithm.model.forward(x)
        dec_mean1, z, enc_logvar1=vae_algorithm.model.forward(x)
        
        decoded_dim.append(z[0].cpu().detach().numpy())
    decoded_dim=np.squeeze(np.array(decoded_dim))
    return decoded_dim





def main():

    causal = False

    nn = MLPRegressor(
        hidden_layer_sizes=(10,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
        random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    if causal:      

        config_file="VAEConv2D_v2_CausalDsprite_ber_shape2_scale5_ld3"
        model_name="C-ld-3"
        #checkpoint_files=["vae_checkpoint"+ str(i) + ".pth" for i in range(50)]  
        checkpoint_file="vae_lastCheckpoint.pth"

        dataset_zip = np.load('datasets/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

        print('Keys in the dataset:', dataset_zip.keys())
        imgs = dataset_zip['imgs']

        #get the idxs
        data_sets=[]
        data_sets_true=[]
        #find corrsbonding axes X
        z_x=-1
        d_sprite_idx,X_true_data,_=caus_utils.calc_dsprite_idxs(num_samples=1000,seed=7,constant_factor=[1,0],causal=causal,color=0,shape=2,scale=5)
        X_data=caus_utils.make_dataset_d_sprite(d_sprite_dataset=imgs,dsprite_idx=d_sprite_idx,img_size=256)
        zs_train= obtain_representation(X_data,config_file,checkpoint_file)
        zs_train=np.array(zs_train)
        zs_std=np.std(zs_train,axis=0)
        z_x_idx=np.argmin(zs_std)
        print("X-CONSTANT: ")
        print(zs_std)

        #find corrsbonding axes X
        z_y=-1
        d_sprite_idx,Y_true_data,_=caus_utils.calc_dsprite_idxs(num_samples=1000,seed=7,constant_factor=[0,1],causal=causal,color=0,shape=2,scale=5)
        Y_data=caus_utils.make_dataset_d_sprite(d_sprite_dataset=imgs,dsprite_idx=d_sprite_idx,img_size=256)
        zs_train= obtain_representation(Y_data,config_file,checkpoint_file)
        zs_train=np.array(zs_train)
        zs_std=np.std(zs_train,axis=0)
        z_y_idx=np.argmin(zs_std)
        print("Y-CONSTANT: ")
        print(zs_std)

        if z_y_idx==z_x_idx:
            print("z_y: " + str(z_y_idx))
            print("z_x: " + str(z_x_idx))
            print("Fatal error! could not get unique g to z")
            a=1/0

        #time to learn some causal relationships!!!!
        d_sprite_idx,true_data,_=caus_utils.calc_dsprite_idxs(num_samples=10000,seed=12345,constant_factor=[0,0],causal=causal,color=0,shape=2,scale=5)
        data=caus_utils.make_dataset_d_sprite(d_sprite_dataset=imgs,dsprite_idx=d_sprite_idx,img_size=256)
        zs_data= obtain_representation(data,config_file,checkpoint_file)


        #normalize data
        true_data_norm,_,_,_=normalize(np.array(true_data),remove_constant=False)
        zs_data_norm,_,_,_=normalize(np.array(zs_data),remove_constant=False)

        #train input predictor: 
        input_x_y=true_data_norm[:,:2]
        input_o=np.array(true_data_norm[:,2])        

        n = nn.fit(input_x_y[:8500], input_o[:8500])
        test=nn.predict(input_x_y[8500:])
        input_mse=np.sum(np.square(test-input_o[8500:]))
        print("causal input mse: " + str(input_mse))

        #train latent predictor:
        z_x_y=np.array([zs_data_norm[:,z_x_idx],zs_data_norm[:,z_y_idx]]).T
        z_o_idx_t=[0,1,2]
        z_o_idx=-1
        for i in range(len(z_o_idx_t)):
            if not z_o_idx_t[i] ==z_y_idx and not z_o_idx_t[i] ==z_x_idx:
                z_o_idx=z_o_idx_t[i]

        z_o=zs_data_norm[:,z_o_idx]

        n = nn.fit(z_x_y[:8500], z_o[:8500])
        test=nn.predict(z_x_y[8500:])
        z_mse=np.sum(np.square(test-z_o[8500:]))
        print("causal latent mse: " + str(z_mse))

    
    if not causal:        

        config_file="VAEConv2D_v2_NonCausalDsprite_ber_shape2_scale5_ld3"
        model_name="NC-ld-3"
        #checkpoint_files=["vae_checkpoint"+ str(i) + ".pth" for i in range(50)]  
        checkpoint_file="vae_lastCheckpoint.pth"

        dataset_zip = np.load('datasets/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

        print('Keys in the dataset:', dataset_zip.keys())
        imgs = dataset_zip['imgs']

        #get the idxs
        data_sets=[]
        data_sets_true=[]
        #find corrsbonding axes X
        z_x=-1
        d_sprite_idx,X_true_data,_=caus_utils.calc_dsprite_idxs(num_samples=1000,seed=7,constant_factor=[1,0,0],causal=causal,color=0,shape=2,scale=5)
        X_data=caus_utils.make_dataset_d_sprite(d_sprite_dataset=imgs,dsprite_idx=d_sprite_idx,img_size=256)
        zs_train= obtain_representation(X_data,config_file,checkpoint_file)
        zs_train=np.array(zs_train)
        zs_std=np.std(zs_train,axis=0)
        z_x_idx=np.argmin(zs_std)
        print("X-CONSTANT: ")
        print(zs_std)

        #find corrsbonding axes X
        z_y=-1
        d_sprite_idx,Y_true_data,_=caus_utils.calc_dsprite_idxs(num_samples=1000,seed=7,constant_factor=[0,1,0],causal=causal,color=0,shape=2,scale=5)
        Y_data=caus_utils.make_dataset_d_sprite(d_sprite_dataset=imgs,dsprite_idx=d_sprite_idx,img_size=256)
        zs_train= obtain_representation(Y_data,config_file,checkpoint_file)
        zs_train=np.array(zs_train)
        zs_std=np.std(zs_train,axis=0)
        z_y_idx=np.argmin(zs_std)
        print("Y-CONSTANT: ")
        print(zs_std)

        #find corrsbonding axes O
        z_y=-1
        d_sprite_idx,O_true_data,_=caus_utils.calc_dsprite_idxs(num_samples=1000,seed=7,constant_factor=[0,0,1],causal=causal,color=0,shape=2,scale=5)
        O_data=caus_utils.make_dataset_d_sprite(d_sprite_dataset=imgs,dsprite_idx=d_sprite_idx,img_size=256)
        zs_train= obtain_representation(O_data,config_file,checkpoint_file)
        zs_train=np.array(zs_train)
        zs_std=np.std(zs_train,axis=0)
        z_o_idx=np.argmin(zs_std)
        print("O-CONSTANT: ")
        print(zs_std)

        if z_y_idx==z_x_idx or z_y_idx==z_o_idx or z_o_idx==z_x_idx:
            print("z_y: " + str(z_y_idx))
            print("z_x: " + str(z_x_idx))
            print("z_o: " + str(z_o_idx))
            print("Fatal error! could not get unique g to z")
            a=1/0

        #time to learn some causal relationships!!!!
        d_sprite_idx,true_data,_=caus_utils.calc_dsprite_idxs(num_samples=10000,seed=12345,constant_factor=[0,0,0],causal=causal,color=0,shape=2,scale=5)
        data=caus_utils.make_dataset_d_sprite(d_sprite_dataset=imgs,dsprite_idx=d_sprite_idx,img_size=256)
        zs_data= obtain_representation(data,config_file,checkpoint_file)

        #normalize data
        true_data_norm,_,_,_=normalize(np.array(true_data),remove_constant=False)
        zs_data_norm,_,_,_=normalize(np.array(zs_data),remove_constant=False)

        #train input predictor: 
        input_x_y=true_data_norm[:,:2]
        input_o=np.array(true_data_norm[:,2])        

        n = nn.fit(input_x_y[:8500], input_o[:8500])
        test=nn.predict(input_x_y[8500:])
        input_mse=np.sum(np.square(test-input_o[8500:]))
        print("Non-causal input mse: " + str(input_mse))

        #train latent predictor:
        z_x_y=np.array([zs_data_norm[:,z_x_idx],zs_data_norm[:,z_y_idx]]).T        

        z_o=zs_data_norm[:,z_o_idx]

        n = nn.fit(z_x_y[:8500], z_o[:8500])
        test=nn.predict(z_x_y[8500:])
        z_mse=np.sum(np.square(test-z_o[8500:]))
        print("Non-causal latent mse: " + str(z_mse))


   


if __name__== "__main__":
    main()






#fit_visualise_quantify(regressor, params, err_fn, importances_attr,z_encodes_train,z_encodes_test,gt_labels_train,gt_labels_test,  save_plot=False):


#random forrest
# n_estimators = 10
# all_best_depths = [[12, 10, 10, 10, 10] , [12, 10, 3, 3, 3], [12, 10, 3, 3, 3], [4, 5, 2, 5, 5]]

# # populate params dict with best_depths per model per target (z gt)
# params = [[]] * n_models
# for i, z_max_depths in enumerate(all_best_depths):
#     for z_max_depth in z_max_depths:
#         params[i].append({"n_estimators":n_estimators, "max_depth":z_max_depth, "random_state": rng})

# importances_attr = 'feature_importances_'
# err_fn = nrmse # norm root mean sq. error
# test_time = True
# save_plot = False

# fit_visualise_quantify(RandomForestRegressor, params, err_fn, importances_attr, test_time, save_plot)


# #plots
# zs = [0,0] + list(range(n_z))
# all_import_codes = [[5,8,4,0,2,1,1],[2,5,9,6,1,8,3],[5,7,2,6,1,9,3],[0,8,1,3,4,9,2]]
# n_samples = 5000
# fig, axs = plt.subplots(len(zs), n_models, figsize=(20, 25), facecolor='w', edgecolor='k', sharey=True, sharex=True)

# for i, import_codes in zip(range(n_models), all_import_codes):
#     X_train = m_codes[i][0]    
#     for j, (z, c) in enumerate(zip(zs, import_codes)):
#         X = X_train[:, c:c+1]
#         y = gts[0][:, z]
#         X, y = subset_of_data(X, y, n_samples)
        
#         if i == 0: # set column titles
#             axs[j,i].set_ylabel('$z_{0}$'.format(z), fontsize=28)
        
#         if j == 0:
#             axs[j,i].set_title('{0}'.format(model_names[i]), fontsize=28)

#         axs[j,i].set_xlabel('$c_{0}$'.format(c), fontsize=28)
#         axs[j,i].scatter(y, X, color='black', linewidth=0)        
#         axs[j,i].legend(loc=1, fontsize=21)
#         axs[j,i].set_ylim([-3.5,3.5])
#         axs[j,i].set_xlim([-2,2])
#         axs[j,i].grid(True)
#         axs[j,i].set_axisbelow(True)

# plt.rc('text', usetex=True)
# fig.tight_layout()
# #plt.show()
# plt.savefig(os.path.join(figs_dir, "cvsz.pdf"))


