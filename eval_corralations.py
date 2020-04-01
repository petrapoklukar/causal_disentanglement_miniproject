from __future__ import print_function
import argparse
import os
import sys
from importlib.machinery import SourceFileLoader
from algorithms import VAE_Algorithm as alg
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
#from pygraphviz import *


import matplotlib
matplotlib.use('Qt5Agg')



def eval_corralation(test_set,config_file,checkpoint_file):

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
    f= open(test_set, 'rb')
    input_data = pickle.load(f)


    input_x=[]
    input_y=[]
    decoded_dim=[]
    all_recon_img=[]
    for i in range(len(input_data)):
        img=input_data[i][0]
        img_in=img/255.
        input_x.append(input_data[i][1])
        input_y.append(input_data[i][2])

        #toch the img and encode it
        x=torch.tensor(img/255.).float().permute(2, 0, 1)
        x=x.unsqueeze(0)
        x = Variable(x).to(device)
        dec_mean1, dec_logvar1, z, enc_logvar1=vae_algorithm.model.forward(x)
        decoded_dim.append(z[0].cpu().detach().numpy())
        recon_img=dec_mean1[0].detach().permute(1,2,0).cpu().numpy() 
        all_recon_img.append(np.concatenate([img_in,recon_img],axis=1))

        #cv2.imshow("img",np.concatenate([img_in,recon_img],axis=1))
        #cv2.waitKey(0)
        #print(z)

    grid=10
    grid_v=[]
    for i in range(grid):
        grid_h=[]
        for j in range(grid):
            ridx=random.randint(0,len(all_recon_img)-1)
            img=all_recon_img[ridx]*255
            img=img.astype('uint8')
            img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(0,0,0))
            grid_h.append(img)

        grid_v.append(np.concatenate([grid_h[x] for x in range(len(grid_h))],axis=1))
        print(test_set[9:-4] + "_" + config_file + "_examples.png")
    cv2.imwrite("corr_experiment/"+test_set[9:-4] + "_" + config_file + "_examples.png",np.concatenate([grid_v[x] for x in range(len(grid_v))],axis=0))

    decoded_dim=np.array(decoded_dim)
    print(decoded_dim.shape)

    #now we want to know the corralations
    print("----Corralation Girls -----")
    print(config_file)
    print(test_set)
    print("input X :")   
    fig, axs = plt.subplots(decoded_dim.shape[1], 2) 
    for i in range(decoded_dim.shape[1]):
        corr, _ = pearsonr(input_x, decoded_dim[:,i])
        axs[i, 0].plot(input_x,label='input x')
        axs[i, 0].plot(decoded_dim[:,i],label='ld ' + str(i))
        axs[i, 0].legend(loc="upper left")
        axs[i, 0].set_title("(x and ld " + str(i) + "): " + str(np.round(corr,3)))

        #plt.show()
        #plt.savefig("corr_experiment/"+test_set[9:-4] + "_" + config_file + "_input_X.png")
        #print(decoded_dim[:,i].shape)
        print("pearson corr for (x and ld " + str(i) + "): " + str(corr))

    print("input Y :")    
    for i in range(decoded_dim.shape[1]):
        corr, _ = pearsonr(input_y, decoded_dim[:,i])
        axs[i, 1].plot(input_y,label='input y')
        axs[i, 1].plot(decoded_dim[:,i],label='ld ' + str(i))
        axs[i, 1].legend(loc="upper left")
        axs[i, 1].set_title("(y and ld " + str(i) + "): " + str(np.round(corr,3)))

        #plt.show()
        
        print("pearson corr for (y and ld " + str(i) + "): " + str(corr))
    for ax in axs.flat:
        ax.label_outer()
    fig.tight_layout()
    fig.set_size_inches(20.5, 12.5)
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.savefig("corr_experiment/"+test_set[9:-4] + "_" + config_file + "_corr.png")
    print("-----------------")

    #write to file
    file_writer = open("corr_experiment/"+'results.txt', 'a')
    file_writer.write("----Corralation Girls -----\n")
    file_writer.write(config_file + '\n')
    file_writer.write(test_set+'\n')
    file_writer.write("input X :"+'\n')
    for i in range(decoded_dim.shape[1]):
        corr, _ = pearsonr(input_x, decoded_dim[:,i])
        file_writer.write("pearson corr for (x and ld " + str(i) + "): " + str(corr) + '\n')

    file_writer.write("input Y :"+ '\n')    
    for i in range(decoded_dim.shape[1]):
        corr, _ = pearsonr(input_y, decoded_dim[:,i])
        file_writer.write("pearson corr for (y and ld " + str(i) + "): " + str(corr)+ '\n')

    file_writer.write("-----------------"+ '\n')









def main():

     

    test_sets=["X_data.pkl","Y_data.pkl"]

    config_files=["VAE_CausalData_ld2","VAE_CausalData_ld3","VAE_CausalData_ld4","VAE_CausalData_ld6","VAE_CausalData_ld10"]

    checkpoint_file="vae_lastCheckpoint.pth"

    datetime_object = datetime.datetime.now()
    file_writer = open("corr_experiment/"+'results.txt', 'w')
    file_writer.write(datetime_object.strftime("%m/%d/%Y, %H:%M:%S"))
    file_writer.write('\n')
    file_writer.write('\n')
    for test_set in test_sets:
        for config_file in config_files:
            eval_corralation("datasets/"+test_set,config_file,checkpoint_file)

    print("Donzo!")



if __name__== "__main__":
  main()
