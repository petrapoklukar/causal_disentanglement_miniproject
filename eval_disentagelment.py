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
#from pygraphviz import *


import matplotlib
# matplotlib.use('Qt5Agg')
matplotlib.use('Agg')


def generate_data_4_classifier(n_samples=10000, causal=False):
    dataset_zip = np.load('datasets/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

    print('Keys in the dataset:', dataset_zip.keys())
    imgs = dataset_zip['imgs']

    # d_sprite_idx,X_true_data,labels=caus_utils.calc_dsprite_idxs(
    #     num_samples=10000,seed=999,constant_factor=[0,0],causal=True,color=0,shape=2,scale=5)
    # D_data=caus_utils.make_dataset_d_sprite(d_sprite_dataset=imgs,dsprite_idx=d_sprite_idx,img_size=256)
    # with open('datasets/causal_dsprite_shape2_scale5_imgs_for_classifier.pkl', 'wb') as f:
    #     pickle.dump({'data': D_data, 'labels':labels}, f)
    
    d_sprite_idx,X_true_data,labels=caus_utils.calc_dsprite_idxs(
        num_samples=n_samples,seed=999,constant_factor=[0,0,0],causal=False,color=0,shape=2,scale=5)
    D_data=caus_utils.make_dataset_d_sprite(d_sprite_dataset=imgs,dsprite_idx=d_sprite_idx,img_size=64)
    # with open('datasets/noncausal_dsprite_shape2_scale5_imgs_for_classifier.pkl', 'wb') as f:
    #     pickle.dump({'data': D_data, 'labels':labels}, f)
    print('Label min/max:', min(labels), max(labels))
    zipped_list = list(zip(D_data, labels))
    random.seed(2610)
    random.shuffle(zipped_list)

    prefix = 'non' if not causal else ''
    filename = '{0}causal_dsprite_shape2_scale5_imgs_for_classifier.pkl'.format(prefix)
    splitratio = int(len(zipped_list) * 0.15)
    train_data = zipped_list[splitratio:]
    test_data = zipped_list[:splitratio]
    if 'dsprite' in filename:
        train_data1 = list(map(
            lambda t: (torch.tensor(t[0]).float().unsqueeze(0), 
                       torch.tensor(t[1]).long()), train_data))
        test_data1 = list(map(
            lambda t: (torch.tensor(t[0]).float().unsqueeze(0), 
                       torch.tensor(t[1]).long()), test_data))
    else:
        train_data1 = list(map(
            lambda t: (torch.tensor(t[0]/255.).float().permute(2, 0, 1), 
                       torch.tensor(t[1]).long()), train_data))
        test_data1 = list(map(
            lambda t: (torch.tensor(t[0]/255.).float().permute(2, 0, 1), 
                       torch.tensor(t[1]).long()), test_data))
    print('Train and test split lengths:', len(train_data1), len(test_data1))
    print('An element is of type ', type(train_data1[0]))
    print(' Whereas image and label ', type(train_data1[0][0]), type(train_data1[0][1]))

    with open('datasets/train_'+filename, 'wb') as f:
        pickle.dump(train_data1, f)
    with open('datasets/test_'+filename, 'wb') as f:
        pickle.dump(test_data1, f)


def generate_data_4_vae(n_samples, causal, constant_factor):
    dataset_zip = np.load('datasets/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

    print('Keys in the dataset:', dataset_zip.keys())
    imgs = dataset_zip['imgs']

    d_sprite_idx,X_true_data,labels = caus_utils.calc_dsprite_idxs(
        num_samples=n_samples, seed=12345, constant_factor=constant_factor, 
        causal=causal, color=0, shape=2, scale=5)
    
    D_data = caus_utils.make_dataset_d_sprite(
        d_sprite_dataset=imgs, dsprite_idx=d_sprite_idx, img_size=64)
    prefix = 'non' if not causal else ''
    filename = '{0}causal_dsprite_shape2_scale5_imgs.pkl'.format(prefix)
    random.seed(2610)   

    splitratio = int(len(D_data) * 0.15)
    train_data = D_data[splitratio:]
    test_data = D_data[:splitratio]
    if 'dsprite' in filename:
        train_data1 = list(map(
            lambda t: torch.tensor(t).float().unsqueeze(0), train_data))
        test_data1 = list(map(
            lambda t: torch.tensor(t).float().unsqueeze(0), test_data))
    else:
        train_data1 = list(map(
            lambda t: torch.tensor(t/255.).float().permute(2, 0, 1), train_data))
        test_data1 = list(map(
            lambda t: torch.tensor(t/255.).float().permute(2, 0, 1), test_data))
        
    with open('datasets/train_'+filename, 'wb') as f:
        pickle.dump(train_data1, f)
    with open('datasets/test_'+filename, 'wb') as f:
        pickle.dump(test_data1, f)
    return train_data1, test_data1

    




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


#https://github.com/iffsid/disentangling-disentanglement/blob/public/src/metrics.py
def compute_disentanglement(zs, ys, L=1000, M=20000):
    '''Metric introduced in Kim and Mnih (2018)'''
    N, D = zs.size()
    _, K = ys.size()
    zs_std = torch.std(zs, dim=0)
    ys_uniq = [c.unique() for c in ys.split(1, dim=1)]  # global: move out
    V = torch.zeros(D, K, device=zs_std.device)
    ks = np.random.randint(0, K, M)      # sample fixed-factor idxs ahead of time

    for m in range(M):
        k = ks[m]
        fk_vals = ys_uniq[k]
        # fix fk
        fk = fk_vals[np.random.choice(len(fk_vals))]
        # choose L random zs that have this fk at factor k
        zsh = zs[ys[:, k] == fk]
        zsh = zsh[torch.randperm(zsh.size(0))][:L]
        d_star = torch.argmin(torch.var(zsh / zs_std, dim=0))
        V[d_star, k] += 1

    return torch.max(V, dim=1)[0].sum() / M


def d_sprite_causal_data_all():
    dataset_zip = np.load('datasets/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

    print('Keys in the dataset:', dataset_zip.keys())
    imgs = dataset_zip['imgs']

    #get the idxs
    data_sets=[]
    data_sets_true=[]
    d_sprite_idx,X_true_data,_=caus_utils.calc_dsprite_idxs(num_samples=100,seed=12345,constant_factor=[0,0],causal=True,color=0,shape=2,scale=5)
    

    latents_bases=[ 737280, 245760,  40960,   1024,     32,      1]
    latents_all=[]
    for i in range(32):
        for j in range(32):

            posX= i
            posY= j
            orientations= int(((float(posX)*float(posY))/(31.*31.))*39.)
            latents=[0,2,5,orientations,posX,posY]
            latents_all.append(latents)        

    
    d_sprite_idx=np.dot(np.array(latents_all), latents_bases).astype(int)

    D_data=caus_utils.make_dataset_d_sprite(d_sprite_dataset=imgs,dsprite_idx=d_sprite_idx,img_size=256)
    print("----")
    print(d_sprite_idx.shape)
    print(len(latents_all))
    print(len(D_data))
    grid=32
    grid_v=[]
    for i in range(grid):
        grid_h=[]
        for j in range(grid):
            #print(i*grid+j)
            t_img=D_data[i*grid+j]*255
            bordersize=4
            t_img = cv2.copyMakeBorder(
            t_img,
            top=bordersize,
            bottom=bordersize,
            left=bordersize,
            right=bordersize,
            borderType=cv2.BORDER_CONSTANT,
            value=[100,100,100]
)
            grid_h.append(t_img)

        grid_v.append(np.concatenate([grid_h[x] for x in range(len(grid_h))],axis=1))
    c_dsprite=np.concatenate([grid_v[x] for x in range(len(grid_v))],axis=0)
    cv2.imwrite("cau_test.png",c_dsprite)


def d_sprite_data_example():
    dataset_zip = np.load('datasets/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

    print('Keys in the dataset:', dataset_zip.keys())
    imgs = dataset_zip['imgs']

    #get the idxs
    data_sets=[]
    data_sets_true=[]
    d_sprite_idx,X_true_data,_=caus_utils.calc_dsprite_idxs(num_samples=100,seed=12345,constant_factor=[0,0],causal=True,color=0,shape=2,scale=5)
    
    D_data=caus_utils.make_dataset_d_sprite(d_sprite_dataset=imgs,dsprite_idx=d_sprite_idx,img_size=256)
   
    grid=10
    grid_v=[]
    for i in range(grid):
        grid_h=[]
        for j in range(grid):
            #print(i*grid+j)
            t_img=D_data[i*grid+j]*255
            bordersize=4
            t_img = cv2.copyMakeBorder(
            t_img,
            top=bordersize,
            bottom=bordersize,
            left=bordersize,
            right=bordersize,
            borderType=cv2.BORDER_CONSTANT,
            value=[100,100,100]
)
            grid_h.append(t_img)

        grid_v.append(np.concatenate([grid_h[x] for x in range(len(grid_h))],axis=1))
    c_dsprite=np.concatenate([grid_v[x] for x in range(len(grid_v))],axis=0)
   
    d_sprite_idx,X_true_data,_=caus_utils.calc_dsprite_idxs(num_samples=100,seed=12345,constant_factor=[0,0,0],causal=False,color=0,shape=2,scale=5)
    D_data=caus_utils.make_dataset_d_sprite(d_sprite_dataset=imgs,dsprite_idx=d_sprite_idx,img_size=256)

    grid=10
    grid_v=[]
    for i in range(grid):
        grid_h=[]
        for j in range(grid):
            t_img=D_data[i*grid+j]*255
            bordersize=4
            t_img = cv2.copyMakeBorder(
            t_img,
            top=bordersize,
            bottom=bordersize,
            left=bordersize,
            right=bordersize,
            borderType=cv2.BORDER_CONSTANT,
            value=[100,100,100]
)
            grid_h.append(t_img)

        grid_v.append(np.concatenate([grid_h[x] for x in range(len(grid_h))],axis=1))
    nc_dsprite=np.concatenate([grid_v[x] for x in range(len(grid_v))],axis=0)
    buffer_img=np.ones((nc_dsprite.shape[0],200), np.uint8)*100
    cv2.imwrite("data_examples_dsprite.png",np.concatenate([c_dsprite,buffer_img,nc_dsprite],axis=1))



def d_sprite_tests(causal):

        
    checkpoint_files_t=["vae_checkpoint"+ str(i) + ".pth" for i in range(5,50,3)]
    checkpoint_files=["vae_checkpoint0.pth","vae_checkpoint1.pth","vae_checkpoint2.pth","vae_checkpoint3.pth"]  
    checkpoint_files.extend(checkpoint_files_t)


    dataset_zip = np.load('datasets/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

    print('Keys in the dataset:', dataset_zip.keys())
    imgs = dataset_zip['imgs']

    #get the idxs
    data_sets=[]
    data_sets_true=[]

    if causal:

        checkpoint_files=["vae_checkpoint0.pth","vae_checkpoint1.pth","vae_checkpoint2.pth","vae_checkpoint3.pth","vae_checkpoint332.pth","vae_checkpoint665.pth","vae_lastCheckpoint.pth"]  

        #config_files=["VAE_CausalDsprite_ber_shape2_scale5_ld2","VAE_CausalDsprite_ber_shape2_scale5_ld3","VAE_CausalDsprite_ber_shape2_scale5_ld4","VAE_CausalDsprite_ber_shape2_scale5_ld6","VAE_CausalDsprite_ber_shape2_scale5_ld10"]
        #m_names=["C-ld-2","C-ld-3","C-ld-4","C-ld-6","C-ld-10"]
        config_files=["VAEConv2D_v2_CausalDsprite_ber_shape2_scale5_ld2","VAEConv2D_v2_CausalDsprite_ber_shape2_scale5_ld3","VAEConv2D_v2_CausalDsprite_ber_shape2_scale5_ld4"
        ,"VAEConv2D_v2_CausalDsprite_ber_shape2_scale5_ld6","VAEConv2D_v2_CausalDsprite_ber_shape2_scale5_ld10"]
        m_names=["C-ld-2","C-ld-3","C-ld-4","C-ld-6","C-ld-10"]


        d_sprite_idx,X_true_data,_=caus_utils.calc_dsprite_idxs(num_samples=1000,seed=54321,constant_factor=[1,0],causal=causal,color=0,shape=2,scale=5)
        fix_X_data=caus_utils.make_dataset_d_sprite(d_sprite_dataset=imgs,dsprite_idx=d_sprite_idx,img_size=256)
        X_true_data=np.array(X_true_data)[:,:2]
        
        d_sprite_idx,Y_true_data,_=caus_utils.calc_dsprite_idxs(num_samples=1000,seed=54321,constant_factor=[0,1],causal=causal,color=0,shape=2,scale=5)
        fix_Y_data=caus_utils.make_dataset_d_sprite(d_sprite_dataset=imgs,dsprite_idx=d_sprite_idx,img_size=256)
        Y_true_data=np.array(Y_true_data)[:,:2]

        data_sets.append(fix_X_data)
        data_sets.append(fix_Y_data)

        data_sets_true.append(X_true_data)
        data_sets_true.append(Y_true_data)

    
    if not causal:

        config_files=["VAEConv2D_v2_NonCausalDsprite_ber_shape2_scale5_ld2","VAEConv2D_v2_NonCausalDsprite_ber_shape2_scale5_ld3","VAEConv2D_v2_NonCausalDsprite_ber_shape2_scale5_ld4"
        ,"VAEConv2D_v2_NonCausalDsprite_ber_shape2_scale5_ld6","VAEConv2D_v2_NonCausalDsprite_ber_shape2_scale5_ld10"]


        checkpoint_files_t=["vae_checkpoint"+ str(i) + ".pth" for i in range(332,4680,333)]
        checkpoint_files=["vae_checkpoint0.pth","vae_checkpoint1.pth","vae_checkpoint2.pth","vae_checkpoint3.pth"] 
        checkpoint_files.extend(checkpoint_files_t)
        checkpoint_files_t=["vae_lastCheckpoint.pth"]
        checkpoint_files.extend(checkpoint_files_t)

        m_names=["NC-ld-2","NC-ld-3","NC-ld-4","NC-ld-6","NC-ld-10"]
    

        d_sprite_idx,X_true_data,_=caus_utils.calc_dsprite_idxs(num_samples=1000,seed=54321,constant_factor=[1,0,0],causal=causal,color=0,shape=2,scale=5)
        fix_X_data=caus_utils.make_dataset_d_sprite(d_sprite_dataset=imgs,dsprite_idx=d_sprite_idx,img_size=256)
        
        d_sprite_idx,Y_true_data,_=caus_utils.calc_dsprite_idxs(num_samples=1000,seed=54321,constant_factor=[0,1,0],causal=causal,color=0,shape=2,scale=5)
        fix_Y_data=caus_utils.make_dataset_d_sprite(d_sprite_dataset=imgs,dsprite_idx=d_sprite_idx,img_size=256)

        d_sprite_idx,O_true_data,_=caus_utils.calc_dsprite_idxs(num_samples=1000,seed=54321,constant_factor=[0,0,1],causal=causal,color=0,shape=2,scale=5)
        fix_O_data=caus_utils.make_dataset_d_sprite(d_sprite_dataset=imgs,dsprite_idx=d_sprite_idx,img_size=256)
    
        data_sets.append(fix_X_data)
        data_sets.append(fix_Y_data)
        data_sets.append(fix_O_data)

        data_sets_true.append(X_true_data)
        data_sets_true.append(Y_true_data)
        data_sets_true.append(O_true_data)


    #do it
    for d in range(len(data_sets)):
        plot_conf_list=[]
        for config_file in config_files:
            plot_check_list=[]
            for checkpoint_file in checkpoint_files:
                print("obtaining representations (zs)")
                zs= obtain_representation(data_sets[d],config_file,checkpoint_file)

                #compute disentagelment
                print("calculatiung Kim and Mnih (2018) disentegelment")
                zs_t=torch.tensor(zs)
                true_data_t=torch.tensor(data_sets_true[d])
                dis=compute_disentanglement(zs_t, true_data_t, L=1000, M=20000)
                dis_np=dis.numpy()

                print(config_file + " , " + checkpoint_file + " , " + str(dis_np))
                plot_check_list.append(dis_np)
            plot_conf_list.append(plot_check_list)

        #ploting results
        print("****************** plotting **************************")
        plt.figure(d)
        x=np.arange(1,len(checkpoint_files)+1,1)
        for i in range(len(plot_conf_list)):
            plt.plot(x,plot_conf_list[i],label=m_names[i])

        plt.legend(loc="lower right")
        plt.ylabel('disentagelment metric')
        plt.xticks(x, checkpoint_files, size='small',rotation='vertical')
        plt.xlabel('model checkpoints')
        plt.ylim(0, 1.1)
        if d==0 and not causal:
            plt.title("Non-Causal data - X fixed")
        if d==1 and not causal:
            plt.title("Non-Causal data - Y fixed")
        if d==2 and not causal:
            plt.title("Non-Causal data - O fixed")
        if d==0 and  causal:
            plt.title("Causal data - X fixed")
        if d==1 and  causal:
            plt.title("Causal data - Y fixed")

        if causal:
            plt.savefig("causal_" + str(d) + "_disentagelment_dsprite.png",bbox_inches='tight')
        if not causal:
            plt.savefig("non-causal_" + str(d) + "_disentagelment_dsprite.png",bbox_inches='tight')
        #plt.show()

    print("DONZO!")



# def causal_girls_test():
#      #hyper:
#     config_files=["VAE_CausalData_ld2","VAE_CausalData_ld3","VAE_CausalData_ld4","VAE_CausalData_ld6","VAE_CausalData_ld10"]
   
#     checkpoint_files=["vae_checkpoint1.pth","vae_checkpoint3.pth","vae_checkpoint6.pth","vae_checkpoint9.pth","vae_checkpoint19.pth",
#                     "vae_checkpoint29.pth","vae_checkpoint39.pth","vae_checkpoint49.pth","vae_checkpoint59.pth","vae_checkpoint69.pth",
#                     "vae_checkpoint79.pth","vae_checkpoint89.pth","vae_checkpoint99.pth","vae_checkpoint109.pth","vae_checkpoint119.pth",
#                     "vae_checkpoint129.pth","vae_checkpoint139.pth","vae_checkpoint149.pth","vae_checkpoint159.pth","vae_checkpoint169.pth",
#                     "vae_checkpoint179.pth","vae_checkpoint189.pth","vae_checkpoint199.pth"]

    

#     #gen dependend data:
#     print("producing data")
#     data_sets=[]
#     data_sets_true=[]
#     fix_X_data,X_true_data=caus_utils.make_dataset_c_girls(num_samples=1000,seed=12345,constant_factor=[1,0],causal=causal,img_size=255,k=5,mu=0,sigma=10)
#     fix_Y_data,Y_true_data=caus_utils.make_dataset_c_girls(num_samples=1000,seed=12345,constant_factor=[0,1],causal=causal,img_size=255,k=5,mu=0,sigma=10)
#     data_sets.append(fix_X_data)
#     data_sets.append(fix_Y_data)
#     data_sets_true.append(X_true_data)
#     data_sets_true.append(Y_true_data)

#     #do it
#     for d in range(len(data_sets)):
#         plot_conf_list=[]
#         for config_file in config_files:
#             plot_check_list=[]
#             for checkpoint_file in checkpoint_files:
#                 print("obtaining representations (zs)")
#                 zs= obtain_representation(data_sets[d]/255.,config_file,checkpoint_file)

#                 #compute disentagelment
#                 print("calculatiung Kim and Mnih (2018) disentegelment")
#                 zs_t=torch.tensor(zs)
#                 true_data_t=torch.tensor(data_sets_true[d])
#                 dis=compute_disentanglement(zs_t, true_data_t, L=1000, M=20000)
#                 dis_np=dis.numpy()

#                 print(config_file + " , " + checkpoint_file + " , " + str(dis_np))
#                 plot_check_list.append(dis_np)
#             plot_conf_list.append(plot_check_list)

#         #ploting results
#         print("****************** plotting **************************")
#         plt.figure(d)
#         x=np.arange(1,len(checkpoint_files)+1,1)
#         for i in range(len(plot_conf_list)):
#             plt.plot(x,plot_conf_list[i],label=config_files[i])

#         plt.legend(loc="lower right")
#         plt.ylabel('disentagelment metric')
#         plt.xticks(x, checkpoint_files, size='small',rotation='vertical')
#         plt.xlabel('model checkpoints')
#         plt.ylim(0, 1.1)
#         plt.title("disentagelment evaluation")
#         plt.savefig(str(d) + "_disentagelment.png",bbox_inches='tight')
#         #plt.show()

#     print("DONZO!")





def main():
    # the custem girls dataset
    #causal_girls_test()

    #the desprite case
    causal=False
    d_sprite_tests(causal)

    #make some example images

    # d_sprite_data_example()
   


if __name__== "__main__":
    main()
