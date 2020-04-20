#Causality mini project

#Data generation

import os
import sys
import numpy as np
import cv2
import pickle
import random
import math  
import itertools


def make_img_c_girls(img,shape_id,color_id,size,cp,colors,r=20,a=40,b=40/4):

    #circle
    color=(int(colors[color_id][0]),int(colors[color_id][1]),int(colors[color_id][2]))
    if shape_id==1: 
        r=int(size+r)
        img=cv2.circle(img, cp, r, color, -1) 

    #square
    if shape_id==2:
        a=a+size
        p1=(int(cp[0]-a/2),int(cp[1]-a/2))
        p2=(int(cp[0]+a/2),int(cp[1]+a/2))
        img=cv2.rectangle(img,p1,p2,color,-1)

    #triangle
    if shape_id==3:
        a=int(a+size)
        h= math.sqrt(3)/2*a
        p1=[int(cp[0]-a/2),int(cp[1]+h/2)]
        p2=[int(cp[0]+a/2),int(cp[1]+h/2)]
        p3=[int(cp[0]),int(cp[1]-h/2)]
        pts = np.array([p1,p2,p3], np.int32)
        pts = pts.reshape((-1,1,2))
        img=cv2.fillPoly(img,[pts],color)

    #cross
    if shape_id==4:
        a=int(a+size)
        b=int(b+size/4)
        p1=[cp[0]-b/2,cp[1]-int(b/2)]
        p2=[p1[0],p1[1]-int(a/2-b/2)]
        p3=[p2[0]+b,p2[1]]
        p4=[p3[0],p3[1]+int(a/2-b/2)]
        p5=[p4[0]+int(a/2-b/2),p4[1]]
        p6=[p5[0],p5[1]+b]
        p7=[p6[0]-int(a/2-b/2),p6[1]]
        p8=[p7[0],p7[1]+int(a/2-b/2)]
        p9=[p8[0]-b,p8[1]]
        p10=[p9[0],p9[1]-int(a/2-b/2)]
        p11=[p10[0]-int(a/2-b/2),p10[1]]
        p12=[p11[0],p11[1]-b]
        pts = np.array([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12], np.int32)
        pts = pts.reshape((-1,1,2))
        img=cv2.fillPoly(img,[pts],color)

    return img


def get_causal_labels(posX, posY, Xrange=32, Yrange=32, nclasses=8):
    """
    Generate image label from the given raw positions.
    """
    x_label = posX // (Xrange/nclasses)
    y_label = posY // (Yrange/nclasses)
    xy_class = nclasses * x_label + y_label
    return xy_class

def get_noncausal_labels(posX, posY, theta, Xrange=32, Yrange=32, thetarange=40,
                         n_pos_classes=8, n_theta_classes=4):
    """
    Generate image label from the given raw positions.
    """
    classes = list(itertools.product(
        range(n_pos_classes), range(n_pos_classes), range(n_theta_classes)))

    x_label = posX // (Xrange/n_pos_classes)
    y_label = posY // (Yrange/n_pos_classes)
    theta_label = theta // (thetarange/n_theta_classes)
    
    xyt_class = [int(classes.index(element)) for element in list(zip(x_label, y_label, theta_label))]
    return xyt_class
    

def calc_dsprite_idxs(num_samples,seed,constant_factor,causal=True,color=0,shape=0,scale=0, 
                      posXclass_min=0, posXclass_max=31,  posYclass_min=0, posYclass_max=31,
                      orient_min=0, orient_max=39):
    #the generative factors are Possition X and Y rotation depends on X and Y for causal case
    #'color', 'shape', 'scale', 'orientation', 'posX', 'posY'
    
    latents_bases=[ 737280, 245760,  40960,   1024,     32,      1]
    colors=np.ones(num_samples)*color
    shapes=np.ones(num_samples)*shape
    scales=np.ones(num_samples)*scale
    orientations=np.ones(num_samples)*random.randint(orient_min, orient_max)
    posXs=np.ones(num_samples)*random.randint(posXclass_min, posXclass_max)
    posYs=np.ones(num_samples)*random.randint(posYclass_min, posYclass_max)
    latents_list=[]

    for i in range(num_samples):
        if constant_factor[0]==0:
            posXs[i]= random.randint(0, 31)
        if constant_factor[1]==0:
            posYs[i]= random.randint(0, 31)
        if causal:
            orientations[i]= int(((float(posXs[i])*float(posYs[i]))/(31.*31.))*39.)
        if not causal:
            if constant_factor[2]==0:
                orientations[i]= random.randint(0, 39)

    if causal:
        img_clases = get_causal_labels(posXs, posYs)
    else:
        img_clases = get_noncausal_labels(posXs, posYs, orientations)
    latents=np.column_stack((colors,shapes,scales,orientations,posXs,posYs))
    #latents=np.concatenate([colors,shapes,scales,orientations,posXs,posYs],axis=1)
    print(latents[0])
    dsprite_idx=np.dot(latents, latents_bases).astype(int)
    print(dsprite_idx[0])
    true_data=[]
    for i in range(latents.shape[0]):
        if causal:
            true_data.append([latents[i][4],latents[i][5]])
        if not causal:
            true_data.append([latents[i][4],latents[i][5],latents[i][3]])
        

    return dsprite_idx,true_data,img_clases


def make_dataset_d_sprite(d_sprite_dataset,dsprite_idx,img_size=256):
    ds_dataset=d_sprite_dataset[dsprite_idx]
    img_size_data=[]
    for i in range(ds_dataset.shape[0]):
        img=ds_dataset[i]
        img=cv2.resize(img, (img_size,img_size), interpolation = cv2.INTER_AREA)
        img_size_data.append(img)

    return img_size_data




def make_dataset_c_girls(num_samples,seed,constant_factor,causal=True,img_size=256,k=5,mu=0,sigma=10):
    
    #set seed
    random.seed(seed)
    # select if constant efactor is keep
    if constant_factor[0]==1:
        u1=random.randint(1, 4) 
    if constant_factor[1]==1:
        u2=random.randint(1, 5)
    if not causal:
        if constant_factor[2]==1:
            z=random.randint(1, 4*5)*k

    #sample the data
    data_list=[]
    true_data_list=[]
    for i in range(num_samples):
        # pick the not constant factors
        if constant_factor[0]==0:
            u1=random.randint(1, 4) 
        if constant_factor[1]==0:
            u2=random.randint(1, 5) 
        #sample noise
        n,m= np.random.normal(mu, sigma, 2)
        c=np.random.normal(mu, sigma, 3)
        
        #add noise to color
        red=(220+c[0],39+c[1],0+c[2])
        red=np.clip(red,0,255).astype('int')
        blue=(12+c[0],54+c[1],211+c[2])
        blue=np.clip(blue,0,255).astype('int')
        green=(35+c[0],204+c[1],1+c[2])
        green=np.clip(green,0,255).astype('int')
        yellow=(229+c[0],222+c[1],0+c[2])
        yellow=np.clip(yellow,0,255).astype('int')
        pink=(245+c[0],0+c[1],204+c[2])
        pink=np.clip(pink,0,255).astype('int')
        colors=[red,blue,green,yellow,pink]

        # add noise to center point
        n=int(n)
        m=int(m)        
        cp=(int(img_size/2+n),int(img_size/2+m))

        #calculate dependend variable z
        if causal:
            z=u1*u2*k

        if not causal:
            z=random.randint(1, 4*5)*k

        #make images
        img=np.ones((img_size,img_size,3),np.uint8)*255
        img=make_img_c_girls(img,u1,u2-1,z,cp,colors)
        #Debugg
        #cv2.imshow("img",img)
        #cv2.waitKey(0)
        data_list.append(img)
        if causal:
            true_data_list.append([u1,u2])
        if not causal:
            true_data_list.append([u1,u2,z])

    return data_list,true_data_list


