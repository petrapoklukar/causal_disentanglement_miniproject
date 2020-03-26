#Causality mini project

#Data generation

import os
import sys
import numpy as np
import cv2
import pickle
import random
import math  

# def pick_color(colors_low,colors_high):
# 	color=[-1,-1,-1]
# 	if colors_low[0]<colors_high[0]:
# 		color[0]=random.randint(colors_low[0],colors_high[0])
# 	elif colors_low[0]>colors_high[0]:
# 		color[0]=random.randint(colors_high[0],colors_low[0])
# 	else:
# 		color[0]=colors_low[0]

# 	if colors_low[1]<colors_high[1]:
# 		color[1]=random.randint(colors_low[1],colors_high[1])
# 	elif colors_low[1]>colors_high[1]:
# 		color[1]=random.randint(colors_high[1],colors_low[1])
# 	else:
# 		color[1]=colors_low[1]

# 	if colors_low[2]<colors_high[2]:
# 		color[2]=random.randint(colors_low[2],colors_high[2])
# 	elif colors_low[2]>colors_high[2]:
# 		color[2]=random.randint(colors_high[2],colors_low[2])
# 	else:
# 		color[2]=colors_low[2]

# 	return color





def make_img(img,shape_id,color_id,size,cp,color,r=20,a=40,b=40/4):

	#circle
	color=colors[color_id]
	if shape_id==1:	
		r=int(size+r)
		img=cv2.circle(img, cp, r, color, -1) 

	#square
	if shape_id==2:
		a=a+size
		p1=(cp[0]-a/2,cp[1]-a/2)
		p2=(cp[0]+a/2,cp[1]+a/2)
		img=cv2.rectangle(img,p1,p2,color,-1)

	#triangle
	if shape_id==3:
		a=int(a+size)
		h= math.sqrt(3)/2*a
		p1=[int(cp[0]-a/2),int(cp[1]+h/2)]
		p2=[int(cp[0]+a/2),int(cp[1]+h/2)]
		p3=[int(cp[0]),int(cp[1]-h/2)]
		print(p1)
		print(p2)
		print(p3)
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






#SCM:
# U1=[1,2,3,4] [circle,square, triangle, cross]
# U2=[1,2,3,4,5] [red, blue, green, yellow, pink]
# n = N(mu,sig)
# m = N(mu,sig)
# c = N(mu,sig)
# X:= U1+(centroid(u=img_size/2+n,u=img_size/2+m))
# Y:= U2+m(color spectrum)
# Z:= X*Y*k

num_img=5000
save_list=[]

seed=12345
img_size=256
mu=0
sigma=10
k=5

random.seed(seed)


for i in range(num_img):

	u1=random.randint(1, 4) 
	u2=random.randint(1, 5) 
	n,m= np.random.normal(mu, sigma, 2)
	c=np.random.normal(mu, sigma, 3)
	c=np.clip(c,0,255).astype('int')
	red=(220+c[0],39+c[1],0+c[2])
	blue=(12+c[0],54+c[1],211+c[2])
	green=(35+c[0],204+c[1],1+c[2])
	yellow=(229+c[0],222+c[1],0+c[2])
	pink=(245+c[0],0+c[1],204+c[2])
	colors=[red,blue,green,yellow,pink]
	n=int(n)
	m=int(m)
	#center point
	cp=(img_size/2+n,img_size/2+m)
	z=abs(u1*u2)*k

	img=np.ones((img_size,img_size,3),np.uint8)*255

	img=make_img(img,u1,u2-1,z,cp,colors)

	#cv2.imshow("img",img)
	#cv2.waitKey(0)

	save_list.append(img)


with open('causal_data.pkl', 'wb') as f:
        pickle.dump(save_list, f)


print("generated: " + str(len(save_list)) + " images.")
print("Causality Girls done!")
