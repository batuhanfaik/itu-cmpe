# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2


# %%
img  = nib.load('/Users/ata/Desktop/vision/BLG453/term project/part4/V.nii').dataobj
gt = nib.load('/Users/ata/Desktop/vision/BLG453/term project/part4/V_seg.nii').dataobj


# %%
def dice_score(seg,gt,eps = 0.0000001):
    dice = ((np.sum(seg[gt==1])*2.0)+eps) / (np.sum(seg) + np.sum(gt)+eps)
    return dice
def seed_selector(img,th = 10):
    seeds=[]
    w = 10
    h = 20
    w_size =10
    for i in range (w_size) :
        window = img[h*(i):h*(i+1),w*(i):w*(i+1)]
        temp = np.sum(window)
        if temp > th:            
            inds = np.where(window == 255)
            y = inds[0][0]+ (h*(i)) 
            x = inds[1][0]+ (w*(i)) 
            seeds.append((y,x))
    return seeds


# %%
def region_grow(img, seeds, points):
    h,w = img.shape
    seg_map = np.zeros(img.shape)
     
    while(len(seeds)>0):
        pt = seeds.pop(0)
        seg_map[pt] =1
        for i in range(len(points)):
            y = pt[0] + points[i][0]
            x = pt[1] + points[i][1]
            if y < 0 or x < 0 or y >= h or x >= w:
                continue
            diff = img[pt] - img[y,x]
            if diff==0 and seg_map[y,x] == 0:
                seg_map[y,x] = 1
                seeds.append((y,x))
    return seg_map
    


# %%
gt = np.array(gt)
for i in range(img.shape[2]):
    if i ==0:
        a = gt[:,:,i].flatten()
    else :
        a = np.concatenate((a,gt[:,:,i].flatten()),axis = None)


# %%
points_4 = [ (0, -1), (1, 0),(0, 1), (-1, 0)]
# points_8 = [ Point(0, -1), Point(1, 0),Point(0, 1), Point(-1, 0)]
label_list = []
for i in range(img.shape[2]):
    den = np.uint8(img[:,:,i]*255)
    dst = cv2.bilateralFilter(den,9,100,100)
    dst = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY)[1]
    seeds = seed_selector(dst)
    if seeds == []:        
        labels = np.zeros(dst.shape) #if empty         
    else:
        labels = region_grow(dst,seeds,points_4 )
    if i ==0:
        b = labels.flatten()
    else :
        b = np.concatenate((b,labels.flatten()),axis = None)
    label_list.append(labels)


# %%
dice_score(a,b)


# %%
plt.imshow(gt[:,:,55])


# %%
plt.imshow(label_list[55])

# %% [markdown]
# # 3D 

# %%
img  = nib.load('/Users/ata/Desktop/vision/BLG453/term project/part4/V.nii').dataobj
gt = nib.load('/Users/ata/Desktop/vision/BLG453/term project/part4/V_seg.nii').dataobj


# %%
def region_grow3d(img, seeds, points):
    h,w,d = img.shape
    seg_map = np.zeros(img.shape)
     
    while(len(seeds)>0):
        pt = seeds.pop(0)
        seg_map[pt] =1
        for i in range(len(points)):
            y = pt[0] + points[i][0]
            x = pt[1] + points[i][1]
            z = pt[2] + points[i][2]
            if y < 0 or x < 0 or z < 0 or y >= h or x >= w or z >=d:
                continue
            diff = img[pt] - img[y,x,z]
            if diff==0 and seg_map[y,x,z] == 0:
                seg_map[y,x,z] = 1
                seeds.append((y,x,z))
    return seg_map
def seed_selector3d(img,th = 10):
    h_img,w_img,d_img = img.shape
    seeds=[]
    w = 10
    h = 20
    d = 5 
    seed_number =5
    w_size = 10
    for i in range (d_img//d) :
        window = img[:,:,d*(i):d*(i+1)]
        inds = np.where(window == 255)
        for j in range(len(inds[0])):
            y = inds[0][0] 
            x = inds[1][0]
            z = inds[2][0]+ (d*(i))
            if y+w_size < h_img and x+w_size < w_img:
                tmp = np.sum(img[y:y+w_size, x: x+ w_size])
                if tmp > th: 
                    seeds.append((y,x,z))
                break 
        if len(seeds) == seed_number:
            break
        
    return seeds


# %%
points_26 = []
for y in range(-1,2):
    for x in range(-1,2):
        for z in range(-1,2):
            if (y,x,z)!= (0,0,0):
                points_26.append((y,x,z))


# %%
for z in range(patient.shape[2]):
    patient[:,:,z] = cv2.bilateralFilter(patient[:,:,z],9,100,100)
    patient[:,:,z] = cv2.threshold(patient[:,:,z], 127, 255, cv2.THRESH_BINARY)[1]

seeds = seed_selector3d(patient)
labels = region_grow3d(patient,seeds,points_26)


# %%
seeds


# %%
plt.imshow(gt[:,:,50])


# %%
plt.imshow(patient[:,:,50])


# %%
plt.imshow(img[:,:,50])


# %%
def region_grow3d(img, seeds, points,th = 10):
    h,w,d = img.shape
    seg_map = np.zeros(img.shape)
     
    while(len(seeds)>0):
        pt = seeds.pop(0)
        seg_map[pt] =1
        for i in range(len(points)):
            y = pt[0] + points[i][0]
            x = pt[1] + points[i][1]
            z = pt[2] + points[i][2]
            if y < 0 or x < 0 or z < 0 or y >= h or x >= w or z >=d:
                continue
            diff = abs(img[pt] - img[y,x,z])
            if diff<th and seg_map[y,x,z] == 0:
                seg_map[y,x,z] = 1
                seeds.append((y,x,z))
    return seg_map
def seed_selector3d(img,th = 10):
    h_img,w_img,d_img = img.shape
    seeds=[]
    w = 10
    h = 20
    d = 5 
    seed_number =5
    w_size = 10
    for i in range (d_img//d) :
        window = img[:,:,d*(i):d*(i+1)]
        inds = np.where(window > 240)
        for j in range(len(inds[0])):
            y = inds[0][j] 
            x = inds[1][j]
            z = inds[2][j]+ (d*(i))
            if y+w_size < h_img and x+w_size < w_img:
                tmp = np.sum(img[y:y+w_size, x: x+ w_size])
                if tmp > th: 
                    seeds.append((y,x,z))
                break 
        if len(seeds) == seed_number:
            break
        
    return seeds


# %%
patient = np.uint8(img[:]*255)
for z in range(patient.shape[2]):
    patient[:,:,z] = cv2.bilateralFilter(patient[:,:,z],9,100,100)
   

seeds = seed_selector3d(patient)
labels = region_grow3d(patient,seeds,points_26)


# %%
plt.imshow(labels[:,:,50])


# %%

tmp = patient[:,:,47]
inds = np.where(patient > 240)


# %%
np.amax(patient)


# %%
seeds


# %%



