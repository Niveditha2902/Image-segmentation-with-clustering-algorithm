#!/usr/bin/env python
# coding: utf-8

# In[ ]:


K-Means Clustering for Image Segmentation using OpenCV in Python
        --Using K-Means Clustering unsupervised machine learning algorithm to 
            segment different parts of an image using OpenCV in Python.
Image segmentation is the process of partitioning an image into multiple different regions (or segments). 
The goal is to change the representation of the image into an easier and more meaningful image.


# In[37]:


import cv2
img=cv2.imread(r"C:\Users\USER\Downloads\nature.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(gray)
plt.title('PICTURE USED TO SEGMENT')
plt.show()


# In[38]:


#CONVERTING TO RGB 
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


# In[9]:


import numpy as np
#use the cv2.kmeans() function which takes a 2D array as input, and since our original image is 3D
pix_val=img.reshape((-1,3))
#converting to float
pix_val=np.float32(pix_val)
print(pix_val)
#Reshaping 2D array of value to 3 color values(RGB)
print(pix_val.shape)


# In[41]:


#stop either when some number of iterations is exceeded (say 100)
#or if the clusters move less than some epsilon valuecriteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,200,0.2)
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,200,0.3)


# In[42]:


#number of clusters(choosing k=4 as there are 4 colors in our image)
k=4
_,labels,(centers)=cv2.kmeans(pix_val,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)


# In[43]:


#converting back to 8 bit value
centers=np.uint8(centers)


# In[44]:


#flatten the image
labels=labels.flatten()


# In[45]:


import matplotlib.pyplot as plt
#segemnted image
segmented_img=centers[labels.flatten()]
segmented_image = segmented_img.reshape(img.shape)
plt.imshow(segmented_image)
plt.show()


# In[32]:


# disable only the cluster number 2 (turn the pixel into black)
masked_image = np.copy(img)
# convert to the shape of a vector of pixel values
masked_image = masked_image.reshape((-1, 3))
# color (i.e cluster) to disable
cluster = 0
masked_image[labels == cluster] = [0, 0, 0]
# convert back to original shape
masked_image = masked_image.reshape(img.shape)
# show the image
plt.imshow(masked_image)
plt.show()


# In[ ]:


Colour quantization 
    --the method of lessening the abundance of different colours applied in an image.
        We may be required to produce this sort of compression to render an image in media supporting 
        only a restricted number of shades


# In[ ]:


def quantimage(img,k):
    i = np.float32(img).reshape(-1,3)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,200,1.0)
    ret,label,center = cv2.kmeans(i, k , None, condition,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    final_img = center[label.flatten()]
    final_img = final_img.reshape(img.shape)
    return final_img


# In[55]:


plt.imshow(quantimage(img,4))
plt.show()

