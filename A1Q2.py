#@author: S. Vinu Sankar
#@date  : Sept 2
#EE645, Assignment 1, Q2
#EE645, Assignment 1, Q1
#Python 3.6 and supported libraries

import numpy as np 
import cv2
from matplotlib import pyplot as plt

#Function to create gaussian filter
def gaussian(std, size=11): #std is standard deviation, size is filter size
	filter = np.zeros((size, size))
	sum, sum2 = 0, 0	
	for i in range(size):
		for j in range(size):
			x = abs(i-size//2)
			y = abs(j-size//2)
			filter[i][j] = np.exp(-1*(x**2+y**2)/(2*std**2))/(2*np.pi*std**2)
			sum += filter[i][j]	
	#mean = sum/size**2

	# for i in range(size):
	# 	for j in range(size):
	# 		sum2 += (filter[i][j]-mean)**2
	# v = (sum2/size**2)**0.5
	return filter/sum

#Function to convolute filter with image
def convolution(img, fil): #img is image matrix, fil is the filter
    y = np.zeros(img.shape)
    s = len(fil)
    img1 = np.zeros((img.shape[0]+s//2*2, img.shape[1]+s//2*2))
    print('Preparing array...')
    for i in  range(img1.shape[0]):
    	for j in range(img1.shape[1]):
    		if s//2 <= i < img1.shape[0]-s//2 and s//2 <= j < img1.shape[1]-s//2:
    			img1[i][j] = img[i-s//2][j-s//2]
    print('Convoluting...')
    for i in  range(img1.shape[0]):
    	for j in range(img1.shape[1]):
    		if s//2 <= i < img1.shape[0]-s//2 and s//2 <= j < img1.shape[1]-s//2:
    			x = img1[i-s//2:i+s//2+1, j-s//2:j+s//2+1]
    			p = 0
    			for i1 in range(s):
    				for j1 in range(s):
    					p += x[i1][j1]*fil[i1][j1]
    			y[i-s//2][j-s//2] = p
    return y

#Function to produce binary image with zero crossings
def zerocross(x): #x is image passed
    y = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            try:
                if x[i][j]>0 and min(x[i-1][j],x[i+1][j],x[i][j-1],x[i][j+1])<0:
                    y[i][j] = 1
                # if min(x[i-1][j], x[i+1][j])>=x[i][j] and min(x[i][j+1], x[i][j-1])>=x[i][j]:
                #     y[i][j] = 1
                # if x[i-1][j]*x[i+1][j]<=0 or x[i][j-1]*x[i][j+1]<=0:
                #     y[i][j] = 1
            except:
                y[i][j] = 0
    return y

#Function to display image
def disp(img, title=''): #img is image to display, title is title of plot
	plt.title(title)
	plt.imshow(img, cmap='gray', aspect='equal')
	plt.show()

#Output format
np.set_printoptions(precision=3)
#Reporting answers
print('***Part (a)***')
f2 = gaussian(3.0)
f1 = gaussian(1.5)
dog = np.zeros((11,11))
#Evaluating DoG filter
for i in range(11):
    for j in range(11):
        dog[i][j] = (f2[i][j]-f1[i][j])
#Reporting DoG kernel
print('DoG with gaussians of Std 3.0 and 1.5')
print(dog)

#Reporting image output
print('\n***Part (b)***')
#Opening image
img = cv2.resize(cv2.imread('sample.jpg', 0), (200,200)) 
# img = cv2.imread('butterfly.jpg', 0)
#Convolution
y = convolution(img, dog)
#Scaling y to make it of type uint8
y1 = np.array((y-np.min(y))/np.max(y-np.min(y))*255, dtype='uint8')
print('Image: DoG filtered Std 3.0 and 1.5')
disp(img, 'Actual image')
disp(y1, 'DoG filtered Std 3.0 and 1.5')

#Reporting image output
print('\n***Part (c)***')
print('Finding zero crossings...')
y1 = zerocross(y)
y1 = np.array(y1, dtype='uint8')
print('Dsiplay zero crossings...')
disp(y1, 'Binary zero crossings')