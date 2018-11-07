#@author: S. Vinu Sankar
#@date  : Sept 2
#EE645, Assignment 1, Q1
#Python 3.6 and supported libraries

import numpy as np 
import cv2
from matplotlib import pyplot as plt

#Function to create gaussian filter
def gaussian(std, size=9): #std is standard deviation, size is filter size
	filter = np.zeros((size, size))
	sum, sum2 = 0, 0	
	for i in range(size):
		for j in range(size):
			x = abs(i-size//2)
			y = abs(j-size//2)
			filter[i][j] = np.exp(-1*(x**2+y**2)/(2*std**2))/(2*np.pi*std**2)
			sum += filter[i][j]	
	# mean = sum/size**2

	# for i in range(size):
	# 	for j in range(size):
	# 		sum2 += (filter[i][j]-mean)**2
	# v = (sum2/size**2)**0.5
	# print(np.sum(filter/sum))
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

#Function to display image
def disp(img, title):
	plt.title(title)
	plt.imshow(img, cmap='gray', aspect='equal')
	plt.show()

#Output format
np.set_printoptions(precision=3)
#Gaussian filters with std 1, 3 and 20
f1 = gaussian(1)
f3 = gaussian(3)
f20 = gaussian(20)

#Reporting answers
print('***Part (a)***')
print('Gaussian filter with std 1')
print(f1)
print()

print('Gaussian filter with std 3')
print(f3)
print()

print('Gaussian filter with std 20')
print(f20)
print()

#Opening image
img = cv2.resize(cv2.imread('sample.jpg', 0), (200,200)) 
#Displaying image results
print('***Part (b)***')
disp(img, 'Actual image')

#Convolutions
i1 = convolution(img, f1)
i1 = np.array(i1, dtype='uint8')
disp(i1, 'Std 1')
print('Image 1: Std 1')

i3 = convolution(img, f3)
i3 = np.array(i3, dtype='uint8')
disp(i3, 'Std 3')
print('Image 2: Std 3')

i20 = convolution(img, f20)
i20 = np.array(i20, dtype='uint8')
disp(i20, 'Std 20')
print('Image 3: Std 20')