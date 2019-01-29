# -*- coding: utf-8 -*-
"""
Created on Sun Oct 07 14:32:17 2018

@author: User
"""
#Importing required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#Importing the images and their respective labels into variables
lst = os.listdir("CroppedYale")  #Listing all folders in the folder CroppedYale
X=np.zeros((2452,4096))
names=[]
count=0
c=0
for folder in lst:
    folder_path = "CroppedYale/"+folder
    l = os.listdir(folder_path)  #Listing all the files in the that folder
    for name in l:
        if name[-3:]=="pgm":     #Considering files which are only in .pgm format
            img = cv2.imread(folder_path+"/"+name,0)    #Reading the image file
            im1 = cv2.resize(img,(64,64))   #Resizing to 64X64
            im = im1.flatten()      #Flattening the image
            X[c,:]=im
            c+=1
            names.append(count)     #Adding the label of the image into a list
    count+=1
y = np.array(names)     #Labels

#Constructing random vector of everyimage
random_vectors = np.copy(X.T)

#Constructing covariance matrix of random matrix
mean = np.mean(random_vectors, axis=1) #Mean of all images
scaled = np.copy(random_vectors)
for i in range(2452):
    scaled[:,i] = random_vectors[:,i]-mean #Scaling the data by subtracting mean image from all images

std_dev = np.std(scaled) #Standard Deviation of images
normalized = scaled/std_dev #Normalizing the data by dividing by standard deviation of the data

cov_mat = np.matmul(normalized,normalized.T)/2452.0 #Covariance Matrix

#Calculating eigen values and their corresponding eigen vectors
(eigens,eigenvs) = np.linalg.eigh(cov_mat)

components = 256 #Number of components to be taken
projection = eigenvs[:,-components:] #eigen vectors on to which the data to be projected

#Projecting the data onto lower dimensions
data = np.matmul(X,projection)

#Training and testing the data
accs = 0
for i in range(100):
#   clf = KNeighborsClassifier()
    X_train,X_test,y_train,y_test = train_test_split(data,y,test_size=0.2,random_state=0)   #Splitting data into train and test
    clf = MLPClassifier()
    clf.fit(X_train,y_train)        #Training the model
    y_pred = clf.predict(X_test)    #predicting the test set
    acc = accuracy_score(y_test,y_pred)     #Calculating accuracy
    accs = accs+acc
accs = accs/100.0 #Average accuracy
print "Accuracy is",accs

#Again reprojecting the lower dimensional data onto higher dimensions for visualisation.
reprojected_data = np.matmul(data,projection.T)

#Visualising the reprojected images.
print "Visualising projected and reprojected images"
j=0
for i in range(5):
    plt.subplot(5,2,j+1)
    img = np.copy(X[66*i,:])
    img = img.reshape(64,64)
    plt.imshow(img,cmap="gray")
    
    plt.subplot(5,2,j+2)
    rep_img = np.copy(reprojected_data[66*i,:])
    rep_img = rep_img.reshape(64,64)
    plt.imshow(rep_img,cmap="gray")
    j=j+2
plt.show()
plt.close('all')

#Visualising Top ten eigen faces
print "Visualising top ten eigenfaces"
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(eigenvs[:,-1-i].reshape(64,64),cmap="gray")
plt.show()
plt.close('all')

#Visualising least ten eigen faces
print "Visualising least ten eigenfaces"
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(eigenvs[:,i].reshape(64,64),cmap="gray")
plt.show()
plt.close('all')

#Predicting the label of an image
def prediction(file_path):
    img = cv2.imread(file_path,0)   #Reading the image file
    im = cv2.resize(img,(64,64))    #Resizing to 64X64
    im1 = im.flatten()              #Flattening the image
    
    projected_image = np.matmul(im1,projection)     #Projecting image onto lower dimensions
    predicted_label = clf.predict([projected_image])#Predicting the label of image
#Getting name corresponding to label
    if predicted_label[0]<10:
        predicted_name = "yaleB0"+str(predicted_label[0]+1)
    else:
        if predicted_label[0]>12:
            predicted_name = "yaleB"+str(predicted_label[0]+2)
        else:
            predicted_name = "yaleB"+str(predicted_label[0]+1)
    return predicted_name

image_path = raw_input("Path to the image file: ") #Taking input image path
label = prediction(image_path)
print "\nGiven input face belongs to "+label
