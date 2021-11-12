import PIL
import os, sys
from IPython.display import display
from IPython.display import Image as _Imgdis
from PIL import Image,  ImageOps
import numpy as np
import matplotlib.pyplot as plt
from time import time
from time import sleep
from scipy import ndimage
import glob
from numpy import asarray
def process(b):
    final_test_data=[]
    grey_final_test_data=[]
    final_training=[]
    grey_final_training=[]
    training_label= []
    grey_training_data=[]
    onlyfile=[]
    test_label=[]
    test_data=[]
    grey_test_data=[]
    onlyfiles = [f for f in os.listdir(b) if os.path.isfile(os.path.join(b, f))]
    onlyfiles.sort()

    #print(onlyfiles, "\n")
    print()
    #print("Working with {0} images".format(len(onlyfiles)))
    for i in range(0,len(onlyfiles)-100):
        # load image and convert to and from NumPy array
        # load the image
        #print(onlyfiles[i])
        #print(i)
        path1= "flower_photos"
        train_image = Image.open(os.path.join(b,onlyfiles[i]))
        img_resized = train_image.resize((32,32))
        grey_image= ImageOps.grayscale(train_image)
        grey_image_resized = grey_image.resize((32,32))
        # convert image to numpy array
        training_data = asarray(img_resized)
        train=np.array(training_data)
        final_training.append(train)
        #convert grey images into numpy array 
        grey_training_data = asarray(grey_image_resized)
        grey_train=np.array(grey_training_data)
        grey_final_training.append(grey_train)
        
        training_label.append(b)
        #training_data = training_data.reshape(len(onlyfiles)-100, 200, 200,3).transpose(0,2,3,1).astype("float")
        # summarize shape
        #print(training_data.shape)
        # create Pillow image
        image2 = Image.fromarray(training_data)
    for i in range(len(onlyfiles)-100,len(onlyfiles) ):
        # load image and convert to and from NumPy array
        # load the image
        #print(onlyfiles[i])
        #print(i)
        test_image = Image.open(os.path.join(b,onlyfiles[i]))
        img_resized2 = test_image.resize((32,32))
        #for grey images
        grey_test_image = ImageOps.grayscale(test_image)
        grey_img_resized2 = grey_test_image.resize((32,32))

        # convert image to numpy array
        test_data = asarray(img_resized2)
        test_label.append(b)
        test=np.array(test_data)
        final_test_data.append(test)
        #convert grey image to numpy array
        grey_test_data = asarray(grey_img_resized2)
        #grey_test_label.append(b)
        grey_test=np.array(grey_test_data)
        grey_final_test_data.append(grey_test)

        # summarize shape
        #print(test_data.shape)
        # create Pillow image
        image2 = Image.fromarray(test_data)
        
    final_training = np.reshape(final_training, (len(onlyfiles)-100, 32, 32,3))
    grey_final_training = np.reshape(grey_final_training, (len(onlyfiles)-100, 32, 32))

    #.transpose(0,2,3,1).astype("float")
    final_test_data = np.reshape(final_test_data, (100, 32, 32,3))
    grey_final_test_data = np.reshape(grey_final_test_data, (100, 32, 32))

   
    return final_training, final_test_data, test_label, training_label
  
 #a2 class nfso   == Ytest
#list1 predicted == Ypredicted
def ccrn(Ypredicted, Ytest):
    listt = Ytest.tolist()
    list1=[]
    a2= (sum(listt, []))
    for item in listt:
        list1 = Ypredicted.tolist()
    a=[]
   
    for i in range(5):
        num_correct1=[]
        num_correct2=[]
        num_correctt=[]
        first_half = list1[(i*100):100+(i*100)]
        num_correct1.extend(first_half)
        first_half2 = a2[(i*100):100+(i*100)]
        num_correct2.extend(first_half2)
        num_correct=0
        for j in range(0,100):
            if(num_correct1[j]==num_correct2[j]):
                num_correct+=1
        num_correct= num_correct/100
        a.append(num_correct)
        del num_correct1, num_correct2
        
    return a
    
                
    
    
def pltacc(acc, K,fig_num = 0):
    plt.figure(fig_num)
    plt.title("Cross Validation on K")
    plt.xlabel("K")
    plt.ylabel("cross fold accuracy")
    plt.plot(acc, K , '-o')
    plt.show()
    plt.savefig('cross_valid.png')
   
    
    
    
   
