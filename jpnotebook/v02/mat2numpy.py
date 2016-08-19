import numpy as np
import sys
import os
import scipy.io as sio
#from PIL import Image
import h5py # to read mat files
import cv2
import matplotlib.pylab as plt

# get numpy version
print np.__version__


def load_traindata(path2data):
    print ('-'*50)
    print 'Please wait to load data ...'
    try:
        f = h5py.File(path2data)
        print f.keys()
        print ('data was loaded using h5py!')
    except:
        f = sio.loadmat(path2data)
        print 'data was loaded using loadmat.'

    # extract images
    X=f['trainData']
    print 'data shape: ', X.shape

    # labels
    Y = f['trainLabels']
    print "label shape: ", Y.shape
    print ('-'*50)
    return X,Y

def load_testdata(path2data):
    print ('-'*50)
    print 'Please wait to load data ...'
    try:
        f = h5py.File(path2data)
        print f.keys()
        print ('data was loaded using h5py!')
    except:
        f = sio.loadmat(path2data)
        print 'data was loaded using loadmat.'

    # extract images
    X=f['testData']
    print 'data shape: ', X.shape

    # labels
    Y = f['testLabels']
    print "label shape: ", Y.shape
    print ('-'*50)
    return X,Y

##############################################################################
##############################################################################
##############################################################################

# path to data
path2matdata='../matfiles/'
path2traindata=path2matdata+'traindata_lmdb.mat'
path2testdata=path2matdata+'testdata_lmdb.mat'

# load train data
X_train,Y_train=load_traindata(path2traindata)

# load test data
X_test,Y_test=load_testdata(path2testdata)

# sample image
n1=np.random.randint(X_test.shape[0])
plt.subplot(121),plt.imshow(X_test[n1,0],cmap='Greys_r')
plt.subplot(122),plt.imshow(Y_test[n1,0])
  
# create numpy folder if does no exist
path2numpy='../numpy/'
if  not os.path.exists(path2numpy):
    os.makedirs(path2numpy)
    print 'numpy folder created'

# save as numpy files
print 'wait to save data as numpy files'
np.savez(path2numpy+'train', X=X_train,Y=Y_train)
np.savez(path2numpy+'test', X=X_test,Y=Y_test)
print 'numpy file was saved!'






