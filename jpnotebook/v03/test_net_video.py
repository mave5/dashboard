#==============================================================================
# libraries
#==============================================================================
import numpy as np
import sys
import os
import matplotlib.pylab as plt
#%matplotlib inline  
import time
import cv2
import imageio
from skvideo.io import VideoWriter
# caffe root 
caffe_root = '/usr/local/caffejuly2016/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

# configure plotting
plt.rcParams['figure.figsize'] = (20, 15)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

print 'All libs were successfully loaded!'
#%%==============================================================================
# functions
#==============================================================================

          
# clean masks
def clean_mask(Y_pred,Y_true):
    # ground truth min area
    s1 = np.sum(np.sum(Y_true, axis=3), axis=2)  # sum over each mask
    nz1 = np.where(s1 > 0)  # find non zeros masks
    minarea = np.min(s1[nz1])  # min area
    maxarea=np.max(s1[nz1])    # max area
    meanarea=np.mean(s1[nz1]) # average area
    print 'min area: %3.2f, max area:,  and average area: ' % minarea, maxarea, meanarea

    # clean predictions
    s2 = np.sum(np.sum(Y_pred, axis=2), axis=2)  # sum over each mask
    nz2 = np.where(s2 > minarea)  # find indices greater than min area
    Y_clean=np.zeros(Y_true.shape,dtype='float32')
    Y_clean[nz2[0],:,:,:]=Y_pred[nz2[0],:,:,:]
    print 'masks were cleaned.'
    return Y_clean


def grays_to_RGB(img):
    # turn 2D grayscale image into grayscale RGB
    return np.dstack((img, img, img))

def image_with_mask(img, mask,color=(1,0,0)):
    maximg=np.max(img)    
    if np.max(mask)==1:
        mask=mask*255
        mask=np.asarray(mask,dtype='uint8')
    # returns a copy of the image with edges of the mask added in red
    if len(img.shape)<3:     
        img_color = grays_to_RGB(img)
    else:
        img_color=img
    if np.sum(mask)>0:    
        mask_edges = cv2.Canny(mask, 100, 200) > 0
    else:
        mask_edges=mask    
    #print np.sum(mask_edges)
    img_color[mask_edges, 0] = maximg*color[0]  # set channel 0 to bright red, green & blue channels to 0
    img_color[mask_edges, 1] = maximg*color[1]
    img_color[mask_edges, 2] = maximg*color[2]
    
    #img_color=img_color/float(np.max(img))
    return img_color

# sample
def disp_img_mask(img,mask):
    n1=np.random.randint(img.shape[0])
    I1=img[n1,0]
    M1=mask[n1,0]
    imgmask=image_with_mask(I1,M1)
    plt.imshow(imgmask)        

#%%==============================================================================
# main            
#==============================================================================
          
# path to recognition model
model1 = '../models/recognition/net_deploy.prototxt'
if not os.path.isfile(model1):
    print("Cannot find the model file!")

# path to recognition weights
weights1 = '../models/recognition/net_iter_45000.caffemodel'
if not os.path.isfile(weights1):
    print("Cannot find caffemodel!")

# path to segmentation model
model2 = '../models/segmentation/net_deploy.prototxt'
if not os.path.isfile(model2):
    print("Cannot find the model file!")

# path to segmentation weights
weights2 = '../models/segmentation/net_iter_46000.caffemodel'
if not os.path.isfile(weights2):
    print("Cannot find caffemodel!")
     
# set gpu mode and id
gpu_id=2
caffe.set_device(gpu_id)
caffe.set_mode_gpu()

net1 = caffe.Net(model1,weights1,caffe.TEST)    
print 'net was successfully created!'

net2 = caffe.Net(model2,weights2,caffe.TEST)    
print 'net was successfully created!'


#%% testing network with real time video
print ('-' *50)
print 'wait to load video ...'

# load video
filename='videos/vid5.avi'
fn_out='videos/vid5_overlay.mp4'
if not os.path.exists(filename):
    print 'video file does not exist!'   

    
vid = imageio.get_reader(filename,  'ffmpeg')
vid_len=vid.get_length()
print 'number of frames: %s' %vid_len

# croping params
W, H=580, 420
#W,H=1040,768
tl_h, tl_w=174, 150 # top left h,w

# video writer
if os.path.exists(fn_out):
    print 'video overlay exists!'
    write_ena=0
else:
    vid_out = VideoWriter(fn_out, 'H264', 20, (W, H))
    vid_out.open()
    write_ena=1

framenum=0

recog_th=0.8 # recognition threshold
l4_th=0.7  # l4 threshold
l123_th=0.8 # l1 l2 l3 threshold
font = cv2.FONT_HERSHEY_SIMPLEX
while(framenum<vid_len):
    # laoding BGR frame    
    frame_bgr = vid.get_data(framenum) 
    # converting to gray    
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) 
    # cropping ROI    
    frame_crop=frame_gray[tl_h:tl_h+H,tl_w:tl_w+W]
    # shape to W  x H
    frame_trans=np.transpose(frame_crop,(1,0))
    # shape to N x C x W x H by ading two dims
    frame_trans=frame_trans[np.newaxis,np.newaxis,:,:]      
    net1.blobs['data'].reshape(*frame_trans.shape)
    net1.blobs['data'].data[...] = frame_trans/(255.)
    net2.blobs['data'].reshape(*frame_trans.shape)
    net2.blobs['data'].data[...] = frame_trans
    
    # forward to net1    
    tmp=net1.forward()
    out1 = net1.blobs['prob'].data[0]
    
    # checking for recognition threshold
    if out1[1]>recog_th:    
        # feeding to segmentation network
        net2.forward();
        pred=net2.blobs['prob'].data[0]
        out = net2.blobs['prob'].data[0].argmax(axis=0);        
        
        # obtaining confidence probabilities        
        l1_c=np.mean(pred[1,:][out==1])
        l2_c=np.mean(pred[2,:][out==2])
        l3_c=np.mean(pred[3,:][out==3])
        l4_c=np.mean(pred[4,:][out==4])
        
        # shape to H x W    
        out=np.transpose(out,(1,0))

        # add overlay to image    
        if l4_c>l4_th:
            frame_crop_bgr=image_with_mask(frame_crop,out==4,(0,1,0))
        else:
            frame_crop_bgr=grays_to_RGB(frame_crop)    
        if np.mean([l1_c,l2_c,l3_c])>l123_th:
            frame_crop_bgr=image_with_mask(frame_crop_bgr,(out>0)&(out!=4) ,(0,0,1))
    else:
        frame_crop_bgr=grays_to_RGB(frame_crop)    
        l1_c,l2_c,l3_c,l4_c=0,0,0,0
        
    # add text to videos    
    cv2.putText(frame_crop_bgr,'Recognition: %.1f' %(out1[1]),(10,350), font, 1,(255,255,255),2)    
    cv2.putText(frame_crop_bgr,'SCM,ASM,MSM,BP: %.1f,%.1f,%.1f,%.1f' %(l1_c,l2_c,l3_c,l4_c),(10,400), font, 1,(255,255,255),2)    
    cv2.imshow('Video',frame_crop_bgr)

    framenum=framenum+1 # frame number    

    # write video 
    if write_ena:
        vid_out.write(frame_crop_bgr)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.close()
if write_ena:
    vid_out.release()
cv2.destroyAllWindows()
print ('-'*50)

