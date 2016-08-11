import numpy as np
import sys
import os

caffe_root = '/usr/local/caffejuly2016/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import caffe.draw

model = '../net_deploy.prototxt'
if not os.path.isfile(model):
    print("Cannot find the model file!")

weights1 = '../trainedmodel/net_iter_38000.caffemodel'
if not os.path.isfile(weights1):
    print("Cannot find caffemodel!")
    
   
# set gpu mode and id
gpu_id=2
caffe.set_device(gpu_id)
caffe.set_mode_gpu()
#net1 = caffe.Net(model,weights1,caffe.TEST)    
net1 = caffe.Net(model,caffe.TEST)    
print 'net was successfully created!'
#print("blobs {}\nparams {}".format(net1.blobs.keys(), net1.params.keys()))

# obtain input and output size of each layer
def find_iosizes(net1):
    io_size=[]
    for key in net1.blobs.keys():
        #print key
        temp=net1.blobs[key].shape[0],net1.blobs[key].shape[1],net1.blobs[key].shape[2],net1.blobs[key].shape[3]                        
        #print temp
        io_size=np.append(io_size,temp)
    # reshape         
    return np.reshape(io_size,(len(io_size)/4,4))

# obtain input and output size of each layer, except pooling layers
def find_iosizes2(net1):
    io_size=[]
    for key in net1.params.keys():
        #print key
        temp=net1.blobs[key].shape[0],net1.blobs[key].shape[1],net1.blobs[key].shape[2],net1.blobs[key].shape[3]                        
        #print temp
        io_size=np.append(io_size,temp)
    # reshape         
    return np.reshape(io_size,(len(io_size)/4,4))       

# obtain layer parameters: filter size, etc
def find_params(net1):
    w=[]
    b=[]
    layer_size=[]    
    for k in net1.params:
        for i,p in enumerate(net1.params[k]):
            #layer_size=p.data.shape
            tmp=(reduce(lambda x,y: x*y, p.data.shape))
            if i==0:            
                w=np.append(w,tmp)
                layer_size=np.append(layer_size,p.data.shape)
            if i==1:
                b=np.append(b,tmp)
            #print layer_size
    return w,b,layer_size

##############################################################################
##############################################################################
##############################################################################


# input& input of net layers
io_size=find_iosizes(net1) # with pooling layers
io_size2=find_iosizes2(net1) # without pooling layers

# parameters per layers
w,b,lsize=find_params(net1)

# keys    
lnames=net1.blobs.keys()
pnames=net1.params.keys()

total_ops=[]        
for k in range(0,len(w)-1):
    print ('-'*50)
    print 'Layer name: %s' %pnames[k]
    print 'Layer size: %s' %lsize[4*k:4*k+4]    
    print 'number of params:  %.1e' %w[k]    
    print 'IO_size: %s' %io_size2[k,:]
    ops_per_layer=w[k]*io_size2[k,2]*io_size2[k,3]    
    print 'Operations per layer:  %.1e' %(ops_per_layer)            
    total_ops=np.append(total_ops,ops_per_layer)
    print ('-'*50)

# summarize
print ('-'*50)
print 'Total parameters:  %.2e' %(np.sum(w)+np.sum(b))
print 'Total operations:  %.2e' %np.sum(total_ops)    
