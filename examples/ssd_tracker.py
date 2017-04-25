
# coding: utf-8

# # Detection with SSD
# 
# In this example, we will load a SSD model and use it to detect objects.

# ### 1. Setup
# 
# * First, Load necessary libs and set up caffe and caffe_root

# In[3]:

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
# get_ipython().magic(u'matplotlib inline')

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')
sys.path.append('/home/ml/maruf/speed_detection/tracker/caffe/deep_sort/')

import caffe
import matplotlib.patches as patches
caffe.set_device(0)
caffe.set_mode_gpu()


# * Load LabelMap.

# In[4]:

from google.protobuf import text_format
from caffe.proto import caffe_pb2

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# load PASCAL VOC labels
labelmap_file = '/home/ml/maruf/speed_detection/tracker/caffe/data/VOC0712/labelmap_voc.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames


# * Load the net in the test phase for inference, and configure input preprocessing.

# In[ ]:

model_def = '/home/ml/maruf/speed_detection/tracker/caffe/models/VGGNet/models/VGGNet/VOC0712Plus/SSD_512x512_ft/deploy.prototxt'
model_weights = '/home/ml/maruf/speed_detection/tracker/caffe/models/VGGNet/models/VGGNet/VOC0712Plus/SSD_512x512_ft/VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


# ### 2. SSD detection

# * Load an image.

# In[10]:

# set net to batch size of 1
image_resize = 512
net.blobs['data'].reshape(1,3,image_resize,image_resize)


iter = 0
import glob

images = glob.glob('/home/ml/SR/darknet/data/Data/*.bmp')


max_cosine_distance = 0.8
nn_budget = 20

metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

import cv2
cap = cv2.VideoCapture('/home/ml/Desktop/Desktop/2.mp4')
print cap.isOpened()
exit()
while(cap.isOpened()):    
    ret, image = cap.read()
    cv2.imwrite('frame.jpg', image)
    image = caffe.io.load_image('frame.jpg')    

    # print image.shape

    
    # image = image / np.linalg.norm(image)
    # print image
# for image in images:

    # display_img = Image.open(image)
    # draw = ImageDraw.Draw(display_img)
    # image = caffe.io.load_image(image)
    plt.imshow(image)

    # plt.show(block=False)
    # plt.pause(0.1)
    # plt.cla()

    # continue


    # * Run the net and examine the top_k results

    # In[11]:

    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    
    
    detections = net.forward()['detection_out']
    features = net.blobs['conv10_2'].data

    print features.shape
    # continue

    # print features.shape, detections.shape
    # continue
    # print net.blobs.keys()
    # print net.blobs['mbox_conf'].data.shape
    # continue

    
    
    

    # Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]


    # * Plot the boxes

    # In[12]:

    # colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    # print colors
    # exit()

    # plt.imshow(image)
    currentAxis = plt.gca()

    tlwh = []
    confidence = []
    features = []
    detections = []

    for i in xrange(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * image.shape[1]))
        ymin = int(round(top_ymin[i] * image.shape[0]))
        xmax = int(round(top_xmax[i] * image.shape[1]))
        ymax = int(round(top_ymax[i] * image.shape[0]))

        if(xmin * xmax * ymin * ymax <= 0):continue
        if(xmax - xmin < 10 or ymax - ymin < 10): continue

        # print '>> ', ymax - ymin, xmax - xmin, (xmin, xmax, ymin, ymax)
        
        car = image[ymin:ymax, xmin:xmax, :]
        # print car.shape
        # plt.imshow(car)
        # plt.pause(0.1)
        # continue
        TI = transformer.preprocess('data', car)
        net.blobs['data'].data[...] = TI
        net_detections = net.forward()['detection_out']
        feature = net.blobs['fc7'].data.flatten()

        tlwh = np.array([xmin, ymin, xmax-xmin, ymax-ymin])
        confidence = 95.23
        
        # print '============================'
        # print tlwh
        # print confidence
        # print feature
    	detections.append(Detection(tlwh, confidence, feature))

    tracker.predict()
    tracker.update(detections)

    for track in tracker.tracks:
        if not track.is_confirmed or track.time_since_update > 0: continue
        coords = track.to_tlwh()
        if coords[2] < 5 or coords[3] < 5: continue
        # draw.rectangle(coords)
        # display_img.show()
        # print coords
        rect = patches.Rectangle((coords[0], coords[1]), coords[2], coords[3],linewidth=1,edgecolor='r',facecolor='none')
        currentAxis.add_patch(rect)
        currentAxis.text(coords[0], coords[1], str(track.track_id), bbox=dict(facecolor='red', alpha=0.5))
        
    plt.show(block=False)
    plt.pause(0.0001)
    plt.cla()


        # continue
    
    #     score = top_conf[i]
    #     label = int(top_label_indices[i])
    #     label_name = top_labels[i]
    #     display_txt = '%s: %.2f'%(label_name, score)
    #     coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
    #     color = colors[label]
    #     currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
    #     currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})

    # iter += 1
    # plt.show(block=False)
    # plt.savefig("/home/ml/Desktop/cars/" + str(iter) + ".jpg")
    # plt.pause(1)
    # plt.cla()


# In[ ]:



