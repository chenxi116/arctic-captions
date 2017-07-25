import numpy as np
from scipy.sparse import csr_matrix, vstack
import caffe
import time
import cPickle as pkl
import argparse
import pdb

def extract_features(dataset, setname, num_caps, img_path, gpu):
    
    txt_path = '../data/' + dataset + '/'

    caps = []
    with open(txt_path + dataset + '_' + setname + '_caps.txt', 'rb') as f:
        for line in f:
            caps.append(line.strip())
    caps = zip(caps, np.repeat(range(0, len(caps)), num_caps))

    img = []
    with open(txt_path + dataset + '_' + setname + '.txt', 'rb') as f:
        for line in f:
            img.append(line.strip())

    VGG_path = '../external/VGG/'
    model_def = VGG_path + 'VGG_ILSVRC_19_layers_deploy.prototxt'
    model_weights = VGG_path + 'VGG_ILSVRC_19_layers.caffemodel'
    net = caffe.Net(model_def, model_weights, caffe.TEST)

    mu = np.array((104.00698793, 116.66876762, 122.67891434))

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1)) # move image channels to outermost dimension
    transformer.set_mean('data', mu) # subtract mean
    transformer.set_raw_scale('data', 255) # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0)) # swap channels from RGB to BGR

    net.blobs['data'].reshape(1, 3, 224, 224)

    caffe.set_device(gpu)
    caffe.set_mode_gpu()

    feat_flatten_list = []
    for i in range(0, len(img)):
        imname = img[i]
        sl = time.time()
        image = caffe.io.load_image(img_path + imname)
        el = time.time()
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        net.forward()
        feat = net.blobs['conv5_4'].data[0].transpose((1, 2, 0)).flatten()
        ss = time.time()
        feat_flatten_list.append(csr_matrix(feat))
        es = time.time()
        print dataset + ' ' + setname + ' %d/%d load %fs stack %fs' %(i+1, len(img), el - sl, es - ss)

    feat_flatten = vstack(feat_flatten_list, format='csr')

    with open(txt_path + dataset + '_align.' + setname + '.pkl', 'wb') as f:
        pkl.dump(caps, f, protocol = pkl.HIGHEST_PROTOCOL)
        pkl.dump(feat_flatten, f, protocol = pkl.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str) # 'f30k' 'coco'
    parser.add_argument('-s', type=str) # 'train' 'dev' 'test'
    parser.add_argument('-n', type=int, default=5)
    parser.add_argument('-i', type=str) # '../external/flickr30k-center/'
    parser.add_argument('-g', type=int, default=0)

    args = parser.parse_args()
    extract_features(dataset = args.d, setname = args.s, num_caps = args.n, img_path = args.i, gpu = args.g)
