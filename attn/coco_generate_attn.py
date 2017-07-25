from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from stanford_corenlp_pywrapper import CoreNLP
from scipy.sparse import csr_matrix
import gensim
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt 
import argparse
import cv2
import cPickle as pkl
import pdb

def GenAttn(setname, thres, visualize = False):

    if setname == 'dev' or setname == 'test':
        dataType = 'val2014'
    else:
        dataType = 'train2014'

    scene = ['library', 'church', 'office', 'restaurant', 'kitchen', 'room', 
            'bathroom', 'factory', 'campus', 'bedroom', 'auditorium', 'shop', 
            'home', 'hotel', 'classroom', 'cafeteria', 'hospital', 'street', 
            'park', 'beach', 'river', 'village', 'valley', 'market', 'harbor', 
            'yard', 'lighthouse', 'railway', 'playground', 'forest', 'station', 
            'garden', 'farm', 'mountain', 'plaza', 'dining', 'living']

    # load image names and captions
    imgs = [] # list containing image names
    with open('../data/coco/coco_' + setname + '.txt', 'rb') as f:
        for line in f:
            imgs.append(line.strip())
    caps = [] # list containing captions
    with open('../data/coco/coco_' + setname + '_caps.txt', 'rb') as f:
        for line in f:
            caps.append(line.strip())

    toolDir = '../external'
    # initialize stanford parser
    proc = CoreNLP("parse", corenlp_jars=[toolDir + '/stanford-corenlp-full-2015-12-09/*'])
    # load Word2Vec model
    model = gensim.models.Word2Vec.load_word2vec_format(toolDir + '/GoogleNews-vectors-negative300.bin', binary=True)

    # load COCO annotations
    dataDir = '../external/coco'
    # dataType = 'val2014'
    annFile = '%s/annotations/instances_%s.json' %(dataDir, dataType)
    coco = COCO(annFile)

    caps_attn = []
    maps_attn = []
    map_idx = 1

    coef = 9. # coefficient for visualization
    # thres = 0.25 # threshold for word similarity

    for i in range(len(imgs)): # for every image
    # for i in range(100):
        imgname = imgs[i]
        imgid = int(imgname.split('_')[2].split('.')[0])
        imginfo = coco.loadImgs(imgid)[0]

        annIds = coco.getAnnIds(imgIds = imgid)
        anns = coco.loadAnns(annIds)

        if visualize:
            I = io.imread('%s/images/%s/%s'%(dataDir, dataType, imgname))
            plt.figure(1)
            plt.clf()
            plt.imshow(I)
            coco.showAnns(anns) 

            I_c = io.imread('%s/images/%s-center/%s'%(dataDir, dataType, imgname))
            plt.figure(2)
            plt.imshow(I_c)

        for j in range(5): # for every caption associated with the image
            cap = caps[5 * i + j]
            words = cap.split(' ')
            cap_attn = [0]*len(words)
            noun_pos = GetNouns(cap, proc) # get noun positions
            for noun in noun_pos: # for every noun
                if noun >= len(words) or not (words[noun].lower() in model) or words[noun] in scene:
                    continue
                mask = np.zeros([224, 224])
                for ann in anns: # for every annotation
                    cat = coco.loadCats(ann['category_id'])[0]
                    cat_w = cat['name'].split(' ') # the category name could be e.g. 'traffic light'
                    sim = []
                    for w in cat_w:
                        sim.append(model.similarity(words[noun].lower(), w))
                    # print 'word: %s category: %s sim: %f' %(words[noun].lower(), cat['name'], max(sim))

                    # bbox = AdjustBbox(ann['bbox'], imginfo)
                    # mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = max(sim) # pick the highest score
                    
                    if max(sim) < thres: # 'boy' and 'person' is 0.337
                        continue
                    mm = AdjustMask(ann, imginfo)
                    mask[np.nonzero(mm)] = max(sim)

                mask_d = cv2.resize(mask, (14, 14))
                mask_s = sum(sum(mask_d))
                mask_d = mask_d/mask_s
                if np.isnan(sum(sum(mask_d))) or np.isinf(sum(sum(mask_d))):
                    continue

                if visualize:
                    mask_u = cv2.resize(mask_d, (224, 224))
                    max_u = mask_u.max()
                    mask_u = mask_u/max_u
                    m = mask_u*255*(coef-1)/coef
                    I_m = np.dstack((m, m, m))
                    I_attn = (I_c.astype('float32')/coef + I_m).astype('uint8')
                    plt.figure(3)
                    plt.imshow(I_attn)
                    plt.title(words[noun])
                    pdb.set_trace()

                cap_attn[noun] = map_idx
                map_idx += 1
                maps_attn.append(mask_d.flatten()) # DOUBLE CHECK THIS!

            caps_attn.append(cap_attn)

    maps_attn = np.array(maps_attn)
    mask_attn = csr_matrix(maps_attn.astype('float32'))

    print 'Nonzero Percentage: %f' %(len(maps_attn.nonzero()[0])/(maps_attn.size*1.0))

    with open('../data/coco/coco_attn_gt.' + setname + '.pkl', 'wb') as f:
        pkl.dump(caps_attn, f, protocol = pkl.HIGHEST_PROTOCOL)
        pkl.dump(mask_attn, f, protocol = pkl.HIGHEST_PROTOCOL)


def AdjustMask(ann, imginfo):
    h = imginfo['height']
    w = imginfo['width']
    r = min(h/256., w/256.)
    h_r = int(round(h/r))
    w_r = int(round(w/r))

    if type(ann['segmentation']) == list:
        # polygon
        rle = cocomask.frPyObjects(ann['segmentation'], h, w)
    else:
        # mask
        if type(ann['segmentation']['counts']) == list:
            rle = cocomask.frPyObjects([ann['segmentation']], h, w)
        else:
            rle = [ann['segmentation']]
    m = cocomask.decode(rle).sum(2) # h * w * c -> h * w 
    # it's the nonzero elements that matter, so it's safe to do sum(2)

    m_r = cv2.resize(m, (w_r, h_r), interpolation=cv2.INTER_NEAREST) # resize
    h_s = int(round((h_r - 224)/2))
    w_s = int(round((w_r - 224)/2))
    m_c = m_r[h_s:h_s + 224, w_s:w_s + 224] # center crop
    return m_c


def AdjustBbox(bbox, imginfo):
    # bbox is [x, y, width, height]
    assert len(bbox) == 4, 'More than one bbox'
    bbox[2] += bbox[0]
    bbox[3] += bbox[1]
    h = imginfo['height']
    w = imginfo['width']
    r = min(h/256., w/256.)
    h_r = h/r
    w_r = w/r
    bbox_r = [x/r for x in bbox]

    bbox_ret = []
    bbox_ret.append(round(max(0, min(224, bbox_r[0] - (w_r/2.-112)))))
    bbox_ret.append(round(max(0, min(224, bbox_r[1] - (h_r/2.-112)))))
    bbox_ret.append(round(max(0, min(224, bbox_r[2] - (w_r/2.-112)))))
    bbox_ret.append(round(max(0, min(224, bbox_r[3] - (h_r/2.-112)))))
    return bbox_ret


def GetNouns(sentence, proc):
    # return the positions of nouns in the sentence
    p = proc.parse_doc(sentence)
    if len(p['sentences']) < 1:
        pdb.set_trace()
    pos = p['sentences'][0]['pos']
    noun_pos = [idx for idx, noun in enumerate(pos) if noun == 'NN']
    return noun_pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, default='train') # 'train' 'dev' 'test'
    parser.add_argument('-t', type=float, default=1./3)
    parser.add_argument('-v', type=bool, default=False) 

    args = parser.parse_args()
    GenAttn(setname = args.s, thres = args.t, visualize = args.v)