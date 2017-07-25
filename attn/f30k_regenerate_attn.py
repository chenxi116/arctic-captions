import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix
import cPickle as pkl
import argparse

def main(setname = 'train', sc = '10'):

    matDir = '../data/f30k/'
    mat = loadmat(matDir + 'f30k_attn_gt_' + sc + '.' + setname + '.mat')
    mat_idx = mat['attn_gt_idx']
    mat_map = mat['attn_gt_map']

    pkl_idx = []
    for i in range(len(mat_idx)):
        cap = mat_idx[i, 0].tolist()[0]
        pkl_idx.append(cap) # list

    pkl_map = csr_matrix(mat_map.astype('float32')) # numpy ndarray

    with open(matDir + 'f30k_attn_gt_' + sc + '.' + setname + '.pkl', 'wb') as f:
        pkl.dump(pkl_idx, f, protocol = pkl.HIGHEST_PROTOCOL)
        pkl.dump(pkl_map, f, protocol = pkl.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str, default='train')
    parser.add_argument('-c', type=str, default='10')

    args = parser.parse_args()
    main(setname = args.s, sc = args.c)