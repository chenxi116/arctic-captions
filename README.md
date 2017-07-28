# Attention Correctness in Neural Image Captioning

This branch `attn-corr` contains code for [Attention Correctness in Neural Image Captioning](https://arxiv.org/abs/1605.09553), AAAI 2017.

If you use this branch, please cite
```
@inproceedings{liu2017attention,
  title={Attention Correctness in Neural Image Captioning},
  author={Liu, Chenxi and Mao, Junhua and Sha, Fei and Yuille, Alan},
  booktitle={{AAAI}},
  year={2017}
}
```

See the `master` branch for original dependencies, reference, license etc.

## Dependencies

- Run `mkdir external`
- [Download](http://shannon.cs.illinois.edu/DenotationGraph/) or use symlink, such that the Flickr30k images are under `./external/flickr30k-images/`
- [Download](http://mscoco.org/dataset/#download) or use symlink, such chat MS COCO images are under `./external/coco/images/train2014/` and `./external/coco/images/val2014/`, and MS COCO annotations are under `./external/coco/annotations/`
- [Download](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77) or use symlink, such that `VGG_ILSVRC_19_layers_deploy.prototxt` and `VGG_ILSVRC_19_layers.caffemodel` are under `./external/VGG/`
- [Download](http://bplumme2.web.engr.illinois.edu/Flickr30kEntities/) or use symlink, such that the `Flickr30kEntities` folder is under `./external/`
- [Download](https://stanfordnlp.github.io/CoreNLP/) or use symlink, such that the `stanford-corenlp-full-2015-12-09` folder is under `./external/`
- [Download](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) or use symlink, such that `GoogleNews-vectors-negative300.bin` is under `./external/`
- Install [stanford_corenlp_pywrapper](https://github.com/brendano/stanford_corenlp_pywrapper)
- Install [gensim](https://radimrehurek.com/gensim/install.html)
- [MS COCO API](https://github.com/pdollar/coco)
- [coco-caption](https://github.com/tylin/coco-caption)

## Data Preparation

### Image deep features

- Modify and run `./attn/resize_centercrop.m` on COCO train2014, val2014, as well as Flickr30k 
- Extract `conv5_4` features from VGG 19
```
cd attn
python extract_features.py -d f30k -s train -i ../external/flickr30k-center/
python extract_features.py -d f30k -s dev -i ../external/flickr30k-center/
python extract_features.py -d f30k -s test -i ../external/flickr30k-center/
python extract_features.py -d coco -s train -i ../external/coco/images/train2014-center/
python extract_features.py -d coco -s dev -i ../external/coco/images/val2014-center/
python extract_features.py -d coco -s test -i ../external/coco/images/val2014-center/
```

### Strong supervision on Flickr30k

- Run `./attn/f30k_generate_attn.m` with setname `train`, `dev`, `test`
- Convert `mat` files to `pkl` files
```
cd attn
python f30k_regenerate_attn.py -s train
python f30k_regenerate_attn.py -s dev
python f30k_regenerate_attn.py -s test
```

### Weak supervision on COCO

- Run
```
cd attn
python coco_generate_attn.py -s train
python coco_generate_attn.py -s dev
python coco_generate_attn.py -s test
```

## Training

Edit Line 56 and 78 of `evaluate_flickr30k.py` or `evaluate_coco.py` if necessary. `attn-c` is `lambda` in the paper, which controls the relative strength of attention loss. Setting it to zero then the code falls back to Show, Attend and Tell. To initiate training, run
```
THEANO_FLAGS='device=gpu0,floatX=float32,on_unused_input='warn'' python evaluate_flickr30k.py 
THEANO_FLAGS='device=gpu1,floatX=float32,on_unused_input='warn'' python evaluate_coco.py
``` 

## Testing Captioning Performance

```
mkdir cap
mkdir cap/f30k
mkdir cap/coco
python generate_caps.py ./model/f30k/f30k_model_03.npz ./cap/f30k/f30k_03_k5 -d test -k 5
python metrics.py ./cap/f30k/f30k_03_k5.test.txt ref/30k/test/reference*
python generate_caps.py ./model/coco/coco_model_06.npz ./cap/coco/coco_06_k5 -d test -k 5
python metrics.py ./cap/coco/coco_06_k5.test.txt ref/coco/test/reference*
```

## Testing Attention Correctness