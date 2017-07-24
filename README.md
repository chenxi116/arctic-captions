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

## Data Preparation

- Modify and run `./prep/resize_centercrop.m` on COCO train2014, val2014, as well as Flickr 30k 
- Download or use symlink, such that `VGG_ILSVRC_19_layers_deploy.prototxt` and `VGG_ILSVRC_19_layers.caffemodel` are under `./VGG/`
- Extract `conv5_4` features from VGG 19
```
cd prep
python extract_features.py -d coco -s train -i /your/path/to/train2014-center/
python extract_features.py -d coco -s dev -i /your/path/to/val2014-center/
python extract_features.py -d coco -s test -i /your/path/to/val2014-center/
python extract_features.py -d f30k -s train -i /your/path/to/flickr30k-center/
python extract_features.py -d f30k -s dev -i /your/path/to/flickr30k-center/
python extract_features.py -d f30k -s test -i /your/path/to/flickr30k-center/
```

## Training

## Testing