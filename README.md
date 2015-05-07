# arctic-captions

Source code for [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](http://arxiv.org/abs/1502.03044)
runnable on cpu and gpu.

## License

Code is released under the [revised (3-clause) BSD
License](http://directory.fsf.org/wiki/License:BSD_3Clause). If you use this
code as part of any published research, please acknowledge the following paper. 
(It does encourages researchers who publish their code!)

**"Show, Attend and Tell: Neural Image Caption Generation with Visual Attention."**  
Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan
Salakhutdinov, Richard Zemel, Yoshua Bengio.   
University of Toronto & Université de Montréal, 
To appear in ICML (2015)

[[bibtex coming soon]()]

## Dependencies

This code is written in python, to use it you will need:

* Python 2.7
* A relative recent version of [NumPy](http://www.numpy.org/)
* [scikit learn](http://scikit-learn.org/stable/index.html)
* [argparse](https://www.google.ca/search?q=argparse&oq=argparse&aqs=chrome..69i57.1260j0j1&sourceid=chrome&es_sm=122&ie=UTF-8#q=argparse+pip)

In addition, this code is built using
[Theano](http://www.deeplearning.net/software/theano/). If you encounter
problems with specific to Theano, please use a commit from around February 2015
and notify the authors.

To use the evaluation script: see [coco-caption](https://github.com/tylin/coco-caption)

###Installation + Experiment Instructions
1) Install the above dependencies and `$ git clone` the repo  
2) Install Theano using your [favourite method](http://www.deeplearning.net/software/theano/)   
3) TODO, rest of this
