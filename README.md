# Neural Machine Translation Between English and Turkish
Gorkem Ozkaya


This repo contains *a very early version* of a pair of NMT models between English and Turkish (in both directions). One can download the pre-trained models from the `releases` section and use the Jupyter notebook in the root directory as a reference. 

* TF1 version: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gorkemozkaya/nmt-en-tr/blob/master/Turkish_English_NMT.ipynb) 
* TF2 version: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/gorkemozkaya/nmt-en-tr/blob/master/Turkish_English_NMT_tf2.ipynb)

The models are trained on Google Cloud TPU's using the [tensor2tensor](https://github.com/tensorflow/tensor2tensor) library for the TF1 version, and with TensorFlow's official models library for the TF2 version.   As the neural network architechture, the  [Transformer](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) architecture is used with the [transformer_tpu](https://github.com/tensorflow/tensor2tensor/blob/c8fe559e0b357389d8754474e1306b6ca9afc4f3/tensor2tensor/models/transformer.py#L2576) hyperparameter configuration. 

## Acknowledgements
* TFRC Tensorflow Research Cloud program for cloud TPU hours 
* [Opus parallel corpus](http://opus.nlpl.eu) for making the Turkish/English parallel corpus available
* [Open Subtitles](http://www.opensubtitles.org) As being the original source of the movie subtitles parallel corpus. Also see 
```
P. Lison and J. Tiedemann, 2016, OpenSubtitles2016: Extracting Large Parallel Corpora from Movie and TV Subtitles. 
In Proceedings of the 10th International Conference on Language Resources and Evaluation (LREC 2016)
````
* [SETIMES](http://www.setimes.com) As the original source of the news articles corpus. 
