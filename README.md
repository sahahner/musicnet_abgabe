# Musicnet: Implementation of translation-invariant neural network and bidirectional recurrent Network

Master Project WS 2018/19 University of Bonn

Based on work from John Thickstun (2017/18).

## The Dataset: MusicNet

The dataset and the Data Augmentation methods are demonstrated in [01_music_demonstration_py27.ipynb](https://github.com/sahahner/musicnet_abgabe/blob/master/01_music_demonstration_py27.ipynb)

The pitch differences between adjacent windows (as presented in the documentation) where calculated by 
[02_plot_pitch_differences_whole_set_p3.ipynb](https://github.com/sahahner/musicnet_abgabe/blob/master/02_plot_pitch_differences_whole_set_p3.ipynb)

## Translation-Invariant Neural Network

Givent the work of Thikstun and the dataset MusicNet (both linked on the homepage of [MusicNet](https://homes.cs.washington.edu/~thickstn/musicnet.html))
his translation-invariant neural network from the paper [Invariances and Data Augmentation for Supervised Music Transcription](https://arxiv.org/abs/1711.04845) by John Thickstun, Zaid Harchaoui, Dean P. Foster and Sham M. Kakade was implemented.

Also I tested regularization and the sigmoid function as the activation function of the output layer.

The four tested version are (every version has an on ipython notebook):
- Replica of authors version [190312_3layer-trans-inv-REPLICA.ipynb](https://github.com/sahahner/musicnet_abgabe/blob/master/190312_3layer-trans-inv-REPLICA.ipynb)
- Replica trained using parameter-norm penalty [190311_3layer-trans-inv-reg.ipynb](https://github.com/sahahner/musicnet_abgabe/blob/master/190311_3layer-trans-inv-reg.ipynb)
- Sigmoid function in output layer [190314_3layer-trans-inv-sigmoid-noreg.ipynb](https://github.com/sahahner/musicnet_abgabe/blob/master/190314_3layer-trans-inv-sigmoid-noreg.ipynb)
- Sigmoid function in output + parameter-norm penalty [190314_3layer-trans-inv-sigmoid-reg.ipynb](https://github.com/sahahner/musicnet_abgabe/blob/master/190314_3layer-trans-inv-sigmoid-reg.ipynb)


## Bidirectional Neural Network

An LSTM-cell was included after the feed-forward layer in its original version. 
Different hyperparameters were tested for the bidirectional recurrent neural network.

The versions of the bidirectional recurrent neural networl are defined in the [versions-file](https://github.com/sahahner/musicnet_abgabe/blob/master/lib/versions_norm.py).
When only interested in the results, this [ipython notebook](https://github.com/sahahner/musicnet_abgabe/blob/master/05_3layer-LSTM-norm_results.ipynb)
illustrates them nicely. In the beginning the selected version has to be chosen as named in the versions-file.

The training can be done using this [ipyhton notebook](https://github.com/sahahner/musicnet_abgabe/blob/master/05_3layer-LSTM-norm_train.ipynb).

## Visualizations

Results are saved in the directory [results](https://github.com/sahahner/musicnet_abgabe/tree/master/results). 
The plots, included in the documentation, are also saved here.

## Documentation

Finally, the final report and presentation can be found in the directory [documentation](https://github.com/sahahner/musicnet_abgabe/tree/master/documentation).
