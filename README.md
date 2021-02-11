# Seq2Seq-pytorch
A Pytroch Seq2Seq model implementation for spelling checker

## Installation
<ul>
  <li>Install PyTorch by selecting your environment on the website and running the appropriate command.
  <li>Clone this repository
</ul>

## Datasets
There is some example of train dataset in /data directory
### data
<ul>
  <li>source.txt: list of words I want to fix 
  <li>target.txt: labels of each word in source.txt 
</ul>

## Training & Test
### train
<ol>
  <li> change the data paths and other hyperparameters in train.py
  <li> run python train.py
</ol>
### test
<ol>
  <li> change the data paths in test.py
  <li> run python test.py
</ol>

## Authors
Yejin Jeon

## reference
<ul>
  <li>https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
  <li>https://github.com/mdcramer/deep-learning/blob/master/seq2seq/sequence_to_sequence_implementation.ipynb
  <li>https://github.com/gaushh/Deep-Spelling/blob/master/deep_spell_GRU_attention_train.ipynb
</ul>
