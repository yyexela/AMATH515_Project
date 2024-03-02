#!/bin/bash

mkdir ./Datasets
mkdir ./Saved_Models
curl https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o ./Datasets/tiny-shakespeare.txt
