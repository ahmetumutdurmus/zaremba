# Recurrent Neural Network Regularization by Zaremba et al. (2014).
This repository contains the replication of "Recurrent Neural Network Regularization" by Zaremba et al. (2014).

It is one of the earliest successful applications of Dropout on RNNs and had achieved state-of-the-art results on word-level language modeling task on Penn Treebank dataset back in its day. Its best single model, Large Regularized LSTM, achieves a test perplexity of ~78.4 after 55 epochs of training. 

The original paper can be found at: [https://arxiv.org/abs/1409.2329](https://arxiv.org/abs/1409.2329)  
While the original code written in Lua and Torch can be found at: [https://github.com/wojzaremba/lstm](https://github.com/wojzaremba/lstm)

I have replicated the paper using Python 3.7 and PyTorch 1.2 with CUDA Toolkit 10.0. 

The repository contains three scripts:

+ `model.py` contains the model described as in the paper.
+ `main.py` is used to replicate the main results in the paper. 
+ `ensemble.py` is used to replicate the model averaging results in the paper. 

## Experiments
There are three models presented in the paper and each can be replicated from the terminal as follows:

### Non-Regularized LSTM
+ `python main.py --layer_num 2 --hidden_size 200 --lstm_type pytorch --dropout 0.0 --winit 0.1 --batch_size 20 --seq_length 20 --learning_rate 1 --total_epochs 13 --factor_epoch 4 --factor 2 --max_grad_norm 5 --device gpu`

### Medium Regularized LSTM
+ `python main.py --layer_num 2 --hidden_size 650 --lstm_type pytorch --dropout 0.5 --winit 0.05 --batch_size 20 --seq_length 35 --learning_rate 1 --total_epochs 39 --factor_epoch 6 --factor 1.2 --max_grad_norm 5 --device gpu`

### Large Regularized LSTM
+ `python main.py --layer_num 2 --hidden_size 1500 --lstm_type pytorch --dropout 0.65 --winit 0.04 --batch_size 20 --seq_length 35 --learning_rate 1 --total_epochs 55 --factor_epoch 14 --factor 1.15 --max_grad_norm 10 --device gpu`

Note that you can use both my implementation of LSTM by setting `--lstm_type custom` or the PyTorch's embedded C++ implementation using `--lstm_type pytorch`. PyTorch's implementation is about 2 times faster. 

## Model Averaging
The paper also presents model averaging results for the three models described above. The averaging method chosen is a naive implementation. Simple arithmetic means between each model's probability predictions are the ensembling outputs. These results can be replicated from the terminal again as follows:

### 2 Non-Regularized LSTMs
+ `python ensemble.py --ensemble_num 2 --layer_num 2 --hidden_size 200 --lstm_type pytorch --dropout 0.0 --winit 0.1 --batch_size 20 --seq_length 20 --learning_rate 1 --total_epochs 13 --factor_epoch 4 --factor 2 --max_grad_norm 5 --device gpu`

### 5 Medium Regularized LSTMs
+ `python ensemble.py --ensemble_num 5 --layer_num 2 --hidden_size 650 --lstm_type pytorch --dropout 0.5 --winit 0.05 --batch_size 20 --seq_length 35 --learning_rate 1 --total_epochs 39 --factor_epoch 6 --factor 1.2 --max_grad_norm 5 --device gpu`

### 10 Large Regularized LSTMs
+ `python ensemble.py --ensemble_num 10 --layer_num 2 --hidden_size 1500 --lstm_type pytorch --dropout 0.65 --winit 0.04 --batch_size 20 --seq_length 35 --learning_rate 1 --total_epochs 55 --factor_epoch 14 --factor 1.15 --max_grad_norm 10 --device gpu`

So one only needs to run `ensemble.py` instead of `main.py` with the addition of `--ensemble_num <INT>` to specify the number of models to average. 
