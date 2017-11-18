Dynamic evaluation for pytorch language models as implemented in Dynamic Evaluation of Neural Sequence Models (https://arxiv.org/abs/1709.07432). 

Requirements: python 3 (tested in 3.6), pytorch (tested in version 0.1.12)

Instructions for use:  

1. Train a language model using an existing repository, such as the pytorch language modelling tutorial https://github.com/pytorch/examples/tree/master/word_language_model . This should save a .pt file with the trained model

2. Copy the file dynamiceval.py into the repository

3. Run dynamic evaluation with: python dynamiceval.py --model modelname.pt


To replicate results in paper for AWD-LSTM + dynamic eval, train the language model using this repository https://github.com/salesforce/awd-lstm-lm . We used the original codebase from this repository, with the goal of exact replication of results from their paper (which we failed to achieve). The default settings and hyperparameters for dynamiceval.py are tuned for for AWD-LSTM + dynamic eval on PTB.

To get stronger results with any other model, you can run with:

python dynamiceval.py --model modelname.pt --grid

This will do a hyperparameter search on the validation set, takes a few hours on PTB

command line arguements:


--model (required)    -filename of the trained model to be evaluated

--data    -location of the data corpus

--grid    -hyperparameter grid search over lambda and eta, gives both valid and test error

--gridfast    -same as grid, but only uses first 30k validation tokens for search

--val    -measure validation error instead of test error  

--gpu    -specify a gpu device, uses device 0 by default (set negative for cpu)

--bptt    -sequence segment length for dynamic eval, also used for gradient statistics on training data

--batch_size    -batch size for gradient statistics on training data

--lr    -learning rate eta (ignored if --grid is set)

--lamb    -decay rate lambda (ignored if --grid is set)

--epsilon    -stabilization parameter epsilon

--max_batches  -max number of batches for training gradient statistics (-1 uses full training set)

--oldhyper

-The original version code inadvertently scaled a couple of terms differently from the equations described in the paper, which affects the hyperparameters. The code was changed to reflect the paper equations, and this flag applies a hyperparameter transformation in a way that accounts for this change. If you run this version of the code with the --oldhyper flag, it is equivalent to running the old version of the code. Some previous results that report hyperparameters with the old code would require applying this hyperparameter transformation to achieve the same results with this code.


