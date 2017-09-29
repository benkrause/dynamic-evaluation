Dynamic evaluation for pytorch language models as implemented in Dynamic Evaluation of Neural Sequence Models (https://arxiv.org/abs/1709.07432). 

Requirements: python 3 (tested in 3.6), pytorch (tested in version 0.1.12)

Instructions for use:  

1. Train a language model using an existing repository, such as the pytorch language modelling tutorial https://github.com/pytorch/examples/tree/master/word_language_model . This should save a .pt file with the trained model

2. Copy the file dynamiceval.py into the repository

3. run dynamic evaluation with: python dynamiceval.py --model modelname.pt


To replicate results in paper for AWD-LSTM + dynamic eval, train the language model using this repository https://github.com/salesforce/awd-lstm-lm . We used the original codebase from this repository, with the goal of exact replication of results from their paper (which we failed to achieve). The default settings and hyperparameters for dynamiceval.py are the same as used for results in paper for AWD-LSTM + dynamic eval on both wikitext-2 and PTB.


command line arguements:


--model (required)    -filename of the trained model to be evaluated

--data    -location of the data corpus

--val    -measure validation error instead of test error (use for hyperparameter tuning)

--gpu    -specify a gpu device, uses device 0 by default (set negative for cpu)

--bptt    -sequence segment length for dynamic eval, also used for gradient statistics on training data

--batch_size    -batch size for gradient statistics on training data

--lr    -learning rate eta

--lamb    -decay rate lambda

--epsilon    -stabilization parameter epsilon

--max_batches  -max number of batches for training gradient statistics (-1 uses full training set)

--ms

-The code used to obtain results in the paper for AWD-LSTMs inadvertently used the root sum squared gradients in place of the root mean squared gradients from the training data (this is equation 6 in the paper). Using root sum squared gradients and root mean squared gradients are equivalent under a hyperparameter transformation. By default, this code uses sum of squared gradients on training data, but the --ms option can be used for mean squared gradients, which is consistent with the equations in the paper. Using the --ms flag and scaling the hyperparameters --lr and --epsilon each by (1/sqrt(N)), where N is the number of training batches, yields equivalent results to not using the --ms flag.





