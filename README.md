#### Dynamic evaluation for pytorch language models as implemented in [Dynamic Evaluation of Neural Sequence Models](https://arxiv.org/abs/1709.07432). 

#### Requirements: python 3 (tested in 3.5, 3.6), pytorch (tested in 0.1.12, 0.2)

#### Instructions for use:  

1. Train a language model using an existing repository, such as the [pytorch language modeling tutorial](https://github.com/pytorch/examples/tree/master/word_language_model) . This should save a .pt file with the trained model

2. Copy the file dynamiceval.py into the repository

3. Run dynamic evaluation with: `python dynamiceval.py --model modelname.pt`

#### AWD-LSTM

To replicate results in paper for AWD-LSTM + dynamic eval, train the language model using the [Salesforce AWD-LSTM repository](https://github.com/salesforce/awd-lstm-lm). We used the original codebase from this repository, with the goal of exact replication of results from their paper (which we failed to achieve). The default settings and hyper-parameters for dynamiceval.py are tuned for for AWD-LSTM + dynamic eval on PTB. 

#### AWD-QRNN

This code also supports the [pytorch QRNN](https://github.com/salesforce/pytorch-qrnn) with the --QRNN option. AWD-QRNN + dynamic eval obtains very similar results to AWD-LSTM + dynamic eval, and is much faster to train and evaluate. Training an AWD-QRNN on PTB using the Salesforce AWD-LSTM repository, and running dynamic eval with the default settings gives a test perplexity of 50.5. Increasing the sequence segment length from 5 to 20 runs 3x faster (1 minute vs. 3 minutes on PTB), and gives validation (use --val flag) and test perplexities of `51.4/50.5` with the following arguments:

`python dynamiceval.py --model PTB.pt --QRNN --lr 0.00012 --lamb 0.02 --bptt 20`

AWD-QRNN trained on wikitext-2 gives validation (use --val flag) and test perplexities of `45.9/44.0` with the following arguments:

`python dynamiceval.py --model WT2.pt --QRNN --lr 0.00012 --lamb 0.008 --bptt 20 --data data/wikitext-2`

#### Hyper-parameter search

To get stronger results with any other model or dataset, you can run with:

`python dynamiceval.py --model modelname.pt --grid`

This will do a hyper-parameter search on the validation set, takes a few hours on PTB with LSTM and default settings, can be much faster with QRNN and/or larger --bptt. If the default model size/settings are changed too much, the hyper-parameters in `lrlist` and `lamblist` may need to be changed for best results. If you want to do a faster search, you can try running with --gridfast to use a subset of the validation set, or you can reduce the number of elements in lamblist (tuning lr is more important).

#### Command line arguments:


`--model` (required)    -filename of the trained model to be evaluated

`--data`    -location of the data corpus

`--grid`    -hyper-parameter grid search over lambda and eta, gives both valid and test error

`--gridfast`    -same as grid, but only uses first 30k validation tokens for search

`--val `   -measure validation error instead of test error  

`--gpu`    -specify a gpu device, uses device 0 by default (set negative for cpu)

`--QRNN`    -apply dynamic eval to a QRNN

`--bptt`    -sequence segment length for dynamic eval, also used for gradient statistics on training data

`--batch_size`    -batch size for gradient statistics on training data

`--lr`    -learning rate eta (ignored if --grid is set)

`--lamb`    -decay rate lambda (ignored if --grid is set)

`--epsilon`    -stabilization parameter epsilon

`--max_batches`  -max number of batches for training gradient statistics (-1 uses full training set)

`--oldhyper`

-The original version code inadvertently scaled a couple of terms differently from the equations described in the paper, which affects the hyper-parameters. The code was changed to reflect the paper equations, and this flag applies a hyper-parameter transformation in a way that accounts for this change. If you run this version of the code with the --oldhyper flag, it is equivalent to running the old version of the code. This will also print out hyper-parameter values that can be used with the new version of the code (without this flag) to obtain the same results. Some previous results that report hyper-parameters with the old code would require applying this hyper-parameter scaling to achieve the same results with this code. The following replicates the exact settings used to obtain results for AWD-LSTM in the paper for PTB:

`python dynamiceval.py --model PTB.pt --lr 0.002 --lamb 0.02 --epsilon 0.001 --oldhyper`

and for wikitext-2:

`python dynamiceval.py --model WT2.pt --lr 0.002 --lamb 0.02 --epsilon 0.001 --oldhyper --data data/wikitext-2`


Note that while the original hyper-parameters are the same for Wikitext-2 and PTB, the new/scaled hyper-parameters are a little different.
