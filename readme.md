## Fast power spectrum interpolation


### Files & Usage

```
├── requirements.txt : install python dependencies : pip install -r requirements.txt
├── dataset_parallel.py : generate the power spectrum dataset for training & evaluation
├── data_serialization.py : load data
├── model.py : contains models
├── train.py : trains the power spectrum model
├── eval.py : evaluate a set of input parameters
├── weights : directory with the necessary model weight files for saving/loading
│   ├── lstm_.._.h5
│   ├── ae_..._e0_..._.h5
│   ├── ae_..._e1_..._.h5
│   └── ae_..._e2_..._.h5
└── readme.md : describing this text.
```

#### Generating data

1. Run the script `dataset_parallel.py` and set mode `mode = 'parallel_search'`, export directory `export_dirfilename` and dataset index `iter_index` (this refers to the subset of the dataset to compute on a single node; run a different index on each different node) to generate the `.csv` files in parallel.
2. Combine the subset `.csv` in `./data` directory into single `.csv` by running `dataset_parallel.py` with set mode `mode = 'combine_dataframe'`
3. Convert to SQL for faster read/write:
```
from data_serialization import *
generation(iteration = 889100)
```

#### Loading data

Loading training data:

```
from data_serialization import *
train_unique, test_unique = train_test_params(iteration = 889100, train_test_split_iter = 10, denomination = 100)
train_set_expl = data_load(train_unique, iteration = 889100)
x_train, y_train = model_inputs(train_set_expl)
```

Loading evaluation data:

```
from data_serialization import *
eval_set_expl = data_load("eval", iteration = 889100)
x_eval, y_eval = model_inputs(eval_set_expl)

```

#### Training

```
from train import *
train(x_train, y_train,
          cycles = 40, batch_size = 1000, learning_rate = 0.001, 
          RNN__fname = "lstm6_tts10_bs1000", lstm_pretrained = True, lstm_pretrained_fname = 'lstm5_epch240_tts10_bs1000_129_17759.h5',
          k1 = 8e-7, k2 = 0.008, ae_epochs = 40,
          ae_e0_fname = "ae7_e0_tts10_bs100.h5", ae0_pretrained = False, ae0_pretrained_fname = '',
          ae_e1_fname = "ae7_e1_tts10_bs100.h5", ae1_pretrained = False, ae1_pretrained_fname = '',
          ae_e2_fname = "ae7_e2_tts10_bs100.h5", ae2_pretrained = False, ae2_pretrained_fname = '',
         )
```

#### Interpolation

```
from eval import *
x_eN, x_pre_e0, x_pre_e1, x_pre_e2, P_k_ae_e1, P_k_ae_e2, P_k_ae_e3 = prediction(x_eval, y_eval,
               lstm_pretrained = True, lstm_pretrained_fname = 'lstm5_epch240_tts10_bs1000_129_17759.h5',
               ae0_pretrained = True, ae1_pretrained = True, ae2_pretrained = True, 
               ae_e0_fname = 'ae6_e0_tts10_bs100.h5', ae_e1_fname = 'ae6_e1_tts10_bs100.h5', ae_e2_fname = 'ae3_e2_tts10_bs100.h5',
               k1 = 8e-7, k2 = 0.008, 
              )
```

Predicted power spectrum value is stored in `x_eN['P_k_ae']`. 

