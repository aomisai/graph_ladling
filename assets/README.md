
## Requirements

These steps are for easy testing the script. You should use appropiate Torch/CUDA versions for your environment. 
```bash
conda create -n $ladling_env1
conda activate $ladling_env1
```

install pytorch following the instruction on [pytorch installation](https://pytorch.org/get-started/locally/)

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

install pytorch-geometric following the instruction on [pyg installation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

For Torch scatter and sparse, you'll need Microsoft C++ Build tools. Make sure you get both Visual Studio Build Tools 
and VIsual Studio Community. For Visual Studio Community, make sure you have installed Desktop Development with C++

```bash
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu113.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu113.html
pip install torch-geometric
```

install the other dependencies

```bash
pip install ogb # package for ogb datasets
pip install texttable # show the running hyperparameters
pip install h5py # for Label Propagation
```

## Train our soup ingredients and save the model state


```bash
python main.py --cuda_num=0  --type_model=$type_model --dataset=$dataset
# type_model in ['GraphSAGE', 'FastGCN', 'LADIES', 'ClusterGCN', 'GraphSAINT', 'SGC', 'SIGN', 'SIGN_MLP', 'LP_Adj', 'SAGN', 'GAMLP']
# dataset in ['Flickr', 'Reddit', 'Products', 'Yelp', 'AmazonProducts']
```

## Command Line Arguments

Below is a list of the available command-line arguments, their descriptions, and default values:

| Argument                     | Type    | Description                                                                                                                                                 | Default Value                      |
|------------------------------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------|
| `--debug_mem_speed`          | Flag    | Whether to get memory usage and throughput                                                                                                                  | `False`                            |
| `--debug`                    | Flag    | Enable debugging mode                                                                                                                                       | `False`                            |
| `--save_ing`                 | Flag    | Save soup ingredients for soup                                                                                                                              | `False`                            |
| `--tosparse`                 | Flag    | Convert data to sparse format                                                                                                                               | `False`                            |
| `--dataset`                  | String  | The dataset to use (choices: `Flickr`, `Reddit`, `ogbn-products`, `ogbn-papers100M`, `ogbn-arxiv`, `Cora`)                                                  | **Required**                      |
| `--type_model`               | String  | Type of model to run (choices: `GraphSAGE`, `FastGCN`, `LADIES`, `ClusterGCN`, `GraphSAINT`, `SGC`, `SIGN`, `SIGN_MLP`, `LP_Adj`, `SAGN`, `GAMLP`, `EnGCN`) | **Required** |
| `--exp_name`                 | String  | Name of the experiment                                                                                                                                      | `""` (empty string)                |
| `--N_exp`                    | Integer | Number of experiment runs                                                                                                                                   | `20`                               |
| `--resume`                   | Flag    | Resume the previous experiment                                                                                                                              | `False`                            |
| `--cuda`                     | Boolean | Whether to run on CUDA (GPU)                                                                                                                                | `True`                             |
| `--cuda_num`                 | Integer | GPU device number                                                                                                                                           | `0`                                |
| `--num_layers`               | Integer | Number of layers in the model                                                                                                                               | `2`                                |
| `--epochs`                   | Integer | Number of epochs to train                                                                                                                                   | `50`                               |
| `--eval_steps`               | Integer | Interval steps to evaluate the model performance                                                                                                            | `5`                                |
| `--multi_label`              | Boolean | Whether the task is multi-label or single-label                                                                                                             | `False`                            |
| `--dropout`                  | Float   | Dropout rate for input features                                                                                                                             | `0.2`                              |
| `--norm`                     | String  | Normalization technique to use                                                                                                                              | `"None"`                           |
| `--lr`                       | Float   | Learning rate                                                                                                                                               | `0.001`                            |
| `--weight_decay`             | Float   | Weight decay for optimization                                                                                                                               | `0.0`                              |
| `--dim_hidden`               | Integer | Dimension of hidden layers                                                                                                                                  | `128`                              |
| `--batch_size`               | Integer | Batch size                                                                                                                                                  | `20000`                            |
| `--walk_length`              | Integer | Walk length for random walk sampler (used in GraphSAINT)                                                                                                    | `2`                                |
| `--num_steps`                | Integer | Number of steps for the model                                                                                                                               | `5`                                |
| `--sample_coverage`          | Integer | Sample coverage                                                                                                                                             | `0`                                |
| `--use_norm`                 | Boolean | Whether to use normalization                                                                                                                                | `False`                            |
| `--num_parts`                | Integer | Number of parts (for ClusterGCN)                                                                                                                            | `1500`                             |
| `--dst_sample_coverage`      | Float   | Destination sampling rate                                                                                                                                   | `0.1`                              |
| `--dst_walk_length`          | Integer | Random walk length for destination sampling                                                                                                                 | `2`                                |
| `--dst_update_rate`          | Float   | Rate of destination updates                                                                                                                                 | `0.8`                              |
| `--dst_update_interval`      | Integer | Interval for destination updates                                                                                                                            | `1`                                |
| `--dst_T_end`                | Integer | End of destination updates                                                                                                                                  | `250`                              |
| `--dst_update_decay`         | Boolean | Whether to apply decay to the update rate                                                                                                                   | `True`                             |
| `--dst_update_scheme`        | String  | Update scheme for destination sampling                                                                                                                      | `"node3"`                          |
| `--dst_grads_scheme`         | Integer | Gradient scheme for destination sampling                                                                                                                    | `3`                                |
| `--LP__no_prep`              | Integer | Label propagation no prep                                                                                                                                   | `0`                                |
| `--LP__pre_num_propagations` | Integer | Number of propagations for pre-training                                                                                                                     | `10`                               |
| `--LP__A1`                   | String  | Adjacency matrix (choices: `DA`, `AD`, `DAD`)                                                                                                               | `"DA"`                             |
| `--LP__A2`                   | String  | Adjacency matrix for second layer (choices: `DA`, `AD`, `DAD`)                                                                                              | `"AD"`                             |
| `--LP__prop_fn`              | Integer | Propagation function                                                                                                                                        | `1`                                |
| `--LP__num_propagations1`    | Integer | Number of propagations for the first layer                                                                                                                  | `50`                               |
| `--LP__num_propagations2`    | Integer | Number of propagations for the second layer                                                                                                                 | `50`                               |
| `--LP__alpha1`               | Float   | Alpha value for the first propagation                                                                                                                       | `0.9791632871592579`               |
| `--LP__alpha2`               | Float   | Alpha value for the second propagation                                                                                                                      | `0.7564990804200602`               |
| `--LP__num_layers`           | Integer | Number of layers for Label Propagation                                                                                                                      | `3`                                |
| `--SLE_threshold`            | Float   | Threshold for SLE                                                                                                                                           | `0.9`                              |
| `--num_mlp_layers`           | Integer | Number of MLP layers                                                                                                                                        | `3`                                |
| `--use_batch_norm`           | Boolean | Whether to use batch normalization                                                                                                                          | `True`                             |
| `--num_heads`                | Integer | Number of heads (for multi-head attention)                                                                                                                  | `1`                                |
| `--use_label_mlp`            | Boolean | Whether to use MLP for label propagation                                                                                                                    | `True`                             |
| `--GAMLP_type`               | String  | Type of GAMLP model (choices: `JK`, `R`)                                                                                                                    | `"JK"`                             |
| `--GAMLP_alpha`              | Float   | Alpha value for GAMLP                                                                                                                                       | `0.5`                              |
| `--GPR_alpha`                | Float   | Alpha value for GPR                                                                                                                                         | `0.1`                              |
| `--GPR_init`                 | String  | Initialization for GPR (choices: `SGC`, `PPR`, `NPPR`, `Random`, `WS`, `Null`)                                                                              | `"PPR"`          |
| `--type_run`                 | String  | Type of run (choices: `complete`, `filtered`)                                                                                                               | `"filtered"`                       |
| `--filter_rate`              | Float   | Filter rate for run                                                                                                                                         | `0.2`                              |

### Notes:
- Flags like `--save_ing`, `--debug_mem_speed`, `--debug`, `--tosparse`, and `--resume` do not require a value. They are either enabled or disabled when provided.
- Certain parameters like `--dataset` and `--type_model` are required and must be specified.



## Perform model soup across the soup ingredients using linear interpolation 

```
def merge_model(state_dicts):
    alphal = [1/len(state_dicts) for i in range(0, len(state_dicts))]
    sd = {}
    for k in state_dicts[0].keys():
        sd[k] = state_dicts[0][k].clone() * alphal[0]

    for i in range(1, len(state_dicts)):
        for k in state_dicts[i].keys():
            sd[k] = sd[k] + state_dicts[i][k].clone() * alphal[i]

    return sd


def interpolate(state1, state2, model, data, split_idx, evaluator):
    alpha = np.linspace(0, 1, <granularity of interpolation>)
    max_val,  loc = -1,  -1
    for i in alpha:
        sd = {}
        for k in state1.keys():
            sd[k] = state1[k].clone() * i + state2[k].clone() * (1 - i)
        model.load_state_dict(sd)
        train_acc, valid_acc, test_acc = test(model, data, split_idx, evaluator)
        if valid_acc > max_val:
            max_val = valid_acc
            loc = i
    sd = {}
    for k in state1.keys():
        sd[k] = state1[k].clone() * loc + state2[k].clone() * (1 - loc)
    return max_val, loc, sd
```
