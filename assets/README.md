
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

intall pytorch-geometric following the instruction on [pyg installation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

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
