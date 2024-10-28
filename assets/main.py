import gc
import json
import os
import random
from datetime import datetime
import numpy as np
import torch
from options.base_options import BaseOptions
from trainer import trainer
from utils import print_args
import time

def set_seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_num)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)


def main(args):
    start_time = time.time()  # Log start time
    print(f"Script started at: {time.ctime(start_time)}")

    list_test_acc = []
    list_valid_acc = []
    list_train_loss = []

    filedir = f"./logs/{args.dataset}"
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    if not args.exp_name:
        filename = f"{args.type_model}.json"
    else:
        filename = f"{args.exp_name}.json"
    path_json = os.path.join(filedir, filename)

    try:
        resume_seed = 0
        if os.path.exists(path_json):
            if args.resume:
                with open(path_json, "r") as f:
                    saved = json.load(f)
                    resume_seed = saved["seed"] + 1
                    list_test_acc = saved["test_acc"]
                    list_valid_acc = saved["val_acc"]
                    list_train_loss = saved["train_loss"]
            else:
                t = os.path.getmtime(path_json)
                tstr = datetime.fromtimestamp(t).strftime("%Y_%m_%d_%H_%M_%S")
                os.rename(path_json, os.path.join(filedir, filename + "_" + tstr + ".json"))
        if resume_seed >= args.N_exp:
            print("Training already finished!")
            return
    except:
        pass

    print_args(args)

    if args.debug_mem_speed:
        trnr = trainer(args)
        trnr.mem_speed_bench()

    seed_pool = random.sample(range(1, 101), args.N_exp)
    for i, seed in enumerate(seed_pool[resume_seed:]):
        print(f"seed (which_run) = <{seed}>")

        args.random_seed = seed
        set_seed(args)
        # torch.cuda.empty_cache()
        trnr = trainer(args)
        if args.type_model in [
            "SAdaGCN",
            "AdaGCN",
            "GBGCN",
            "AdaGCN_CandS",
            "AdaGCN_SLE",
            "EnGCN",
        ]:
            train_loss, valid_acc, test_acc = trnr.train_ensembling(seed)
        else:
            train_loss, valid_acc, test_acc = trnr.train_and_test(seed)
        list_test_acc.append(test_acc)
        list_valid_acc.append(valid_acc)
        list_train_loss.append(train_loss)

        # Use default values for learning rate and weight decay if not specified
        lr = args.lr if args.lr is not None else 0.001  # Default LR is 0.001
        weight_decay = args.weight_decay if args.weight_decay is not None else 0.0  # Default weight decay is 0.0

        # Save the trained model state for later soup interpolation
        os.makedirs("trained_soup_ingredients", exist_ok=True)
        model_save_path = f"trained_soup_ingredients/model_{args.type_model}_seed_{seed}_dataset_{args.dataset}_lr_{lr}_wd_{weight_decay}.pth"
        torch.save(trnr.model.state_dict(), model_save_path)  # Save model weights
        print(f"Model state saved at {model_save_path}")

        del trnr
        torch.cuda.empty_cache()
        gc.collect()

        ## record training data
        print(
            "mean and std of test acc: {:.4f} {:.4f} ".format(np.mean(list_test_acc), np.std(list_test_acc))
        )

        try:
            to_save = dict(
                seed=seed,
                test_acc=list_test_acc,
                val_acc=list_valid_acc,
                train_loss=list_train_loss,
                mean_test_acc=np.mean(list_test_acc),
                std_test_acc=np.std(list_test_acc),
            )
            with open(path_json, "w") as f:
                json.dump(to_save, f)
        except:
            pass

    print(
        "final mean and std of test acc: ",
        f"{np.mean(list_test_acc):.4f} $\\pm$ {np.std(list_test_acc):.4f}",
    )

    end_time = time.time()  # Log end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Script finished at: {time.ctime(end_time)}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")  # Print elapsed time

    return np.mean(list_test_acc)


if __name__ == "__main__":
    args = BaseOptions().initialize()
    main(args)