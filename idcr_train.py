import os
import random

import torch
import pickle
import argparse
import json

from sklearn.preprocessing import LabelEncoder

import evaluate_data
from model.causalTGAN import CausalTGAN
from helper.feature_info import FeatureINFO
from configuration import TrainingOptions, CausalTGANConfig
from helper.utils import data_transform, load_causal_graph, get_discrete_cols
from helper.trainer import train_model
import time
import numpy as np
import pandas as pd
from helper.kfold import kfold


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(seed = 1):
    RMSEs = []
    TE1s = []
    TE2s = []
    DPDs = []
    FPRDs = []
    set_seed(seed)
    if args.lamb < 0 or args.lamb > 1:
        raise ValueError("lamb must be between 0 and 1")

    device = torch.device('cuda:{}'.format(args.device_idx)) if torch.cuda.is_available() else torch.device('cpu')
    # load data and causal graph
    data_path = os.path.join("data", "real_world", args.data_name, 'data.csv')
    data = pd.read_csv(data_path)
    if args.data_name.startswith("law_school"):
        edges = {
            "race": [
                "ZFYA",
                "UGPA",
                "LSAT"
            ]
        }
        sensitive = "race"
        label = "ZFYA"
        data = data.drop(columns=['first_pf'], axis=1)
        selected_races = data['race'].unique().tolist()
        label_encoder = LabelEncoder()  # Will further encode with OneHotEncoder
        label_encoder.fit(selected_races)
        data['sex'] = data['sex'] - 1  # 1, 2 -> 0, 1
        data['race'] = label_encoder.fit_transform(data['race'])
        # Law School dataset is imbalanced, so stratified sampling is needed
        k_train, k_test = kfold(data, 5, True, "race")
    elif args.data_name.startswith("synthetic"):
        edges = {
            "A": [
                "X2",
                "Y"
            ]
        }
        sensitive = "A"
        label = "Y"
        k_train, k_test = kfold(data, 5)
    else:
        raise NotImplementedError("Unknown datasets")

    discrete_cols, col_names = get_discrete_cols(data, args.data_name)
    for exp_index, train_data in enumerate(k_train):
        set_seed(seed)
        # Create experiments folder
        exp_name = f"{args.data_name}_{exp_index}_l=%.3f(%s)" % (
            args.lamb, time.strftime('%m_%d_%H_%M', time.localtime()))
        this_run_folder = str(os.path.join(args.runs_folder, exp_name))
        os.makedirs(this_run_folder, exist_ok=True)
        # Load causal graph from the data folder
        causal_graph = load_causal_graph(args.data_name)
        # Define the sensitive attribute, label, and discriminative causal relationships (edges)
        train_options = TrainingOptions(
            lamb=args.lamb,
            label=label,
            sensitive_attribute=sensitive,
            edges=edges,
            batch_size=args.batch_size,
            number_of_epochs=args.epochs,
            runs_folder=this_run_folder,
            experiment_name=exp_name,
        )

        # Save experiments setting
        with open(os.path.join(this_run_folder, 'option.json'), 'w') as json_file:
            json_file.write(json.dumps(train_options.__dict__, ensure_ascii=False, indent=4))

        gan_config = CausalTGANConfig(causal_graph=causal_graph, z_dim=args.z_dim,
                                      pac_num=args.pac_num, D_iter=args.d_iter)
        transform_data, transformer, data_dims = data_transform(args.data_name, train_data, discrete_cols)
        full_feature_info = FeatureINFO(col_names, discrete_cols, data_dims)

        with open(os.path.join(this_run_folder, 'options-and-config.pickle'), 'wb+') as f:
            pickle.dump(train_options, f)
            pickle.dump(gan_config, f)
        with open(os.path.join(this_run_folder, 'causal_graph.pickle'), 'wb') as f:
            pickle.dump(causal_graph, f)
        with open(os.path.join(this_run_folder, 'transformer.pickle'), 'wb') as f:
            pickle.dump(transformer, f)

        feature_info = full_feature_info
        model = CausalTGAN(device, gan_config, feature_info, transformer, train_options.label)
        start_time = time.time()
        # Phase 1
        lamb = train_options.lamb
        train_options.lamb = 0
        train_model(train_options, transform_data, model)
        # Phase 2
        train_options.lamb = lamb
        model.load_gy_state_dict()
        train_model(train_options, transform_data, model)
        print("Time: ", time.time() - start_time)
        print("Evaluating...")
        model.causal_controller.set_causal_mechanisms_eval()
        gen_num = len(data)
        r = []
        with torch.no_grad():
            i = 0
            while i < gen_num:
                batch_num = min(1000, gen_num - i)
                z = torch.normal(0, 1, (len(data.columns), batch_num, model.config.z_dim)).to(device)
                i += batch_num
                samples = model.sample(batch_num, z)
                r.append(samples.cpu())
        samples = torch.cat(r, dim=0)
        sample_df = transformer.inverse_transform(samples.data.numpy())
        # If you need to save the data, uncomment the following code
        sample_df.to_csv(os.path.join(this_run_folder, "data_generated.csv"), float_format="%.2f", index=False)
        if args.data_name.startswith("law_school"):
            sample_df["LSAT"] = np.floor(sample_df["LSAT"])
            rmse, te1, te2, dpd, fprd = evaluate_data.eval_law_school(sample_df, k_test[exp_index])
        elif args.data_name.startswith("synthetic"):
            rmse, te1, te2, dpd, fprd = evaluate_data.eval_synthetic_dataset(sample_df, k_test[exp_index])
        else:
            raise NotImplementedError('Unknown dataset.')
        RMSEs.append(float(rmse))
        TE1s.append(float(te1))
        TE2s.append(float(te2))
        DPDs.append(float(dpd))
        FPRDs.append(float(fprd))
    r = {
        "RMSE": RMSEs,
        "TE1": TE1s,
        "TE2": TE2s,
        "DPD": DPDs,
        "FPRD": FPRDs,
    }
    for key, values in r.items():
        data_array = np.array(values)
        m = data_array.mean()
        std_deviation = np.std(data_array, ddof=1)
        uncertainty = std_deviation
        print(f"{key} : %.3f±%.3f" % (m, uncertainty.item()))
    with open(os.path.join(this_run_folder, 'result.json'), 'w') as json_file:
        json_file.write(json.dumps(r, ensure_ascii=False, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IDCR')
    parser.add_argument('--data_name', '-dn', default='law_school', type=str,
                        help='The name of dataset. see data/real_world/')
    parser.add_argument('--device_idx', '-gpu', default=0, type=int, help='CUDA index')
    parser.add_argument('--batch_size', '-b', default=512, type=int, help='The batch size')
    parser.add_argument('--epochs', '-e', default=100, type=int, help='Number of epochs for per phase')
    parser.add_argument('--runs_folder', '-sf',
                        default=os.path.join('.', "checkpoints"),
                        type=str,
                        help='The root folder where trained model are stored')
    parser.add_argument('--pac_num', '-pc', default=1, type=int, help='Number of sample in one pac in pac gan')
    parser.add_argument('--z_dim', '-z', default=3, type=int,
                        help='The exogenous variable size')
    parser.add_argument('--d_iter', '-di', default=3, type=int)
    parser.add_argument('--lamb', '-l', default=0.3, type=float, help='lambda_f value of IDC penalty term')
    parser.set_defaults()
    args = parser.parse_args()
    main(1)
