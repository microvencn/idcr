
import pickle

import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import Trace_ELBO, SVI
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule, PyroSample
import numpy as np

pyro.enable_validation(True)
pyro.set_rng_seed(1)

device = torch.device("cpu")


def quickprocess(var):
    if var is None:
        return var
    var = torch.tensor(var, dtype=torch.float32).view(-1).to(device)
    return var


def to_onehot(var, num_classes=-1):
    var_onehot = F.one_hot(var, num_classes)
    dim = num_classes if num_classes != -1 else var_onehot.shape[1]
    return var_onehot, dim


def onehot_to_int(var):
    var_int = torch.argmax(var, dim=1)
    return var_int


class CausalModel_law(PyroModule):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.one_hot = 0

    def forward(self, data):
        dim_race = 1
        data_race, data_UGPA, data_LSAT, data_ZFYA, data_sex = data['race'], data['UGPA'], data['LSAT'], data['ZFYA'], \
            data['sex']
        data_race, data_UGPA, data_LSAT, data_ZFYA, data_sex = quickprocess(data_race), quickprocess(
            data_UGPA), quickprocess(
            data_LSAT), quickprocess(data_ZFYA), quickprocess(data_sex)
        if data_LSAT is not None:
            data_LSAT = torch.floor(data_LSAT)
        self.pi = pyro.param(self.model_name + "_" + "pi",
                             torch.tensor(ratio).to(device))  # S~Cate(pi)
        self.si = pyro.param(self.model_name + "_" + "si", torch.tensor([0.5, 0.5]).to(device))
        self.b_g = pyro.param(self.model_name + "_" + "b_g", torch.tensor(0.).to(device))
        self.w_g_k = pyro.param(self.model_name + "_" + "w_g_k", torch.tensor(0.).to(device))
        self.w_g_r = pyro.param(self.model_name + "_" + "w_g_r", torch.zeros(dim_race, 1).to(device))
        self.w_g_s = pyro.param(self.model_name + "_" + "w_g_s", torch.tensor(0.).to(device))

        self.sigma_g = pyro.param(self.model_name + "_" + "sigma_g", torch.tensor(1.).to(device))

        self.b_l = pyro.param(self.model_name + "_" + "b_l", torch.tensor(0.).to(device))
        self.w_l_k = pyro.param(self.model_name + "_" + "w_l_k", torch.tensor(0.).to(device))
        self.w_l_r = pyro.param(self.model_name + "_" + "w_l_r", torch.zeros(dim_race, 1).to(device))
        self.w_l_s = pyro.param(self.model_name + "_" + "w_l_s", torch.tensor(0.).to(device))

        self.w_f_k = pyro.param(self.model_name + "_" + "w_f_k", torch.tensor(0.).to(device))
        self.w_f_r = pyro.param(self.model_name + "_" + "w_f_r", torch.zeros(dim_race, 1).to(device))
        self.w_f_s = pyro.param(self.model_name + "_" + "w_f_s", torch.tensor(0.).to(device))

        n = len(data_race)
        with pyro.plate('observe_data', size=n, device=device):
            knowledge = pyro.sample('knowledge', pyro.distributions.Normal(torch.tensor(0.).to(device), 1.)).to(
                device)
            race = pyro.sample('obs_race',
                               pyro.distributions.Categorical(self.pi),
                               obs=data_race)
            race_out = race
            data_sex = pyro.sample("obs_sex", pyro.distributions.Categorical(self.si), obs=data_sex)

            gpa_mean = self.b_g + self.w_g_k * knowledge + self.w_g_s * data_sex + self.w_g_r * race_out
            sat_mean = torch.exp(
                self.b_l + self.w_l_k * knowledge + self.w_l_r * race_out + self.w_l_s * data_sex)
            fya_mean = self.w_f_k * knowledge + race_out * self.w_f_r + self.w_f_s * data_sex
            # The model comes from:
            # Kusner, Matt J., et al. "Counterfactual fairness." Advances in neural information processing systems 30 (2017).
            gpa_obs = pyro.sample("obs_UGPA", dist.Normal(gpa_mean, torch.abs(self.sigma_g)), obs=data_UGPA).view(-1, 1)
            sat_obs = pyro.sample("obs_LSAT", dist.Poisson(sat_mean), obs=data_LSAT).view(-1, 1)
            fya_obs = pyro.sample("obs_ZFYA", dist.Normal(fya_mean, 1), obs=data_ZFYA).view(-1, 1)

        data_return = {'knowledge': knowledge, 'race': race, 'LSAT': sat_obs, 'UGPA': gpa_obs, 'ZFYA': fya_obs,
                       "sex": data_sex}
        return data_return

    def sample(self, race=None, knowledge=None, data_sex=None):
        race = race
        race_out = race

        gpa_mean = self.b_g + self.w_g_k * knowledge + self.w_g_s * data_sex + self.w_g_r * race_out
        sat_mean = torch.exp(
            self.b_l + self.w_l_k * knowledge + self.w_l_r * race_out + self.w_l_s * data_sex)
        fya_mean = self.w_f_k * knowledge + race_out * self.w_f_r + self.w_f_s * data_sex

        gpa_obs = pyro.sample("obs_UGPA", dist.Normal(gpa_mean, torch.abs(self.sigma_g))).view(-1, 1)
        sat_obs = pyro.sample("obs_LSAT", dist.Poisson(sat_mean)).view(-1, 1)
        fya_obs = pyro.sample("obs_ZFYA", dist.Normal(fya_mean, 1)).view(-1, 1)

        data_return = {'knowledge': knowledge, 'race': race, "sex": data_sex, 'LSAT': sat_obs, 'UGPA': gpa_obs, 'ZFYA': fya_obs,
                       }
        return data_return


def generate_counterfactuals(model, target_race, knowledge, sex):
    data_counterfactuals = data.copy()
    data_counterfactuals['race'] = target_race
    return model.sample( race=target_race, knowledge=knowledge, data_sex=sex)


if __name__ == '__main__':
    ###################
    # Data preprocessing
    ###################
    # data = {'race': ..., 'sex': ..., 'UGPA': ..., 'LSAT': ..., 'ZFYA': ...}
    csv_data = pd.read_csv("data.csv")
    total_count = len(csv_data['race'])
    ratio = []
    selected_races = csv_data['race'].unique()
    selected_races.sort()
    selected_races = selected_races.tolist()
    print("select races: ", selected_races)
    for value in selected_races:
        count = csv_data['race'].value_counts().get(value, 0)
        ratio.append(count / total_count)
    select_index = np.arange(0, len(csv_data['race']))
    np.random.shuffle(select_index)
    LSAT = csv_data[['LSAT']].to_numpy()[select_index]  # n x 1
    UGPA = csv_data[['UGPA']].to_numpy()[select_index]  # n x 1
    x = csv_data[['LSAT', 'UGPA']].to_numpy()[select_index]  # n x d
    ZFYA = csv_data[['ZFYA']].to_numpy()[select_index]  # n x 1
    sex = csv_data[['sex']].to_numpy()[select_index] - 1  # 1,2 -> 0,1
    n = ZFYA.shape[0]
    rr = csv_data['race']
    env_race = csv_data['race'][select_index].to_list()  # n, string list
    env_race_id = np.array([selected_races.index(env_race[i]) for i in range(n)]).reshape(-1, 1)
    data_save = {'data': {'race': env_race_id, "sex": sex, 'LSAT': LSAT, 'UGPA': UGPA, 'ZFYA': ZFYA}}
    data = data_save['data']

    ###################
    # Model training
    ###################
    train = False
    if train:
        model_name = "law_model"
        model = CausalModel_law(model_name)
        guide = AutoDiagonalNormal(model)
        adam = pyro.optim.Adam({"lr": 1e-3})
        svi = SVI(model, guide, adam, loss=Trace_ELBO())
        pyro.clear_param_store()
        for j in range(15000):
            # calculate the loss and take a gradient step
            loss = svi.step(data)  # all data is used here
            if j % 100 == 0:
                print("[iteration %04d] loss: %.4f" % (j + 1, loss))
        # Train successfully
        for name, value in pyro.get_param_store().items():
            print(name, pyro.param(name))
        # Save model
        with open("./param.pt", "wb") as f:
            pickle.dump(model, f)
        guide.requires_grad_(False)
    else:
        with open("./param.pt", "rb") as f:
            model: CausalModel_law = pickle.load(f)

    ###################
    # Data sampling
    ###################
    model.eval()
    # Sample race counterfactual data under same knowledge
    knowledge = torch.normal(0, 1, (2000, 1))
    sex = torch.randint(0, 2, (2000, 1)).float()
    for i in range(len(selected_races)):
        data_to_gen = {
            "race": None,
            "LSAT": None,
            "UGPA": None,
            "ZFYA": None,
        }
        target_race = torch.tensor([float(i)] * 2000, device=device).view(-1, 1)
        counterfactual_data = generate_counterfactuals(model, target_race, knowledge, sex)
        for k, v in counterfactual_data.items():
            counterfactual_data[k] = v.detach().clone().view(-1).numpy().tolist()
        df = pd.DataFrame(counterfactual_data)
        df.drop(["knowledge"], axis=1, inplace=True)
        df.to_csv(f"{selected_races[i]}.csv", index=False, float_format="%.2f")
        print(counterfactual_data)
