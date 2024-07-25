
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
tqdm.pandas()

from s0configs import helpers
from s1data import dict_loading, site_proc



def train_test_indeces(n_x=1000, heldout_frac=0.2, shuffle = True):
    if shuffle:
        ran_inds = torch.randperm(n_x)
    else: ## do not shuffle
        ran_inds = torch.arange(0, n_x, 1)
    split_ind = int(len(ran_inds) * (1-heldout_frac))

    train_ind = ran_inds[:split_ind]  ## split
    test_ind = ran_inds[split_ind:]  ## split
    return train_ind, test_ind



def prepare_nan_masking(data_values):

    data_type = data_values.dtype
    nan_index = torch.isnan(data_values)
    notnan_index = ~nan_index

    if len(torch.unique(data_values[notnan_index])) <= 4:
        nan_value = 0.0
    else:
        nan_value = 0.0001

    corr_data_values = torch.where(nan_index, nan_value, data_values.double()).type(data_type)
    mask = torch.where(nan_index, 0.0, 1.0)  ## missing values are 0
    mask = mask.bool() ## make boolean
    return corr_data_values, mask



def prepare_data_masks(data, indeces, print_params = True):

    if print_params:
        print(f"prepare_data_masks {indeces[:10]}...")
    masked_data = defaultdict(dict)
    masked_data["data"] = dict()
    masked_data["mask"] = dict()

    for k, v in data["data"].items():
        data_v, mask_v = prepare_nan_masking(v[indeces])
        masked_data["data"][k] = data_v
        masked_data["mask"][k] = mask_v
    return masked_data



def pack_dicts(df, include_data = ["SC", "SW", "VA", "GI", "HL", "MP"]):
    data = defaultdict(dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(device)
    for c in include_data:
        if c in df.columns:
            if df[c].dtype == int:
                data["data"][c] = torch.LongTensor(df[c]).to(device)
            if df[c].dtype == float:
                data["data"][c] = torch.FloatTensor(df[c]).to(device)
            elif df[c].dtype == object:
                tuple_tensor = torch.empty((len(df), 2))
                tuple_tensor[:] = np.NaN
                notnull_index = df[c].notnull()
                tuple_tensor[notnull_index, :] = torch.FloatTensor(df.loc[notnull_index, c])
                data["data"][c] = tuple_tensor

    return data



def prepare_df(df, lexica = [], heldout_frac = 0.2, shuffle = True, to_range_sites = {"SW":[0.0, 1.0], "SC":[0.0, 1.0], "VA":[0.0, 1.0], "EMB":[0.0, 1.0]}, random_seed = 0, print_params = True):

    torch.manual_seed(helpers.set_random_seed(random_seed))

    df_trans = site_proc.transform_df_sites(df.copy(), to_range_sites= to_range_sites)
    df_trans = site_proc.smooth_boundaries(df_trans, to_range_sites=to_range_sites)

    df_trans = dict_loading.combine_dicts(df_trans, include_dicts=["SW", "MP", "SC", "VA", "HL", "GI"], new_dict_name="combined")
    data = pack_dicts(df_trans[lexica], include_data=lexica)

    ## train, test
    train_ind, test_ind = train_test_indeces(len(df_trans), heldout_frac= heldout_frac, shuffle = shuffle)
    train_data = prepare_data_masks(data, train_ind, print_params = print_params)

    if len(test_ind.shape) > 0:
        test_data = prepare_data_masks(data, test_ind, print_params = print_params)
    else:
        test_data = torch.empty()

    return train_data, test_data, df_trans


if __name__ == "__main__":

    df = dict_loading.load_dicts(mode= "max_nan_count_4")
    train_data, test_data, df = prepare_df(df, lexica = ["SW", "MP"], heldout_frac = 0.2)

