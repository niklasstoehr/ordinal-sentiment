from s0configs import configs, helpers
from s1data import dict_loading
from sentivae import sentiments

import pandas as pd
import numpy as np


def merge_dicts(df_main, df_new, dict_name = "VAE"):

    df_main = df_main.set_index("word")
    df_new = df_new.set_index("word")
    df_new = df_new.rename(columns = {"sent": dict_name})

    df = pd.merge(df_main, df_new, how = "left", left_index = True, right_index = True)
    return df.reset_index()



def to_single_scale(v):

    v = np.array(v) - 1.0
    v = (v[1] * (+1)) + (v[0] * (-1))
    return v

def process_vae(df):

    df["sent"] = df.apply(lambda row: to_single_scale(row["sent"]), axis = 1)
    return df


if __name__ == "__main__":

    config = configs.ConfigBase()
    path_dict = dict_loading.get_sentiment_paths(config.get_path("sentiments"), dict_loading.FPATHS)

    vae_df = sentiments.parse_vae(path_dict["sentiVAE"], sent_cols=['alpha_1', 'alpha_2', 'alpha_3'])
    vae_df = process_vae(vae_df)

    df_name = "min_nan_count_2"
    df = dict_loading.load_dicts(mode= df_name)
    merge_dicts(df, vae_df, dict_name="VAE")
    helpers.store_df(df, path_name="proc_sentiments", file_name=df_name)



