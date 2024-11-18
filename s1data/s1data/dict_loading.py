from s0configs import configs, helpers
from sentivae import sentiments

from pathlib import Path
import numpy as np

from tqdm import tqdm
tqdm.pandas()

SW_single = False

FPATHS = {
    # sentiments
    'sentiwordnet': 'SentiWordNet_3.0.0_20130122.txt',
    'mpqa': 'subjclueslen1-HLTEMNLP05.tff',
    'senticnet': 'senticnet5.txt',
    'vader': 'vader_lexicon.txt',
    'huliu': [
        'positive-words.txt',
        'negative-words.txt',
    ],
    'general_inquirer': 'inquirerbasic.csv',
    'sentiVAE': 'sentiVAE_dict.csv',
}


def get_sentiment_paths(data_path, FPATHS):
    path_dict = {}

    for k, v in FPATHS.items():
        if isinstance(v, str):
            path_dict[k] = data_path / Path(v)
        elif isinstance(v, list):
            v_list = list()
            for v_elem in v:
                v_list.append(data_path / Path(v_elem))
            path_dict[k] = v_list

    return path_dict


def process_sentiwordnet(v, SW_single = True):

    if SW_single:
        pos = v[0]
        neg = v[1]
        v = pos - neg
    return v

def process_mpqa(v):
    ## 0.0, 1.0
    return v

def process_senticnet(v):
    ## [-1.0, 1.0]
    return v


def process_vader(v):
    ## (-inf, +inf)
    return v


def process_huliu(v):
    ## 0.0, 1.0
    return v


def process_general_inquirer(v):
    ## 0.0, 1.0
    return v

def process_sentivae(v, single = True):
    if single:
        v = np.array(v) - 1.0
        v = (v[1] * (+1)) + (v[0] * (-1))
    return v


def process_dict_values(df):
    fn_map = {"sentiwordnet": process_sentiwordnet,
              "mpqa": process_mpqa,
              "senticnet": process_senticnet,
              "vader": process_vader,
              "huliu": process_huliu,
              "general_inquirer": process_general_inquirer,
              "sentiVAE": process_sentivae
              }

    df["sent"] = df.apply(lambda row: fn_map[row["source"]](row["sent"]), axis=1)
    return df




def merge_dicts_by_word(df, max_nan_count=5, rename_columns=True):
    df_stacked = df.reset_index().groupby(['word', 'source'])['sent'].aggregate('first').unstack()

    if "sentiVAE" in list(df_stacked.columns):
        df_stacked["nan_count"] = df_stacked.drop("sentiVAE", axis=1).shape[1] - df_stacked.drop("sentiVAE", axis=1).count(axis=1)
    else:
        df_stacked["nan_count"] = df_stacked.shape[1] - df_stacked.count(axis=1)

    df_stacked = df_stacked[df_stacked["nan_count"] <= max_nan_count]
    print(df_stacked)

    column_dict = {'sentiwordnet': ['SW', "float"],
                   'mpqa': ['MP', "int"],
                   'senticnet': ['SC', "float"],
                   'vader': ['VA', "float"],
                   'huliu': ['HL', "int"],
                   'general_inquirer': ['GI', "int"],
                   'sentiVAE': ['sentiVAE', "float"]
                   }

    if rename_columns:
        df_stacked = df_stacked.rename(columns={k: v[0] for k, v in column_dict.items()})

    #if cat_to_int:  ## no int for column containing nan
    #    for k, v in column_dict.items():
    #        if v[1] == "int":
    #            notnan_index = df_stacked[v[0]].notna()
    #            df_stacked[v[0]] = df_stacked[v[0]][notnan_index].astype(int)
    #print(df_stacked)
    return df_stacked



def combine_dicts(df, include_dicts=["SW", "MP", "SC", "VA", "HL", "GI"], new_dict_name = "combined_unscaled"):
    df_notnull = df.copy().fillna(0)
    df[new_dict_name] = df_notnull.progress_apply(lambda row: sum(row[include_dicts]) / (len(include_dicts) - row["nan_count"]), axis=1)
    return df



def load_dicts(mode = "max_nan_count_4"):

    if isinstance(mode, int):
        config = configs.ConfigBase()
        path_dict = get_sentiment_paths(config.get_path("sentiments"), FPATHS)
        df = sentiments.read_all_sentiment_data(path_dict)
        df_proc = process_dict_values(df)
        df_stacked = merge_dicts_by_word(df_proc, max_nan_count=mode)
        df_stacked = combine_dicts(df_stacked)
        df_stacked = df_stacked.reset_index()
        helpers.store_df(df_stacked, path_name= "proc_sentiments", file_name="max_nan_count_" + str(mode))
    elif isinstance(mode, str):
        df_stacked = helpers.load_df(path_name="proc_sentiments", file_name=mode)
    return df_stacked


if __name__ == "__main__":
    load_dicts(mode  =  6)
