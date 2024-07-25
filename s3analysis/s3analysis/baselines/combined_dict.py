from s0configs import configs, helpers
from s1data import dict_loading

from tqdm import tqdm
tqdm.pandas()


def combine_dicts(df, include_dicts=["SW", "MP", "SC", "VA", "HL", "GI"]):
    df_notnull = df.copy().fillna(0)
    df["combined"] = df_notnull.progress_apply(lambda row: sum(row[include_dicts]) / (len(include_dicts) - row["nan_count"]), axis=1)
    return df


if __name__ == "__main__":

    df_name = "min_nan_count_5"
    df = dict_loading.load_dicts(mode= df_name)
    df = combine_dicts(df)
    helpers.store_df(df, path_name="proc_sentiments", file_name=df_name)




