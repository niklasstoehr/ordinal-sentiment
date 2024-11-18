

def transform_df_sites(df, to_range_sites = {"SW":[0.0, 1.0], "SC":[0.0, 1.0], "VA":[0.0, 1.0]}, print_params = False):

    for sent, to_range in to_range_sites.items():
        if sent in df.columns:
            if df[sent].dtype == 'int64' or df[sent].dtype == 'float64' or df[sent].dtype == 'float32':

                if len(to_range) == 2:
                    if to_range[0] > to_range[1]:
                        df[sent] = (-1 * df[sent])  ## flip direction
                        lower = to_range[1]
                        upper = to_range[0]
                    else:
                        lower = to_range[0]
                        upper = to_range[1]

                    not_null = df[sent].notnull()
                    x = df[sent][not_null].astype(float)
                    df.loc[not_null,sent] = ((((upper - lower) * (x - min(x))) / (max(x) - min(x))) + lower)
            elif df[sent].dtype == 'object':
                notnull = df[sent].notnull()
                df.loc[notnull,sent] = df[notnull].apply(lambda row: tuple(map(float,(row[sent].replace('(', '').replace(')', '').split(',')))), axis=1)

            if print_params:
                print(f"{sent:} unique: {len(df[sent].value_counts())}, min:, {min(df[sent])}, max: {max(df[sent])}")
    return df



def smooth_boundaries(df, to_range_sites = {"SW":[0.0, 1.0], "SC":[0.0, 1.0], "VA":[0.0, 1.0], "EMB":[0.0, 1.0]}, epsilon=10 ** (-4)):

    for c in df.columns:
        if c in to_range_sites and (df[c].dropna().min() == 0.0 and df[c].dropna().max() == 1.0):

            zero_index = (df[c] == 0.0)
            df.loc[zero_index, c] = epsilon

            one_index = (df[c] == 1.0)
            df.loc[one_index, c] = 1.0 - epsilon
    return df
