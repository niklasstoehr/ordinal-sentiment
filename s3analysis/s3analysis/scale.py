import math

from s0configs import helpers
from s1data import dataprepping, dict_loading, site_proc
from s2model.evaluation import trace_handlers



def infer_Z(m, data = {}, posterior_samples = {}, infer = 1):

    trace = trace_handlers.generate_trace(m, data, posterior_samples, cond="cond", infer = infer)
    Z, _ = trace_handlers.sites_to_tensor(trace, sites=["Z"])
    pooled_Z, pooled_Z_conf = trace_handlers.pool_posterior(Z, pool_type = {"Z": "mean"})
    return pooled_Z, pooled_Z_conf, trace




def prepare_sentiment_dict(model = "SC_SW_VA_GI_HL_MP_nc_5", data = "max_nan_count_4", keep_orig_df=True, store=True):

    model_params = helpers.load_model(file_name=model)
    file_name = str(data) + "_" + str(model)

    if store:
        df = dict_loading.load_dicts(mode=data)

        lexica = ["SC","SW","VA","GI","HL","MP","combined"] #"sentiVAE", "combined"

        data, _, sent_df = dataprepping.prepare_df(df.copy(deep=True), heldout_frac=0.0, lexica=lexica, shuffle=False)

        if not keep_orig_df:
            df = sent_df

        sites, sites_conf, trace = infer_Z(model_params["m"], data, model_params["posterior_params"])
        df["Z"] = sites["Z"].detach().cpu().numpy()
        df["Z_std"] = sites_conf["Z"].detach().cpu().numpy()
        df = site_proc.transform_df_sites(df, to_range_sites = {"Z": [0.0, model_params["m"].n_c-1]}, print_params = True) #"Z_std": [0.0, 1.0]

        #sent_df = site_proc.transform_df_sites(df, to_range_sites={"sentiVAE": [-1.0, 1.0], "SW":[-1.0, 1.0], "SC":[-1.0, 1.0], "VA":[-4.0, 4.0], "HL": [-1, 1], "GI": [-1, 1], "MP": [-1, 1]})
        helpers.store_df(df=df, path_name= "scale", file_name = file_name)
    else:
        df = helpers.load_df(path_name = "scale", file_name = file_name)

    df = df.set_index("word")
    sent_dict = df.to_dict()
    sentiments = dict()
    for lex in sent_dict.keys():
        if lex != "word":
            sentiments[lex] = {word: sent_dict[lex][word] for word in sent_dict[lex] if not math.isnan(sent_dict[lex][word])} ## remove nans from dicts
    return sentiments, df




if __name__ == "__main__":

    prepare_sentiment_dict(model="SC_SW_VA_GI_HL_MP_nc_5", data="max_nan_count_4", store=True)



