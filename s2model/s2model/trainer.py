
from s0configs import helpers
from s1data import dataprepping, dict_loading
from s2model.models import inference, model


def single_run(m, df, lexica, kwargs, print_params = True):

    train_data, test_data, _ = dataprepping.prepare_df(df, lexica=lexica, heldout_frac=0.2, random_seed = kwargs["random_seed"], print_params = print_params)
    mcmc = inference.run_mcmc(m, train_data, kwargs, print_params)

    fitted_params = inference.get_posterior(mcmc, n_samples=kwargs["post_samples"])
    random_params, _ = inference.get_random(m, train_data, num_samples = fitted_params["pi_Z_c"].shape[0])

    model.evaluate(m, fitted_params, train_data, "fitted")
    #model.evaluate(m, random_params, train_data, "random")
    #helpers.store_model(m, fitted_params)



if __name__ == "__main__":

    kwargs = {"n_c":5, "n_samples": 50, "n_warmup": 50, "random_seed": 400, "init_n": 50, "nuts_tree": 5, "accept_prob": 0.8, "post_samples": -1}

    df = dict_loading.load_dicts(mode="max_nan_count_4")
    lexica = ["SC", "SW", "VA", "GI", "HL", "MP"] #"SC", "SW", "VA", "GI", "HL","MP"
    m = model.get_model(model_type = lexica, n_c = kwargs["n_c"])
    single_run(m, df, lexica, kwargs)

