
from s2model.models.model_types import word_level_model
from s2model.evaluation import predictive, metrics


def get_lex_name(lex):
    if isinstance(lex, str):
        lex_name = lex
    elif isinstance(lex, list):
        lex_name = str(lex[0][0]) + "-" + str(str(lex[1][0]))
    return lex_name


def lex_to_model_type(lexica = None):

    model_name = ""
    if isinstance(lexica, list):
        for i, lex in enumerate(lexica):
            lex_name = get_lex_name(lex)
            model_name += lex_name
            if i < len(lexica) - 1:
                model_name += "_"
    return model_name


def get_model(model_type = None, n_c = "4"):

    model_name = lex_to_model_type(model_type)
    m = word_level_model.Model(n_c, model_name)
    return m



def evaluate(m, params, test_data, posterior_type):

    sites, loglik = predictive.make_prediction(params, m, test_data)

    metrics.compute_exp_pred_lik(loglik, posterior_type=posterior_type, data= test_data)
    metrics.evaluate_point_predictons(sites, test_data, posterior_type=posterior_type)
    metrics.evaluate_distr(sites, test_data, posterior_type=posterior_type)


