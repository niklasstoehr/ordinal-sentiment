import torch
import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.metrics import mean_squared_error, f1_score



def select_nonnan_data(k, hat_data, gt_data):
    not_nan = gt_data["mask"][k]
    gt = gt_data["data"][k][not_nan].float()

    if len(hat_data[k].shape) == 2: ## samples, data_points
        hat = hat_data[k][:, not_nan].float()
    else: ## data_points
        hat = hat_data[k][not_nan].float()
    return hat.cpu(), gt.cpu()



### Point Prediction_____________________________

def compute_metrics(hat, gt, posterior_type=None):

    if torch.is_floating_point(gt):
        res = mean_squared_error(gt.type(torch.float), hat.type(torch.float))
        res = {"mse": np.around(res.item(), 3)}
    else:
        res = f1_score(gt.type(torch.int), hat.type(torch.int), average='weighted')
        res = {"f1": round(res, 3)}

    if posterior_type != None:
        print(f"{str(posterior_type)}: {res}")
    else:
        return res



def evaluate_point_predictons(hat_data, gt_data, posterior_type=None):
    pred_comp = dict()
    for k in gt_data["data"].keys():
        if k in hat_data.keys():
            hat, gt = select_nonnan_data(k, hat_data, gt_data)
            pred_comp[k] = compute_metrics(hat, gt, posterior_type=None)
    print(f"{posterior_type} pred: {pred_comp}")




### Posterior Predictive Density_____________________________

### Posterior Predictive Density_____________________________

def compute_exp_pred_lik(post_loglik, posterior_type=None, data = None, epsilon=10 ** (-5)):
    ### computes pointwise expected log predictive density at each data point

    exp_log_density = dict()
    mean_exp_log_density = 0

    for k in post_loglik.keys():
        if len(post_loglik[k].shape) == 2:  ## need to pool samples
            ## post_loglik is of shape n_samples x n_data
            sample_mean_exp_n = torch.mean(torch.exp(post_loglik[k]), 0)
            sample_mean_exp_n[sample_mean_exp_n <= epsilon] = epsilon ## combat numerical issue
            if data != None:
                mask = data["mask"][k].view(sample_mean_exp_n.shape)
                if len(mask[mask > 0]) > 0:
                    sample_mean_exp_n = sample_mean_exp_n[mask.bool()]
                else: ## all tokens masked
                    sample_mean_exp_n = torch.ones(sample_mean_exp_n.shape)
            exp_log_lik = torch.exp(torch.mean(torch.log(sample_mean_exp_n), axis=0))
            exp_log_density[k] = round(exp_log_lik.item(),3)
            #exp_log_density[k] = (post_loglik[k].logsumexp(0) - math.log(post_loglik[k].shape[0])).sum().item()

            mean_exp_log_density += exp_log_lik.item() * (1 / len(post_loglik.keys()))
        else:
            pass

    mean_exp_log_density = round(mean_exp_log_density, 3)
    if posterior_type != None:
        print(f"{str(posterior_type)} exp_log_density: {mean_exp_log_density}, {exp_log_density}")
    else:
        return exp_log_density, mean_exp_log_density




### Distribution Comparison_____________________________

def distr_comparison(hat_distr, gt_distr, posterior_type = None):

    if isinstance(hat_distr, torch.Tensor):
        hat_distr = hat_distr.detach().numpy()
    if isinstance(gt_distr, torch.Tensor):
        gt_distr = gt_distr.detach().numpy()

    emd = wasserstein_distance(gt_distr, hat_distr)
    emd = round(emd, 3)
    ks_test = ks_2samp(gt_distr, hat_distr)

    if posterior_type != None:
        print(f"{str(posterior_type)}: emd {emd}")#, ks_test stat {np.around(ks_test[0],4)}, ks_test p {np.around(ks_test[1],4)}")
    else:
        return emd, ks_test




def evaluate_distr(hat_data, gt_data, posterior_type = None):

    distr_comp = dict()

    for k in gt_data["data"].keys():
        if k in hat_data.keys():
            hat, gt = select_nonnan_data(k, hat_data, gt_data)
            emd, ks_test = distr_comparison(hat, gt, posterior_type= None)
            distr_comp[k] = emd

    print(f"{posterior_type} emd: {distr_comp}")