
from sklearn.metrics import f1_score, mean_squared_error, accuracy_score
import numpy as np
import math

from s0configs import configs, helpers
from s3analysis.classification import logReg, majVote, decTree, svm


def evaluation(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    #mse = mean_squared_error(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'acc: {acc*100:0.2f}, f1: {f1*100:0.2f}\n') #mse: {mse:0.3f}
    return acc, f1


def data_impute(x, strategy="mean"):
    nan_idx = np.isnan(x)
    if strategy == "mean":  ## impute missing values
        x[nan_idx] = x[~nan_idx].mean()  ## overall mean
    else:
        x[nan_idx] = math.nan  ## nan
    return x


def compute_quantile(x, step=.5):
    eps = 1e-4
    quants = np.arange(0, 1 + eps, step)
    x_features = np.empty((x.shape[0], quants.shape[0]))

    for i, q in enumerate(quants):
        x_features[:, i] = np.quantile(x, q, axis=1)
    return x_features


def feature_preprocessing(task_data, lexicon):
    for part in ["train", "test"]:
        x = task_data[part][lexicon]
        x = data_impute(x)
        seq = x.shape[-1]
        if seq > 1:  ## task is sequence (not single value)
            x = compute_quantile(x)
            ## additional features
            #nan_idx = ~np.isnan(x_seq)  ## sentiment scores
            #sent_count = np.sum(nan_idx.astype(int) , axis=1)
            #sent_count = np.reshape(sent_count, (-1, 1))
            #x = np.concatenate((x, sent_count), axis=1)
        task_data[part][lexicon] = x
    return task_data, seq, x.shape[-1]


def run_classification(data, models = [logReg.Logistic_Regression]): #logReg.Logistic_Regression, majVote.Majority_Vote, decTree.XG_Boost, svm.SVM

    for task, task_data in data.items():
        for lexicon in task_data["train"].keys():
            if lexicon != "y":
                task_data, seq, n_features = feature_preprocessing(task_data, lexicon)
                for model in models:
                    model = model(task_type = task, data_type = lexicon, seq = seq, n_features= n_features)
                    model.train(x_train = task_data["train"][lexicon], y_train = task_data["train"]["y"])
                    y_pred = model.predict(x_test=task_data["test"][lexicon])
                    evaluation(task_data["test"]["y"], y_pred)


if __name__ == "__main__":
    data = helpers.load_pickle(path_name= "task_processed", file_name= "yelp_GI_HL_MP_SC_SW_VA_combined_Z_seq_1")
    run_classification(data)
