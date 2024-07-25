from s0configs import configs, helpers
from s3analysis import scale
from s3analysis.task_loading import task_helpers, task_load
from s3analysis.classification import class_main

from collections import  defaultdict


def prepare_tasks(tasks = ["semeval"], scorer = [], lexica = [], limit_to = 500, max_seq_len = 1, include_test = True):

    data = defaultdict(dict)
    config = configs.ConfigBase()
    path_dict = task_helpers.get_task_paths(config.get_path("tasks"), task_load.DATA_PATHS)

    for task in tasks:
        data[task]['train'] = task_load.FUNCTIONS[task](path_dict[task][0], lexica, scorer, limit_to=limit_to, max_seq_len=max_seq_len)
        data[task]['train'] = task_helpers.shuffle_dataset(data[task]['train'])

        if include_test:
            data[task]['test'] = task_load.FUNCTIONS[task](path_dict[task][1], lexica, scorer, limit_to=limit_to, max_seq_len=max_seq_len)
            data[task]['test'] = task_helpers.shuffle_dataset(data[task]['test'])

    print(f"\nprepared {tasks} with lexica {lexica.keys()} and seq length {max_seq_len}")
    file_name = "_".join(tasks) + "_" + "_".join(list(lexica.keys())) + "_seq_" + str(max_seq_len)
    helpers.pickle_data(data, path_name="task_processed", file_name=file_name)
    return data




if __name__ == "__main__":


    sentiments, sent_df = scale.prepare_sentiment_dict(model="SC_SW_VA_GI_HL_MP_nc_5", data="max_nan_count_4", keep_orig_df=False)

    limit_to = 10000#int(1e5)
    max_seq_len = 1

    #sentiVAE
    lexica = dict((k, v) for k, v in sentiments.items() if k in ["SC","SW","VA","GI","HL","MP","combined","Z"]) #"SC","SW","VA","GI","HL","MP",
    #semaxis = {"semaxis": word_embeddings.SemAxis()}
    #lexica = {**semaxis, **lexica}
    scorer = lambda text, sent_data: task_helpers.score_sent(text, sent_data, max_seq_len=max_seq_len, norm_by="words_in_dict") # words_in_dict, sent_len

    #'imdb', 'yelp', 'semeval', 'multi-domain-sentiment', 'acl_2017', 'iclr_2017'
    data = prepare_tasks(tasks = ['imdb', 'yelp', 'semeval', 'multi-domain-sentiment', 'acl_2017', 'iclr_2017'], scorer = scorer, lexica = lexica, limit_to = limit_to, max_seq_len = max_seq_len)

    class_main.run_classification(data)


