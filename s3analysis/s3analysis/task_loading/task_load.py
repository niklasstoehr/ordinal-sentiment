
import numpy as np
import json
import os
import pandas as pd
import ijson

from s3analysis.task_loading import task_helpers

DATA_PATHS = {
    # sentiments
    'imdb': [
        'train',
        'test',
    ],
    'yelp': [
        'yelp_academic_dataset_review_train.json',
        'yelp_academic_dataset_review_test.json',
    ],
    'semeval': [
        'SemEval2017-task4-dev.subtask-A.english.INPUT_train.txt',
        'SemEval2017-task4-dev.subtask-A.english.INPUT_test.txt',
    ],
    'multi-domain-sentiment': [
        'multi-domain-sentiment_indomain_train.txt',
        'multi-domain-sentiment_indomain_test.txt',
    ],
    'acl_2017': [
        'train',
        'test',
    ],
    'iclr_2017': [
        'train',
        'test',
    ],
}





def gen_semeval_data(fpath, sent_dicts, score_fn, limit_to, max_seq_len = 20):
    """
    Create semeval dataset from sentiment lexicon
    """

    data = pd.read_csv(fpath, sep='\t', names=['id', 'sent', 'text', '_'], encoding='utf-8')
    data['sent'] = data.sent.replace({'negative': 0, 'neutral': 1, 'positive': 2})

    n = limit_to or len(data)
    result_dict = task_helpers.set_up_result_dict(sent_dicts, n = n, max_seq_len = max_seq_len)

    i = 0
    for _, row in data.iterrows():
        tokens = task_helpers.text_to_tokens(row.text)
        for j, (sent_key, word_sent_dict) in enumerate(sent_dicts.items()):
            score = score_fn(tokens, word_sent_dict)
            result_dict[sent_key][i] = score

        result_dict["y"][i] = row.sent
        i += 1

        print(f'\r{str(fpath)}    {i/n*100:0.2f}%', end='')
        if i >= n:
            break
    return result_dict




def gen_yelp_data(fpath, sent_dicts, score_fn, limit_to, max_seq_len=20):

    n = limit_to
    result_dict = task_helpers.set_up_result_dict(sent_dicts, n = n, max_seq_len = max_seq_len)

    i = 0

    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            tokens = task_helpers.text_to_tokens(data['text'])

            for j, (sent_key, word_sent_dict) in enumerate(sent_dicts.items()):
                sent_score = score_fn(tokens, word_sent_dict)
                result_dict[sent_key][i] = sent_score

            result_dict["y"][i] = int(data['stars'] - 1)
            i += 1

            print(f'\r{str(fpath)}    {i/n*100:0.2f}%', end='')
            if i >= n:
                break
    return result_dict




def gen_multidom_data(fpath, sent_dicts, score_fn, limit_to=None, max_seq_len=20):
    """
    Create multi-domain sentiment analysis dataset from sentiment lexicon
    """

    n = limit_to
    result_dict = task_helpers.set_up_result_dict(sent_dicts, n = n, max_seq_len = max_seq_len)

    i = 0
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            text, label = line.strip("\n").split("\t")
            tokens = task_helpers.text_to_tokens(text)

            for j, (sent_key, word_sent_dict) in enumerate(sent_dicts.items()):
                score = score_fn(tokens, word_sent_dict)
                result_dict[sent_key][i] = score

            result_dict["y"][i] = int(label)
            i += 1

            print(f'\r{str(fpath)}    {i/n*100:0.2f}%', end='')
            if i >= n:
                break
    return result_dict






def gen_imdb_data(dir, sent_dicts, score_fn, limit_to=None, max_seq_len=20):
    """
    Create imdb dataset from sentiment lexicon
    """

    pos_data = [('pos', fname) for fname in os.listdir(os.path.join(dir, 'pos'))][:int(limit_to / 2)]
    neg_data = [('neg', fname) for fname in os.listdir(os.path.join(dir, 'neg'))][:int(limit_to / 2)]

    n = len(pos_data) + len(neg_data)
    result_dict = task_helpers.set_up_result_dict(sent_dicts, n = n, max_seq_len = max_seq_len)
    result_dict["y"] = np.concatenate([np.ones(len(pos_data)), np.zeros(len(neg_data))])

    for i, (sent, fname) in enumerate(pos_data + neg_data):
        with open(os.path.join(dir, sent, fname), 'r', encoding='latin1') as textfile:
            text = textfile.read()
            tokens = task_helpers.text_to_tokens(text)
            for j, (sent_key, word_sent_dict) in enumerate(sent_dicts.items()):
                score = score_fn(tokens, word_sent_dict)
                result_dict[sent_key][i] = score

        print(f'\r{str(dir)}    {i/n*100:0.2f}%', end='')
    return result_dict




def gen_acl_data(dir, sent_dicts, score_fn, limit_to=None, max_seq_len=20, merge=True):
    """
    Create PeerReview ACL dataset from sentiment lexicon
    """

    acl_data = [fname for fname in os.listdir(os.path.join(dir, 'reviews'))]
    if "train" in str(dir):
        if limit_to <= 248:
            n = limit_to
        else:
            n = 248
    elif "test" in str(dir):
        if limit_to <= 15:
            n = limit_to
        else:
            n = 15
    result_dict = task_helpers.set_up_result_dict(sent_dicts, n = n, max_seq_len = max_seq_len)

    if merge:
        score2norm = {"1": 0, "2": 0, "3": 0, "4": 1, "5": 1, "6": 1}
    else:
        score2norm = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5}

    count = 0

    for i, fname in enumerate(acl_data):
        currpath = os.path.join(dir, 'reviews', fname)
        f = open(currpath, encoding="utf-8")
        objects = ijson.items(f, 'reviews')
        for ii, obj in enumerate(objects):
            for j, objj in enumerate(obj):
                text = objj["comments"]
                tokens = task_helpers.text_to_tokens(text)
                for k, (sent_key, word_sent_dict) in enumerate(sent_dicts.items()):
                    score = score_fn(tokens, word_sent_dict)
                    result_dict[sent_key][count] = score
                result_dict["y"][count] = score2norm[objj["RECOMMENDATION"]]
                count += 1
                if count == (n - 1):
                    return result_dict
        print(f'\r{str(dir)}  {count/n*100:0.2f}%', end='')





def gen_iclr_data(dir, sent_dicts, score_fn, limit_to=None, max_seq_len=20, merge=True):
    """
    Create PeerReview ICLR dataset from sentiment lexicon
    """

    iclr_data = [fname for fname in os.listdir(os.path.join(dir, 'reviews'))]
    if "train" in str(dir):
        if limit_to <= 2166:
            n = limit_to
        else:
            n = 2166
    elif "test" in str(dir):
        if limit_to <= 230:
            n = limit_to
        else:
            n = 230
    result_dict = task_helpers.set_up_result_dict(sent_dicts, n = n, max_seq_len = max_seq_len)

    if merge:
        score2norm = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 1, "6": 2, "7": 2, "8": 2, "9": 2, "10": 2}
    else:
        score2norm = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "9": 8, "10": 9}

    count = 0

    for i, fname in enumerate(iclr_data):
        currpath = os.path.join(dir, 'reviews', fname)
        f = open(currpath, encoding="utf-8")
        objects = ijson.items(f, 'reviews')
        for ii, obj in enumerate(objects):
            for j, objj in enumerate(obj):
                # some are meta-reviews without scores
                if not "RECOMMENDATION" in objj.keys():
                    continue
                text = objj["comments"]
                tokens = task_helpers.text_to_tokens(text)
                for j, (sent_key, word_sent_dict) in enumerate(sent_dicts.items()):
                    score = score_fn(tokens, word_sent_dict)
                    result_dict[sent_key][count] = score

                result_dict["y"][count] = score2norm[str(objj["RECOMMENDATION"])]
                count += 1
                if count == (n - 1):
                    return result_dict
        print(f'\r{str(dir)}    {count/n*100:0.2f}%', end='')





FUNCTIONS = {
    # sentiments
    'imdb': gen_imdb_data,
    'yelp': gen_yelp_data,
    'semeval': gen_semeval_data,
    'multi-domain-sentiment': gen_multidom_data,
    'acl_2017': gen_acl_data,
    'iclr_2017':gen_iclr_data,
}