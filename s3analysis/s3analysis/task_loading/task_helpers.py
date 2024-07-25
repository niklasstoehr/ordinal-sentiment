from pathlib import Path
import numpy as np
import nltk
import math
from nltk import word_tokenize
nltk.data.path.append('/Users/niklasstoehr/Libraries/nltk_data')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

stop_words = list(set(stopwords.words('english')))
lem = WordNetLemmatizer()


def get_task_paths(data_path, FPATHS):
    path_dict = {}

    for k, v in FPATHS.items():
        if isinstance(v, str):
            path_dict[k] = data_path / Path(k) / Path(v)
        elif isinstance(v, list):
            v_list = list()
            for v_elem in v:
                v_list.append(data_path / Path(k) / Path(v_elem))
            path_dict[k] = v_list

    return path_dict



def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)



def text_to_tokens(text):

    tokens_final = []
    tokens = word_tokenize(text.lower())
    for token in tokens:
        if token not in stop_words and token.isalnum():
            token = lem.lemmatize(token, pos = get_wordnet_pos(token))
            tokens_final.append(token)
    return tokens_final



def score_sent(tokens, sent, max_seq_len = 20, norm_by ="none", spec_pad = False):

    if len(tokens) < max_seq_len and spec_pad != True:
        score = np.empty(max_seq_len)
    else:
        score = np.empty(len(tokens))
    score[:] = np.nan
    n_tok_in_dict = 0

    for i, token in enumerate(tokens):
        if isinstance(sent, dict):
            if token in sent.keys():
                score[i] = np.array(sent[token])
                n_tok_in_dict += 1
        else:
            w2v_value = sent.get_sem_sim(w = token)
            if math.isnan(w2v_value) == False:
                score[i] = np.array(w2v_value).reshape(-1)
                n_tok_in_dict += 1

    if max_seq_len == 1:  ## single score
        nan_idx = np.isnan(score)
        single_score = np.sum(score[~nan_idx], axis=0)

        if norm_by == "seq_len" and len(tokens) > 0:
            single_score = single_score / len(tokens)
        elif norm_by == "words_in_dict" and n_tok_in_dict > 0:
            single_score = single_score / n_tok_in_dict
        elif norm_by == "none":
            pass
        final_score = np.array(single_score).reshape(1)
        #print(single_score, "\n")

    else:
        if spec_pad == True:
            n_pad = max_seq_len - len(tokens)
            if n_pad > 0:
                pad = np.ones(n_pad) * (-99)
                score = np.concatenate([score, pad])
        final_score = score[:max_seq_len]

    return final_score.reshape(-1)




def set_up_result_dict(sent_dicts, n = int(1e5), max_seq_len = 20):

    result_dict = dict()
    for sent_dict in sent_dicts.keys():
        if max_seq_len == 1:
            result_dict[sent_dict] = np.zeros((n, 1))
        else:
            result_dict[sent_dict] = np.zeros((n, max_seq_len))
        result_dict["y"] = np.zeros(n)

    return result_dict


def shuffle_dataset(data):

    ## shuffle
    idx = np.random.permutation(np.arange(len(data['y'])))
    for k, v in data.items():
        data[k] = data[k][idx]
    return data


