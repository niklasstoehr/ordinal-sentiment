
from s0configs import configs, helpers
from s1data import dict_loading

import torchtext
import torch
import numpy as np


class SemAxis():

    def __init__(self, pos_seed=["like", "good", "positive"], neg_seed=["dislike", "bad", "negative"]):

        self.glove = torchtext.vocab.GloVe(name="6B", dim=300)
        self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

        pos_emb, neg_emb = self.get_seed_embs(pos_seed, neg_seed)
        self.sem_axis = self.get_sem_axis(pos_emb, neg_emb)

    def get_seed_embs(self, pos_seed, neg_seed):
        pos_embs = torch.Tensor(len(pos_seed), self.glove.dim)
        neg_embs = torch.Tensor(len(neg_seed), self.glove.dim)

        for i, p in enumerate(pos_seed):
            pos_embs[i, :] = self.glove[p]

        for i, n in enumerate(neg_seed):
            neg_embs[i, :] = self.glove[n]

        pos_emb = torch.mean(pos_embs, axis=0)
        neg_emb = torch.mean(neg_embs, axis=0)
        return pos_emb, neg_emb

    def get_sem_axis(self, pos_emb, neg_emb):
        sem_axis = pos_emb - neg_emb
        return sem_axis

    def get_word_emb(self, w):
        if w in self.glove.stoi.keys():
            w_emb = self.glove[w]
        else:
            w_emb = np.nan
        return w_emb

    def get_sem_sim(self, w):
        w_emb = self.get_word_emb(w)
        if isinstance(w_emb, torch.Tensor):
            sim = self.cos_sim(w_emb.view(1, -1), self.sem_axis.view(1, -1)).item()
        else: ## return nan
            sim = w_emb
        return sim


def semaxis_score_words(word, semAxis):
    w_emb = semAxis.get_sem_sim(w = word)
    return w_emb



if __name__ == "__main__":

    df_name = "min_nan_count_5"
    df = dict_loading.load_dicts(mode= df_name)

    semAxis = SemAxis()
    df["EMB"] = df.apply(lambda row: semaxis_score_words(row["word"], semAxis), axis=1)
    helpers.store_df(df, path_name="proc_sentiments", file_name=df_name)
