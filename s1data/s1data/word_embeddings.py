
import torchtext
import torch
import numpy as np
from s0configs import configs


class SemAxis():

    def __init__(self, config, pos_seed=["like", "good", "positive"], neg_seed=["dislike", "bad", "negative"], min_polarity = 0.2):

        #self.glove = torchtext.vocab.GloVe(name="6B", dim=300)
        self.name = self.lex_to_model_type(lexica=[[pos_seed, neg_seed]])
        self.glove = torchtext.vocab.Vectors(name="glove.6B.300d.txt", cache=config.get_path("glove"))
        self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        self.min_polarity = min_polarity

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

    def get_score(self, w):
        w_emb = self.get_word_emb(w)
        if isinstance(w_emb, torch.Tensor):
            sim = self.cos_sim(w_emb.view(1, -1), self.sem_axis.view(1, -1)).item()
            if abs(sim) < self.min_polarity: ## filter non-polar words
                sim = np.nan
        else: ## return nan
            sim = w_emb
        return sim

    def lex_to_model_type(seld, lexica=None):
        model_name = ""
        if isinstance(lexica, list):
            for i, lex in enumerate(lexica):
                if isinstance(lex, str):
                    lex_name = lex
                elif isinstance(lex, list):
                    lex_name = str(lex[0][0]) + "-" + str(str(lex[1][0]))
                model_name += lex_name
                if i < len(lexica) - 1:
                    model_name += "_"
        return model_name



if __name__ == "__main__":

    config = configs.ConfigBase()
    semAxis = SemAxis(config, min_polarity = 0.1)
    sim = semAxis.get_score("like")
    print(semAxis.name, sim)

