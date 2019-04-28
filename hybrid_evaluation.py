__author__ = 'liming-vie'

import os

from referenced_metric import Referenced
from unreferenced_metric import Unreferenced


class Hybrid(object):
    def __init__(self,
                 data_dir,
                 word2vec_file,
                 query_w2v_file,
                 reply_w2v_file,
                 train_dir,
                 query_max_len=20,
                 reply_max_len=30,
                 pooling_type='max_min',
                 gru_units=128,
                 mlp_units=None):
        if mlp_units is None:
            mlp_units = [256, 512, 128]
        self.data_dir=data_dir
        self.ref = Referenced(data_dir, word2vec_file, pooling_type)
        self.unref = Unreferenced(query_max_len, reply_max_len,
                                  os.path.join(data_dir, query_w2v_file),
                                  os.path.join(data_dir, reply_w2v_file),
                                  gru_units, mlp_units,
                                  train_dir=train_dir)

    def train_unref(self, data_dir, query_file, reply_file):
        self.unref.train(data_dir, query_file, reply_file)

    def _normalize(self, scores):
        smin = min(scores)
        smax = max(scores)
        diff = smax - smin
        ret = [(s - smin) / diff for s in scores]
        return ret

    def get_scores(self, query_file, reply_file, generated_file, query_vocab_file, reply_vocab_file):
        ref_scores = self.ref.get_scores(self.data_dir, reply_file, generated_file)
        ref_scores = self._normalize(ref_scores)

        unref_scores = self.unref.get_scores(self.data_dir,
                                             query_file, generated_file, query_vocab_file, reply_vocab_file)
        unref_scores = self._normalize(unref_scores)

        return [min(a, b) for a, b in zip(ref_scores, unref_scores)]


