__author__ = 'liming-vie'

import logging
from referenced_metric import Referenced
from unreferenced_metric import Unreferenced

logger = logging.getLogger(__name__)


class Hybrid(object):
    def __init__(self,
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
        logger.info('creating ref model')
        self.ref = Referenced(word2vec_file, pooling_type)

        logger.info('creating unref model')
        self.unref = Unreferenced(query_max_len, reply_max_len,
                                  query_w2v_file,
                                  reply_w2v_file,
                                  gru_units, mlp_units,
                                  train_dir=train_dir)

    def train_unref(self, query_file, reply_file):
        logger.info('training unref model')
        self.unref.train(query_file, reply_file)

    def _normalize(self, scores):
        smin = min(scores)
        smax = max(scores)
        diff = smax - smin
        ret = [(s - smin) / diff for s in scores]
        return ret

    def get_ref_scores(self, reply_file, generated_file):
        logger.info('computing ref_scores')
        ref_scores = self.ref.get_scores(reply_file, generated_file)
        ref_scores = self._normalize(ref_scores)
        return ref_scores

    def get_unref_scores(self, generated_file, query_file, query_vocab_file, reply_vocab_file):
        logger.info('computing unref_scores')
        unref_scores = self.unref.get_scores(query_file, generated_file, query_vocab_file, reply_vocab_file)
        unref_scores = self._normalize(unref_scores)
        return unref_scores

    def get_scores(self, query_file, reply_file, generated_file, query_vocab_file, reply_vocab_file):
        ref_scores = self.get_ref_scores(reply_file, generated_file)
        unref_scores = self.get_unref_scores(generated_file, query_file, query_vocab_file, reply_vocab_file)
        # min() combiner.
        return [min(a, b) for a, b in zip(ref_scores, unref_scores)]
