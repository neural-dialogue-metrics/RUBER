import os
import argparse
import numpy as np
from hybrid_evaluation import Hybrid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument()
    args = parser.parse_args()

    model = Hybrid(word2vec_file=args.w2v_file, query_w2v_file=args.query_w2v_file, reply_w2v_file=args.reply_w2v_file,
                   train_dir=args.train_dir, query_max_len=args.query_max_len, reply_max_len=args.reply_max_len,
                   pooling_type=args.pooling_type)

    scores = model.get_scores(
        query_file=os.path.basename(args.query_file),
        reply_file=os.path.basename(args.reply_file),
        generated_file=args.generated_file,
        query_vocab_file=args.query_vocab_file,
        reply_vocab_file=args.reply_vocab_file,
    )

    print(np.mean(scores))
