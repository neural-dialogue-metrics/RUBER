import os
import argparse
from hybrid_evaluation import Hybrid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument()
    args = parser.parse_args()


    model = Hybrid(
        train_dir=args.train_dir,
        data_dir=os.path.basename(args),
        word2vec_file=args.w2v_file,
        query_w2v_file=args.query_w2v_file,
        reply_w2v_file=args.reply_w2v_file,
        query_max_len=args.query_max_len,
        reply_max_len=args.reply_max_len,
        pooling_type=args.pooling_type,
    )

    scores = model.get_scores(
        query_file=os.path.basename(args.query_file),
        reply_file=os.path.basename(args.reply_file),
        generated_file=
    )