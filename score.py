import argparse
import numpy as np
from hybrid_evaluation import Hybrid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_dir', required=True)
    parser.add_argument('-query_max_len', default=20, type=int, help='max length of query')
    parser.add_argument('-reply_max_len', default=30, type=int, help='max length of reply')
    parser.add_argument('-w2v_file', help='word2vec in text format')
    parser.add_argument('-pooling_type', default='max_min', choices=(
        'max_min',
        'avg',
        'all',
    ))
    parser.add_argument('-query_w2v_file', help='query embedding file', required=True)
    parser.add_argument('-reply_w2v_file', help='reply embedding file', required=True)
    parser.add_argument('-query_file', required=True)
    parser.add_argument('-reply_file', required=True)
    parser.add_argument('-query_vocab_file', required=True)
    parser.add_argument('-reply_vocab_file', required=True)
    parser.add_argument('-generated_file', required=True)
    args = parser.parse_args()

    model = Hybrid(
        word2vec_file=args.w2v_file,
        query_w2v_file=args.query_w2v_file,
        reply_w2v_file=args.reply_w2v_file,
        train_dir=args.train_dir,
        query_max_len=args.query_max_len,
        reply_max_len=args.reply_max_len,
        pooling_type=args.pooling_type,
    )

    scores = model.get_scores(
        query_file=args.query_file,
        reply_file=args.reply_file,
        generated_file=args.generated_file,
        query_vocab_file=args.query_vocab_file,
        reply_vocab_file=args.reply_vocab_file,
    )

    print(np.mean(scores))
