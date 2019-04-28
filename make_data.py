import argparse
import os

import data_helpers

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', required=True, help='where the output data will be stored')
    parser.add_argument('-w2v_file', required=True, help='word2vec in text format')
    parser.add_argument('-max_len', required=True, type=int, help='max length of query')
    parser.add_argument('-min_freq', type=int, default=10, help='min frequency of word')
    parser.add_argument('-raw_input', required=True,
                        help='either reply or query file; one utterance per line')
    args = parser.parse_args()

    vocab_file, _, _ = data_helpers.process_train_file(
        data_dir=args.data_dir,
        filename=args.raw_input,
        max_length=args.max_len,
        min_frequency=args.min_freq,
    )

    w2v, vec_dim, _ = data_helpers.load_word2vec(args.w2v_file)
    data_helpers.make_embedding_matrix(
        data_dir=args.data_dir,
        prefix=os.path.basename(args.raw_input),
        word2vec=w2v,
        vec_dim=vec_dim,
        vocab_file=vocab_file,
    )
