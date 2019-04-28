import argparse
from unreferenced_metric import Unreferenced

GRU_NUM_UNITS = 128
MLP_UNITS = (256, 512, 128)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_dir', required=True)
    parser.add_argument('-query_file', required=True)
    parser.add_argument('-reply_file', required=True)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-checkpoint_freq', help='checkpoint frequency', type=int, default=100)
    parser.add_argument('-query_max_len', default=20, type=int, help='max length of query')
    parser.add_argument('-reply_max_len', default=30, type=int, help='max length of reply')
    parser.add_argument('-query_embed_file', help='query embedding file')
    parser.add_argument('-reply_embed_file', help='reply embedding file')
    args = parser.parse_args()

    model = Unreferenced(
        qmax_length=args.query_max_len,
        rmax_length=args.reply_max_len,
        fqembed=args.query_embed_file,
        frembed=args.reply_embed_file,
        gru_num_units=GRU_NUM_UNITS,
        mlp_units=MLP_UNITS,
    )

    model.train(
        fquery=args.query_file,
        freply=args.reply_file,
        batch_size=args.batch_size,
        steps_per_checkpoint=args.checkpoint_freq,
    )
