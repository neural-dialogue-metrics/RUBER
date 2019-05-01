# Train and save word2vec on the target corpus.
# The pretrained one suffer from OOV problem (especially on Ubuntu).

import argparse
# from gensim.models

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
