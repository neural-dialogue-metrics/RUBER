__author__ = 'liming-vie'

import os
import pickle

from tensorflow.contrib import learn


def tokenizer(iterator):
    for value in iterator:
        yield value.split()


def load_file(filename):
    print('Loading file %s' % (filename))
    lines = open(filename).readlines()
    return [line.rstrip() for line in lines]


def process_train_file(data_dir, filename, max_length, min_frequency=10):
    """
    Make vocabulary and transform into id files

    Return:
        vocab_file
        vocab_dict: map vocab to id
        vocab_size
    """
    vocab_file = '%s.vocab%d' % (filename, max_length)
    foutput = os.path.join(data_dir, vocab_file)
    if os.path.exists(foutput):
        print('Loading vocab from file %s' % foutput)
        vocab = load_vocab(vocab_file)
        return vocab_file, vocab, len(vocab)

    vocab_processor = learn.preprocessing.VocabularyProcessor(
        max_length,
        tokenizer_fn=tokenizer,
        min_frequency=min_frequency
    )

    x_text = load_file(filename)
    print('Vocabulary transforming')
    # will pad 0 for length < max_length
    ids = list(vocab_processor.fit_transform(x_text))
    print("Vocabulary size %d" % len(vocab_processor.vocabulary_))
    fid = os.path.join(data_dir, filename + '.id%d' % max_length)
    print('Saving %s ids file in %s' % (filename, fid))
    pickle.dump(ids, open(fid, 'wb'), protocol=2)

    print('Saving vocab file in %s' % foutput)
    size = len(vocab_processor.vocabulary_)
    vocab_str = [vocab_processor.vocabulary_.reverse(i) for i in range(size)]
    with open(foutput, 'w') as fout:
        fout.write('\n'.join(vocab_str))

    vocab = load_vocab(vocab_file)
    return vocab_file, vocab, len(vocab)


def load_data(filename):
    """
    Read id file data

    Return:
        data list: [[length, [token_ids]]]
    """
    print('Loading data from %s' % filename)
    ids = pickle.load(open(filename, 'rb'))
    data = []
    for vec in ids:
        length = len(vec)
        if vec[-1] == 0:
            length = list(vec).index(0)
        data.append([length, vec])
    return data


def load_vocab(vocab_file):
    """
    Load vocab
    """
    print('Loading vocab from %s' % vocab_file)
    vocab = {}
    with open(vocab_file) as fin:
        for i, s in enumerate(fin):
            vocab[s.rstrip()] = i
    return vocab


def transform_to_id(vocab, sentence, max_length):
    """
    Transform a sentence into id vector using vocab dict
    Return:
        length, ids
    """
    words = sentence.split()
    ret = [vocab.get(word, 0) for word in words]
    l = len(ret)
    l = max_length if l > max_length else l
    if l < max_length:
        ret.extend([0 for _ in range(max_length - l)])
    return l, ret[:max_length]


def make_embedding_matrix(data_dir, prefix, word2vec, vec_dim, vocab_file):
    foutput = os.path.join(data_dir, "%s.embed" % prefix)
    if os.path.exists(foutput):
        print('Loading embedding matrix from %s' % foutput)
        return pickle.load(open(foutput, 'rb'))

    vocab_str = load_file(vocab_file)
    print('Saving embedding matrix in %s' % foutput)
    matrix = []
    for vocab in vocab_str:
        vec = word2vec[vocab] if vocab in word2vec else [0.0 for _ in range(vec_dim)]
        matrix.append(vec)
    pickle.dump(matrix, open(foutput, 'wb'), protocol=2)
    return matrix


def load_word2vec(w2v_file):
    """
    Return:
        word2vec dict
        vector dimension
        dict size
    """
    print('Loading word2vec dict from %s' % w2v_file)
    vecs = {}
    with open(w2v_file) as fin:
        size, vec_dim = list(map(int, fin.readline().split()))
        for line in fin:
            ps = line.rstrip().split()
            vecs[ps[0]] = list(map(float, ps[1:]))
    return vecs, vec_dim, size
