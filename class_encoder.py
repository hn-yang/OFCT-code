import pickle


def class_encoder():

    # with open('/home/wangz/yanghn/MES/word_vectors.pkl', 'rb') as fp:
    with open('E:/byyq/word_vectors.pkl', 'rb') as fp:
        vocab = pickle.load(fp)
    vocab = vocab
    micro_vec = vocab['id2vec'][vocab['w2id']['micro']]
    expr_vec = vocab['id2vec'][vocab['w2id']['expression']]
    return micro_vec -expr_vec

if __name__ == '__main__':
    vec = class_encoder()
    print(vec)
