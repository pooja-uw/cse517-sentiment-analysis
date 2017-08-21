import gensim
import smart_open
from scipy import spatial
import sys

# Returns the top 10 similar words in the order of most similar to least similar.
def findTop10Similar(embedding, word):
    similarWords = embedding.most_similar(positive=[word], topn=20)
    wordVector = embedding.word_vec(word)
    vectorPairs = [
        [embedding.word_vec(similarWord[0]), wordVector] for similarWord in similarWords
    ]
    similarities = list(map(lambda args: 1 - spatial.distance.cosine(args[0],args[1]), vectorPairs))
    indices = sorted(range(len(similarities)), key=lambda i: similarities[i])[-10:]
    return [similarWords[idx][0] for idx in indices][::-1]

# GloVe needs some.. love.. before it can be handled by Gensim.
# Prepending it with dimensions and number of words so Gensim can load GloVe.
def preprocess_glove(glove_path):
    line_count = sum(1 for line in smart_open.smart_open(glove_path))
    # Make this more robust.
    dimension = glove_path.split('.')[2].split('d')[0]
    with open(glove_path, 'r', encoding="utf8") as original: data = original.read()
    with open(glove_path + "_processed", 'w', encoding="utf8") as modified: modified.write(
        "{} {}".format(line_count, dimension) + '\n' + data)

# Load GloVe embedding.
def load_glove(glove_path):
    preprocess_glove(glove_path)
    return gensim.models.KeyedVectors.load_word2vec_format(glove_path + "_processed")

# Load Word2Vec embedding.
def load_word2vec(word2vec_path):
    return gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

# Where everything starts...
def main(embedding_name=None, embedding_path=None):
    if embedding_name not in ["glove", "word2vec"] or embedding_path is None:
        print("Please re-run as: python3 antonyms.py <embedding-name> <embedding-path>")
        print("For example: python3 antonyms.py glove /tmp/glove.6B.300d.txt")
        print("or: python3 antonyms.py word2vec /tmp/GoogleNews-vectors-negative300.bin.gz")
        sys.exit(1)

    print("Loading embedding..")

    embedding = None
    if embedding_name == "glove":
        embedding = load_glove(embedding_path)
    else:
        embedding = load_word2vec(embedding_path)

    while(True):
        word = input("Enter a word to see top 10 similar words (may include antonyms) or hit CTRL+C to quit: ")
        if embedding_name == "glove":
            word = word.lower()
        print(findTop10Similar(embedding, word))

if __name__ == '__main__':
  main(sys.argv[1], sys.argv[2])