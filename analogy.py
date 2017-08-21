import gensim
import operator
import smart_open
from scipy import spatial
import sys
import re

# Returns the best guess from the provided possible answers using cosine similarity.
def bestGuess(embedding, question, possibleAnswers):
    questionVector = embedding.word_vec(question[1]) + embedding.word_vec(question[2]) - embedding.word_vec(question[0])
    vectorPairs = [
        [embedding.word_vec(guess[0]), questionVector] for guess in possibleAnswers
    ]
    similarities = list(map(lambda args: 1 - spatial.distance.cosine(args[0],args[1]), vectorPairs))
    idx, value = max(enumerate(similarities), key=operator.itemgetter(1))
    return idx

# Evaluates the emedding's ability to answer the question correctly.
# Returns true if the guess matches the answer, false otherwise.
def evaluate(embedding, question, answer):
    possibleAnswers = embedding.most_similar(positive=[question[1], question[2]], negative=[question[0]], topn=10)
    idx = bestGuess(embedding, question, possibleAnswers)
    return str(possibleAnswers[idx][0]) == answer

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

# Run the analogy test using the provided questions.
def analogy_test(embedding_name, embedding, questions_path):
    results = {}
    current_task = None
    current_task_results = {}
    for line in smart_open.smart_open(questions_path):
        line_str = str(line,'utf-8').strip()
        if embedding_name == "glove":
            line_str = line_str.lower()
        if line_str.startswith("//"):
            continue
        if line_str.startswith(": "):
            if current_task is not None:
                current_task_results["accuracy_w_missing"] = current_task_results["correct_answers"] / current_task_results["total_questions"]
                current_task_results["accuracy_wo_missing"] = current_task_results["correct_answers"] / (current_task_results["correct_answers"] + current_task_results["incorrect_answers"])
                print (current_task)
                print (current_task_results)
                print ("=====")
                results[current_task] = current_task_results
            current_task_results = {
                "total_questions": 0,
                "missing_words": 0,
                "correct_answers": 0,
                "incorrect_answers": 0,
                "accuracy_w_missing": 0,
                "accuracy_wo_missing": 0
            }
            current_task = line_str.split(": ")[1]
            continue

        current_task_results["total_questions"] += 1
        qna = re.compile('\w+').findall(line_str)

        try:
            if evaluate(embedding, (qna[0], qna[1], qna[2]), qna[3]) is True:
                current_task_results["correct_answers"] += 1
            else:
                current_task_results["incorrect_answers"] += 1
        except Exception as err:
            current_task_results["missing_words"] += 1
            print(err)

    current_task_results["accuracy_w_missing"] = current_task_results["correct_answers"] / current_task_results[
        "total_questions"]
    current_task_results["accuracy_wo_missing"] = current_task_results["correct_answers"] / (
    current_task_results["correct_answers"] + current_task_results["incorrect_answers"])
    print(current_task)
    print(current_task_results)
    print("=====")
    results[current_task] = current_task_results

    return results

# Where everything starts...
def main(embedding_name=None, embedding_path=None, questions_path=None):
    if embedding_name not in ["glove", "word2vec"] or embedding_path is None or questions_path is None:
        print("Please re-run as: python3 analogy.py <embedding-name> <embedding-path> <questions-path>")
        print("For example: python3 analogy.py glove /tmp/glove.6B.300d.txt /tmp/word-test.v1.txt")
        print("or: python3 analogy.py word2vec /tmp/GoogleNews-vectors-negative300.bin.gz /tmp/word-test.v1.txt")
        sys.exit(1)

    embedding = None
    if embedding_name == "glove":
        embedding = load_glove(embedding_path)
    else:
        embedding = load_word2vec(embedding_path)

    results = analogy_test(embedding_name, embedding, questions_path)

if __name__ == '__main__':
  main(sys.argv[1], sys.argv[2], sys.argv[3])