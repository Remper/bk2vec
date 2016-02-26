import os
import logging
import gensim


class TextReader(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        logger = logging.getLogger()
        logger.info("Started iterating over texts")
        counter = 0
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                for sentence in line.split("."):
                    yield sentence.strip().split()
                counter += 1
                if counter % 100000 == 0:
                    logger.info("Processed "+str(counter)+" texts")


def main():
    logging.basicConfig(level=logging.DEBUG)

    texts = TextReader("corpora")
    model = gensim.models.Word2Vec(texts, workers=8)
    model.save_word2vec_format("model.w2v")
    print(model.accuracy('questions-words.txt'))

if __name__ == "__main__":
    main()
