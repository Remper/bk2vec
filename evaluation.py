from __future__ import absolute_import
from __future__ import print_function

from bk2vec.embeddings import Embeddings
from bk2vec.evaluation import *
from bk2vec.utils import *
from bk2vec.arguments import EvaluationArguments

args = EvaluationArguments().show_args().args

print("Restoring embeddings")
embeddings, dictionary = Embeddings.restore(args.embeddings)
print("Restored embeddings with shape:", embeddings.shape)
relations = Analogy.from_file("datasets/questions-words.txt", dictionary.dict)

thread = AnalogyCalculation(relations, embeddings, Log(args))
thread.start()
thread.join()
for result in thread.results[:100]:
    print("")
    print("Sample: ", [dictionary.rev_dict[ele] for ele in result[:4]])
    print("Candidates: ", [dictionary.rev_dict[ele] for ele in result[6:]])
    print("Relation norms: ", "%.2f" % result[4], "%.2f" % result[5])