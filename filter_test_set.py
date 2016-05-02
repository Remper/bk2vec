from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from params import TEXTS

from bk2vec.textreader import *
from bk2vec.arguments import FilterArguments
from bk2vec.evaluation import EvaluationDumper

args = FilterArguments().show_args().args
text_reader = TextReader(TEXTS)

def restore_evaluation(filename):
    pages = dict()
    categories = 0
    with gzip.open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='')
        for row in reader:
            if len(row) == 0:
                continue
            index = int(row[0])
            pages[index] = np.array(map(int, row[1:]))
            categories += len(pages[index])
    print("Loaded", len(pages.values()), "pages with", categories, "categories")
    return pages

print("Loading dictionary")
timestamp = time.time()
dictionary = text_reader.restore_dictionary()
print("Dictionary restored in", ("%.5f" % (time.time() - timestamp)) + "s")
if not os.path.exists(TEXTS + '_counts'):
    print("Restoring word counts from the original corpus")
    timestamp = time.time()
    _, counts = text_reader.build_dictionary()
    print("Counts restored in", ("%.5f" % (time.time() - timestamp)) + "s")
    print("Saving counts to disk")
    with gzip.open(TEXTS + '_counts', 'wb') as writer:
        count = 0
        for word in counts.keys():
            count += 1
            writer.write(word)
            writer.write('\t')
            writer.write(str(counts[word]))
            writer.write('\n')
            if count % 1000000 == 0:
                print("  ", str(count // 1000000) + "m words saved")
else:
    counts = dict()
    with gzip.open(TEXTS + '_counts', 'rb') as csvfile:
        count = 0
        reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='')
        for row in reader:
            count += 1
            if len(row) != 2:
                print("Shouldn't have happened")
                exit()
            counts[row[0]] = int(row[1])
            if count % 1000000 == 0:
                print("  ", str(count // 1000000) + "m words parsed")

print("Loading test set")
pages = restore_evaluation(args.test)

print("Filtering test set and computing average frequency")
filtered_pages = dict()
freq_sum = 0
freq_count = 0
examples = dict()
for page in pages:
    if page not in dictionary.rev_dict:
        print(page, "not found")
        continue
    word = dictionary.rev_dict[page]
    freq_sum += counts[word]
    freq_count += 1
    if counts[word] < 20:
        filtered_pages[page] = pages[page]
        for category in pages[page]:
            if category not in examples:
                examples[category] = list()
            if len(examples[category]) < 1000 and page not in examples[category]:
                examples[category].append(page)

if not os.path.exists(str(args.output)+'category_dump/'):
    os.makedirs(str(args.output)+'category_dump/')
    print("Created output directory")
print("Examples of filtered categories:")
count = 0
for category in examples.keys():
    if len(examples[category]) < 15:
        continue
    count += 1
    if count <= 10:
        try:
            print(", ".join([dictionary.rev_dict[ele] for ele in examples[category]][:30]))
        except:
            print("Error happened: ", examples[category])
    with open(str(args.output)+'category_dump/'+str(category)+'.txt', 'wb') as file:
        for ele in examples[category]:
            file.write(str(ele))
            file.write('\t')
            file.write(dictionary.rev_dict[ele])
            file.write('\n')
print("Average page frequency:", float(freq_sum)/freq_count)
print("Storing the filtered test set")
EvaluationDumper(filtered_pages, "test-filtered", folder=args.output).run()
