from __future__ import absolute_import
from __future__ import print_function

import gzip
import csv
import time
import random
import numpy as np

CATEGORIES_FILE = 'datasets/wnCategories.tsv.gz'
RELATIONS_FILE = 'datasets/wnRelations.tsv.gz'


class Relations:
    class Word:
        def __init__(self):
            self.forward = dict()
            self.backward = dict()

    class Relation:
        def __init__(self, word1, relation, word2):
            self.word1 = word1
            self.relation = relation
            self.word2 = word2

        def resolve(self, dictionary):
            word1 = dictionary[self.word1]
            relation = dictionary[self.relation]
            word2 = dictionary[self.word2]
            return Relations.Relation(word1, relation, word2)

    def __init__(self, dictionary):
        self._dictionary = dictionary
        self._relations = dict()
        self._categories = dict()
        self._relation_dict = list()

    def get_num_relations(self):
        return len(self._relations)

    def get_num_words_with_categories(self):
        return len(self._categories)

    def add_to_dictionary(self, word):
        if not self._dictionary.has_word(word):
            self._dictionary.add_word(word)

    def add_category_to_dictionary(self, word, relation):
        """Adds category to dictionary and returns it's integer ID"""
        category = word + '_' + relation + '_cat'
        self.add_to_dictionary(category)
        return self._dictionary[category]

    def add_word_to_category(self, word, category):
        """Public method that allows adding words to category. Expects strings"""
        category += '_cat'
        self.add_to_dictionary(category)
        self._add_to_category(self._dictionary[word], self._dictionary[category])

    def _add_to_category(self, word, category):
        """Private method that adds a word into category. Expects IDs"""
        if word not in self._categories:
            self._categories[word] = list()
        self._categories[word].append(category)

    def _convert_to_category(self, relations, category, target, relation):
        """Private method that creates a new category to preserve 1-1 relationship mapping. Expects IDs"""
        old_word = relations[relation]
        self._add_to_category(old_word, category)
        self._add_to_category(target, category)
        relations[relation] = category

    def _get_word_relations(self, word):
        if word not in self._relations:
            self._relations[word] = self.Word()
        return self._relations[word]

    def add_relation(self, word1, relation, word2):
        if relation not in self._relation_dict:
            self._relation_dict.append(relation)
        relation += '_rel'
        self.add_to_dictionary(relation)
        self.add_to_dictionary(word1)
        self.add_to_dictionary(word2)
        relation_tuple = Relations.Relation(word1, relation, word2).resolve(self._dictionary)

        word1_rel = self._get_word_relations(relation_tuple.word1).forward
        word2_rel = self._get_word_relations(relation_tuple.word2).backward
        # If we've already saw this relation in same context it's a 1-n relationship, replacing with new category
        if relation_tuple.relation in word1_rel:
            resolved_category = self.add_category_to_dictionary(word1, relation)
            self._convert_to_category(word1_rel, resolved_category, relation_tuple.word2, relation_tuple.relation)
        else:
            word1_rel[relation_tuple.relation] = relation_tuple.word2

        if relation_tuple.relation in word2_rel:
            resolved_category = self.add_category_to_dictionary('back_' + word2, relation)
            self._convert_to_category(word2_rel, resolved_category, relation_tuple.word1, relation_tuple.relation)
        else:
            word2_rel[relation_tuple.relation] = relation_tuple.word1

    def generate_relational_batch(self, input_examples):
        batch = list()
        # Generating examples
        for word in input_examples[:, 0] + input_examples[:, 1]:
            if word not in self._relations:
                continue
            word_rel = self._relations[word]
            for relation in word_rel.forward.keys():
                batch.append([word, relation, word_rel.forward[relation]])
            for relation in word_rel.backward.keys():
                batch.append([word_rel.backward[relation], relation, word])
        # A batch should never be empty
        if len(batch) == 0:
            batch.append([0, 1, 0])
        # Generating corrupted examples
        indices = range(len(batch))
        for idx in indices:
            # Randomly select wrong source and target word for the same relation
            rand_idx = random.choice(indices)
            rand_word1 = batch[rand_idx][0]
            rand_word2 = batch[rand_idx][2]
            relation = batch[idx][1]

            if rand_word1 in self._relations:
                rel = self._relations[rand_word1].forward
            else:
                rel = self._relations[rand_word2].backward
            if relation in rel and (rel[relation] == rand_word2 or rel[relation] == rand_word1):
                # UNK + NULL_REL != NULL_REL (unless UNK is inferred to be zero, which wouldn't add to objective)
                batch[idx].extend([0, 1])
            else:
                batch[idx].extend([rand_word1, rand_word2])

        return self._arr(batch)

    def generate_categorical_batch(self, input_examples):
        batch = list()
        for word in input_examples[:, 0] + input_examples[:, 1]:
            if len(batch) >= input_examples.shape[0]:
                break
            if word in self._categories:
                for category in self._categories[word]:
                    batch.append([word, category])
        # A batch should never be empty
        if len(batch) == 0:
            batch.append([0, 0])
        return self._arr(batch)

    @staticmethod
    def _arr(arr):
        return np.array(arr, dtype=np.int32)

    def load_relations(self):
        count = 0
        timestamp = time.time()
        print('  Loading relations')
        with gzip.open(RELATIONS_FILE, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='')
            try:
                for row in reader:
                    if len(row) != 4:
                        print('Inconsistent line:', row)
                    word1 = row[1]
                    relation = row[2]
                    word2 = row[3]
                    self.add_relation(word1, relation, word2)
                    count += 1
            except csv.Error:
                print(u"Dunno why this error happens")
        print(count, 'relations parsed in', '%.2f' % (time.time() - timestamp), 'seconds')

    def load_categories(self, filename):
        print('  Loading categories')
        notfound = 0
        found = 0
        timestamp = time.time()
        for filename in [CATEGORIES_FILE, filename]:
            with gzip.open(filename, 'rb') as csvfile:
                reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='')
                try:
                    for row in reader:
                        if filename == CATEGORIES_FILE:
                            word = row[1]
                            categories = row[2:]
                        else:
                            word = row[0]
                            categories = row[1:]
                        if not self._dictionary.has_word(word):
                            notfound += 1
                            continue
                        found += 1
                        if found % 1000000 == 0:
                            print("  " + str(found // 1000000) + "m categories parsed")
                        for category in categories:
                            self.add_word_to_category(word, category)
                except csv.Error:
                    print(u"Dunno why this error happens")
        print(' ', found, 'word -> category entries parsed in', '%.2f' % (time.time() - timestamp), 'seconds')
        print(' ', notfound, 'entries skipped due to missing target word')