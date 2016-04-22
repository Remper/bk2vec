from __future__ import absolute_import
from __future__ import print_function

import gzip
import csv
import os
import random

# Increasing limit for CSV parser
csv.field_size_limit(2147483647)

class TextReader():
  def __init__(self, filename, type='gzip'):
    self.filename = filename
    self.type = type

  def words(self):
    previous = list()
    with self._get_handler(self.filename) as csvfile:
      reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='')
      try:
        for row in reader:
          if (len(row) is not 2):
            print("Inconsistency in relations file. Previous:", previous, "Current:", row)
            continue
          previous = row
          for word in [row[0]] + row[1].split():
            yield word
      except csv.Error:
        print(u"Dunno why this error happens")

  def endless_words(self):
    while True:
      for word in self.words():
        yield word

  def build_dictionary(self):
    print("Building dictionary")
    dictionary = Dictionary()
    processed = 0
    for word in self.words():
      word = str(word)
      processed += 1
      if processed % 100000000 == 0:
        print("  " + str(processed // 1000000) + "m words parsed (last:", word, ", dict size:", len(dictionary), ")")
      if dictionary.has_word(word):
        continue
      dictionary.add_word(word)
    print("Parsing finished")
    return dictionary

  def restore_dictionary(self):
    print('Restoring dictionary')
    processed = 0
    dictionary = Dictionary()
    with self._get_handler(self.filename + "_dict") as csvfile:
      reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='')
      for row in reader:
        row[0] = str(row[0])
        row[1] = int(row[1])
        processed += 1
        if processed % 3000000 == 0:
          print("  " + str(processed // 1000000) + "m words parsed")
        dictionary.put_word(row[1], row[0])
    print('Done')
    return dictionary

  def store_dictionary(self, dictionary):
    print('Storing dictionary')
    dict = dictionary.dict
    with self._get_handler(self.filename + "_dict", 'wb') as csvfile:
      writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='')
      for value in dict.keys():
        writer.writerow([value, dict[value]])
    print('Done')

  def load_dictionary(self):
    if os.path.exists(self.filename + "_dict"):
      dictionary = self.restore_dictionary()
    else:
      dictionary = self.build_dictionary()
      self.store_dictionary(dictionary)
    return dictionary

  def _get_handler(self, filename, mode='rb'):
    if self.type == 'gzip':
      return gzip.open(filename, mode)
    if self.type == 'plain':
      return open(filename, mode)


class Dictionary():
  def __init__(self):
    self.unknown = 'UNK'
    self.dict = {self.unknown: 0}
    self.rev_dict = {0: self.unknown}

  def __len__(self):
    return len(self.dict)

  def add_word(self, word):
    self.put_word(len(self.dict), word)

  def put_word(self, index, word):
    self.dict[word] = index
    self.rev_dict[index] = word

  def has_word(self, word):
    return word in self.dict

  def __getitem__(self, item):
    return self.lookup(item)

  def lookup(self, word):
    if not self.has_word(word):
      word = self.unknown
    return self.dict[word]

# Page builder parameters
max_pages = 0
max_categories_per_page = 0
test_set = 0.1

def build_pages(filename, dictionary, reverse_dictionary):
  print('Building page -> category dictionary')
  global max_pages, max_categories_per_page, test_set
  pages = dict()
  evaluation = dict()
  maxPagesTitle = "Unknown"
  maxPages = 0
  found = 0
  notfound = 0
  category_found = 0
  category_notfound = 0
  test_set_size = 0
  training_set_size = 0
  with gzip.open(filename, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='')
    try:
      for row in reader:
        page_title = row[0]
        if page_title not in dictionary:
          notfound += 1
          continue
        found += 1
        if max_pages > 0 and found > max_pages:
          break
        if max_categories_per_page > 0 and len(row) > max_categories_per_page+1:
          row = row[:max_categories_per_page+1]
        if found % 1000000 == 0:
          print("  " + str(found // 1000000) + "m pages parsed")
        page_index = dictionary[page_title]
        if page_index not in pages:
          pages[page_index] = list()
        if page_index not in evaluation:
          evaluation[page_index] = list()
        page_categories = pages[page_index]
        evaluation_current = evaluation[page_index]
        for word in row[1:]:
          word = word+"_cat"
          if word not in dictionary:
            dictionary[word] = len(dictionary)
            reverse_dictionary[dictionary[word]] = word
            category_notfound += 1
            continue
          category_found += 1
          if test_set > 0 and random.random() <= test_set:
            test_set_size += 1
            evaluation_current.append(dictionary[word])
          else:
            training_set_size += 1
            page_categories.append(dictionary[word])
        if len(page_categories) > maxPages:
          maxPages = len(page_categories)
          maxPagesTitle = page_title
    except csv.Error:
      print(u"Dunno why this error happens")
  print(len(pages), "pages parsed.", "Page with most categories: ", maxPagesTitle, "with", maxPages, "categories")
  print("Training set size:", training_set_size, "Test set size:", test_set_size)
  print("Pages found:", found, "Pages not found:", notfound)
  print("Categories found:", category_found, "Categories not found:", category_notfound)
  return pages, evaluation
