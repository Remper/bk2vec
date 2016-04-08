# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function

import random
import csv

numbers = range(100)
test = random.sample(numbers, 10)
reverse_dict = {}
dict = {}

with open('numbers_test.tsv', 'wb') as csvfile:
  writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
  for num in test:
    writer.writerow([str(num), 'nums'])

with open('numbers.tsv', 'wb') as csvfile:
  writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
  for num in numbers:
    if num not in test:
      writer.writerow([str(num), 'nums'])