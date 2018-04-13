from __future__ import unicode_literals, print_function, division
from io import open
import glob
import unicodedata
import string
import torch

all_letters = string.ascii_letters + ".,;''"
n_letters = len(all_letters)


class Dataset(object):
    def __init__(self, path='data/names/*.txt'):
        self.data_path = path
        self.all_letters = all_letters
        self.n_letters = n_letters
        self.category_lines = {}
        self.all_categories = []
        self.n_categories = 0

    def findFiles(self):
        return glob.glob(self.data_path)

    def read_name_data(self):
        for filename in self.findFiles():
            category = filename.split('/')[-1].split('.')[0]
            self.all_categories.append(category)
            lines = readLines(filename)
            self.category_lines[category] = lines

        self.n_categories = len(self.all_categories)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


def letterToIndex(letter):
    return all_letters.find(letter)


def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor
