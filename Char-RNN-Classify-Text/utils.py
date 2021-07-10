from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import time
import math
import random
import torch
import matplotlib.pyplot as plt

ASCII_LETTERS = string.ascii_letters + " .,;'"
UNIQUE_CHARS = len(ASCII_LETTERS)

def find_files(path):
    return glob.glob(path)

def unicode_to_ascii(s):
    """
    Remove accent from unicode - https://stackoverflow.com/a/518232

    e.g. - Ślusàrski -> Slusarski
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in ASCII_LETTERS)

def read_lines(file):
    lines = open(file, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

def get_data(path):
    categories = []
    category_lines = {}

    for file in find_files(path):
        category = os.path.basename(file)[:-4]
        categories.append(category)
        lines = read_lines(file)
        category_lines[category] = lines

    return categories, category_lines

def letter_to_index(letter):
    return ASCII_LETTERS.find(letter)

def letter_to_tensor(letter):
    tensor = torch.zeros(1, UNIQUE_CHARS)
    tensor[0][letter_to_index(letter)] = 1
    return tensor

def line_to_tensor(line):
    line_arr = []
    for letter in line:
        line_arr.append(letter_to_tensor(letter))
    
    line_tensor = torch.cat(line_arr, dim=0).view(len(line), 1, -1)
    return line_tensor

def category_from_output(output, categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return categories[category_i], category_i


def random_choice(l):
    return l[random.randint(0, len(l) - 1)]

def random_training_example(categories, category_lines):
    category = random_choice(categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def plot_loss(all_losses):
    plt.figure()
    plt.plot(all_losses)
    plt.show()
