from __future__ import unicode_literals, print_function, division
import torch
from torch import nn
from torch import optim
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from model import RNN
from utils import * 

N_ITERS = 5000 # 100000
PRINT_EVERY = 5000
PLOT_EVERY = 1000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_HIDDEN = 128
N_CATEGORIES = None


def train_one_epoch(rnn, category_tensor, line_tensor, criterion, optimizer):
    hidden = rnn.init_hidden()
    hidden = hidden.to(DEVICE)
    optimizer.zero_grad()

    for i in range(line_tensor.shape[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()

    return output, loss.item()

def train(rnn, categories, category_lines, all_losses):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(rnn.parameters(), lr=0.0005)

    start = time.time()

    # Keep track of losses for plotting
    current_loss = 0

    for iter in range(1, N_ITERS + 1):
        category, line, category_tensor, line_tensor = random_training_example(categories, category_lines)
        category_tensor = category_tensor.to(DEVICE)
        line_tensor = line_tensor.to(DEVICE)
        output, loss = train_one_epoch(rnn, category_tensor, line_tensor, criterion, optimizer)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % PRINT_EVERY == 0:
            guess, guess_i = category_from_output(output, categories)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / N_ITERS * 100, time_since(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % PLOT_EVERY == 0:
            all_losses.append(current_loss / PLOT_EVERY)
            current_loss = 0

    torch.save(rnn.state_dict(), 'char-rnn_model.pth')


def evaluate(line_tensor):
    # Just return an output given a line
    rnn = RNN(UNIQUE_CHARS, N_HIDDEN, N_CATEGORIES)
    hidden = rnn.init_hidden()
    rnn.load_state_dict(torch.load('char-rnn_model.pth'))
    rnn.eval()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

def plot_confusion_matrix(categories, category_lines):
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(N_CATEGORIES, N_CATEGORIES)
    n_confusion = 10000
    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = random_training_example(categories, category_lines)
        output = evaluate(line_tensor)
        guess, guess_i = category_from_output(output, categories)
        category_i = categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(N_CATEGORIES):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + categories, rotation=90)
    ax.set_yticklabels([''] + categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()


def predict(input_line, categories, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(line_to_tensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, categories[category_index]))
            predictions.append([value, categories[category_index]])

def main():
    categories, category_lines = get_data('data/names/*.txt')
    global N_CATEGORIES
    N_CATEGORIES = len(categories)
    all_losses = []

    rnn = RNN(UNIQUE_CHARS, N_HIDDEN, N_CATEGORIES)
    rnn = rnn.to(DEVICE)

    train(rnn, categories, category_lines, all_losses)

    plot_loss(all_losses)

    plot_confusion_matrix(categories, category_lines)

    predict('Dovesky', categories)
    predict('Jackson', categories)
    predict('Satoshi', categories)

if __name__ == "__main__":
    main()