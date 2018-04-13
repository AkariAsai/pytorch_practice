import torch
import torch.nn as nn
from torch.autograd import Variable
from model import RNN
import random
import time
import math
from data import Dataset, lineToTensor
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

learning_rate = 0.005
n_hidden = 128

n_iters = 100000
print_every = 5000
plot_every = 1000

current_loss = 0
all_losses = []
criterion = nn.NLLLoss()


def categoryFromOutput(output, dataset):
    # Tensor.topk to get the index of the greatest value
    top_n, top_i = output.data.topk(1)  # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return dataset.all_categories[category_i], category_i


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample(all_categories, category_lines):
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = Variable(torch.LongTensor(
        [all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor


# This is a function to train the RNN for the classification of names.
def train(rnn, category_tensor, line_tensor):
    hidden = rnn.initHidden()
    rnn.zero_grad()

    for i in range(line_tensor.size(0)):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0]


def evalute(rnn, line_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60

    return '%dm %ds' % (m, s)


def main():
    dataset = Dataset()
    dataset.read_name_data()
    rnn = RNN(dataset.n_letters, n_hidden, dataset.n_categories)

    start = time.time()
    current_loss = 0
    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = \
            randomTrainingExample(dataset.all_categories,
                                  dataset.category_lines)
        output, loss = train(rnn, category_tensor, line_tensor)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output, dataset)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters *
                                                    100, timeSince(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    # Plotting the historical loss
    plt.figure()
    plt.plot(all_losses)
    plt.title("The negative log likelihood(NLL) loss per iter")
    plt.xlabel("n_iter")
    plt.ylabel("NLL loss")
    plt.show()

    # Evaluate the trained RNN.
    confusion = torch.zeros(dataset.n_categories, dataset.n_categories)
    n_confusion = 10000

    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = \
            randomTrainingExample(dataset.all_categories,
                                  dataset.category_lines)

        output = evalute(rnn, line_tensor)
        guess, guess_i = categoryFromOutput(output, dataset)
        category_i = dataset.all_categories.index(category)
        confusion[category_i][guess_i] += 1

    for i in range(dataset.n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + dataset.all_categories, rotation=90)
    ax.set_yticklabels([''] + dataset.all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()


if __name__ == "__main__":
    main()
