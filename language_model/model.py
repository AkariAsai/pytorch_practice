'''
Define RNNModel for language modeling.
This RNNModel takes five argument, and two optional parametes.
'''
import torch.nn as nn
from torch.autograd import Variable


class RNNModel(nn.Module):
    def __init__(self, rnn_type, n_token, n_input, n_hidden, n_layers,
                 dropout=0.5, tie_weight=False):
        super(RNNModel, self).__init__()
        # Training中にランダムに入力を一定のdropout rateで0にする。
        # http://pytorch.org/docs/master/nn.html#torch.nn.Dropout
        self.drop = nn.Dropput(dropout)
        self.encoder = nn.Embedding(n_token, n_input)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(
                n_input, n_hidden, n_layers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh',
                                'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")

            self.rnn = nn.RNN(n_input, n_hidden, n_layers,
                              nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(n_hidden, n_token)
        if tie_weights:
            if n_hidden != n_input:
                raise ValueError(
                    'When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()  # weightの初期化
        self.rnn_type = rnn_type
        self.n_hidden = n_hidden
        self.n_layers = n_layers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        embedding = self.drop(self.encoder(input))
        output, hidden = self.rnn(embedding, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(
            output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.n_layers, bsz, self.n_hidden).zero_()),
                    Variable(weight.new(self.n_layers, bsz, self.n_hidden).zero_()))
        else:
            return Variable(weight.new(self.n_layers, bsz, self.n_hidden).zero_())