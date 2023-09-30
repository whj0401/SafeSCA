# borrowed from pseudo-GAN/models/layers.py
import torch.nn as nn
import torch

class DPCNNResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, padding, downsample, dropout=0.2):
        super(DPCNNResidualBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=channels, out_channels=channels,
                kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=channels, out_channels=channels,
                kernel_size=kernel_size, padding=padding)
        )
    def forward(self, x):
        output = self.residual(x)
        output = x + output
        # print(output.shape)
        return output


class DPCNNResidualBlockDownSample(nn.Module):
    def __init__(self, channels, kernel_size, padding, downsample, dropout=0.2):
        super(DPCNNResidualBlockDownSample, self).__init__()
        self.residual = nn.Sequential(
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=channels, out_channels=channels,
                kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=channels, out_channels=channels,
                kernel_size=kernel_size, padding=padding)
        )
        self.pool = nn.MaxPool1d(2)
        self.channels = channels
        # self.pool = nn.MaxPool2d(2)
        self.downsample = downsample
    def forward(self, x):
        in_shape = x.shape
        output = self.residual(x)
        output = x + output
        # we always do downsample in this class
        # non-downsampling cases use the other class
        # if self.downsample:
        #     # crypten only supports MaxPool2d now
        #     # output = torch.repeat_interleave(output, 2, dim=1)
        #     output = self.pool(output)
        # print(output.shape)
        output = self.pool(output)
        return output


class DPCNN(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, dropout):
        super(DPCNN, self).__init__()
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.cnn = nn.Conv1d(
            in_channels=embed_size, out_channels=hidden_size,
            kernel_size=1, padding=0
        )
        self.residual_layer = self._make_layer(num_layers, hidden_size,
                                               kernel_size=5, padding=2,
                                               downsample=True)
        # self.globalpool = nn.AdaptiveAvgPool2d((None, 1))
        self.globalpool = nn.AvgPool2d(1)

    def _make_layer(self, num_layers, channels,
                    kernel_size, padding, downsample):
        layers = []
        for _ in range(num_layers-1):
            layers.append(DPCNNResidualBlockDownSample(channels, kernel_size,
                                             padding, downsample, self.dropout))
        layers.append(DPCNNResidualBlock(channels, kernel_size,
                                         padding, downsample=False,
                                         dropout=self.dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        # embeds = x.permute(0, 2, 1)
        # bs = x.shape[0]
        output = self.cnn(x)
        output = self.residual_layer(output)
        # output = output.reshape((bs, 1, self.hidden_size, 1))
        # output = self.globalpool(output).squeeze(dim=(1, 3))
        return output.squeeze(dim=-1)


class CodeLabelingLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size, dropout=0.2):
        super(CodeLabelingLayer, self).__init__()
        self.dropout = dropout
        if num_hidden_layers > 0:
            self.in_layer = nn.Sequential(nn.Linear(input_size, hidden_size, bias=True),
                                          nn.ReLU())
            self.hidden_layers = self._make_hidden_layer(hidden_size, num_hidden_layers)
            self.out_layer = nn.Sequential(self.hidden_layers,
                                           nn.Linear(hidden_size, output_size, bias=True))
        else:
            self.in_layer = nn.Linear(input_size, hidden_size, bias=True)
            self.out_layer = nn.Linear(hidden_size, output_size, bias=True)

    def _make_hidden_layer(self, hidden_size, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size, bias=True))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.in_layer(x)
        output = self.out_layer(output)
        return output


class DPCNN4Code1(nn.Module):
    def __init__(self, token_ebd_size, vocab_size, code_ebd_size, padding_value, num_layers=12, num_labels=1024, dropout=0.2):
        super(DPCNN4Code1, self).__init__()
        self.code_embedding = nn.Embedding(vocab_size, token_ebd_size, padding_idx=padding_value)
        self.dpcnn = DPCNN(token_ebd_size, code_ebd_size, num_layers=num_layers, dropout=dropout)
        self.output_layer = CodeLabelingLayer(code_ebd_size, 1024, 1, num_labels, dropout=dropout)
        # self.output_layer = nn.Linear(code_ebd_size, num_labels, bias=True)
        # self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1/0.07))

    def forward(self, input, padding=True):
        # assert padding is True or padding in ['max_length', 'longest']
        output = self.code_embedding(input)
        output = self.dpcnn(output.permute(0, 2, 1))
        output = self.output_layer(output)
        return output
        # embed = torch.nn.function_normalizer(embed)
        # return embed


class DPCNN4Code2(nn.Module):
    def __init__(self, token_ebd_size, vocab_size, code_ebd_size, padding_value, num_layers=12, num_labels=1024, dropout=0.2):
        super(DPCNN4Code2, self).__init__()
        self.code_embedding = nn.Embedding(vocab_size, token_ebd_size, padding_idx=padding_value)
        self.dpcnn = DPCNN(token_ebd_size, code_ebd_size, num_layers=num_layers, dropout=dropout)
        # self.output_layer = CodeLabelingLayer(code_ebd_size, 1024, 1, num_labels, dropout=dropout)
        self.output_layer = nn.Linear(code_ebd_size, num_labels, bias=True)
        # self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1/0.07))

    def forward(self, input, padding=True):
        # assert padding is True or padding in ['max_length', 'longest']
        output = self.code_embedding(input)
        output = self.dpcnn(output.permute(0, 2, 1))
        output = self.output_layer(output)
        return output
        # embed = torch.nn.function_normalizer(embed)
        # return embed

class Config:
    def __init__(self):
        self.num_embeddings = 527683
        self.embedding_size = 128  # dimension of the function embedding

        ## RNN PARAMETERS, these parameters are only used for RNN model.
        self.rnn_state_size = 50  # dimesion of the rnn state
        self.rnn_depth = 1  # depth of the rnn
        # self.max_instructions = 150  # number of instructions
        self.residual_layers = 12
        self.max_instructions = 2**(self.residual_layers - 1)  # a large number for TOKEN-based CNN model

        ## ATTENTION PARAMETERS
        self.attention_hops = 10
        self.attention_depth = 250

        # RNN SINGLE PARAMETER
        self.dense_layer_size = 2000

        self.i2t_path = 'dpcnn4bin/checkpoints/i2t.json'


class SAFE2(nn.Module):
    def __init__(self, conf):
        super(SAFE2, self).__init__()
        self.conf = conf
        self.instructions_embeddings = torch.nn.Embedding(
            self.conf.num_embeddings, self.conf.embedding_size
        )
        self.dpcnn = DPCNN(embed_size=self.conf.embedding_size,
                           hidden_size=self.conf.embedding_size,
                           num_layers=self.conf.residual_layers,
                           dropout=0.0)

    def forward(self, input):
        output = self.instructions_embeddings(input)
        output = self.dpcnn(output.permute(0, 2, 1))
        return output

