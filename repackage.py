import torch
import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    """Ref: https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/modules/sequence_modeling.py"""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size], T = num_steps.
        output : contextual feature [batch_size x T x output_size]
        """
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


class CRNN(nn.Module):

    def __init__(self, img_h, in_channels, nclass, nh, leaky_relu=False):
        super().__init__()
        assert img_h % 16 == 0, 'img_h has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = in_channels if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module(f'conv{i}',
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i], bias=not batchNormalization))
            if batchNormalization:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(nOut))
            if leaky_relu:
                cnn.add_module(f'relu{i}',
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module(f'relu{i}', nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling2',
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling3',
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16


        self.cnn = cnn

        self.out_channels = 512  # number of feature channels after CNN
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, 'the height of conv must be 1'
        conv = conv.squeeze(2)
        conv = conv.transpose(1, 2)  # [b, w, c]

        # rnn features
        output = self.rnn(conv)

        return output
model = CRNN(32,3,94,256)  # img_h, in_channels, nclass, nh
state_dict = torch.load('crnn-679d0e31.pt', map_location="cpu")

# Load weights
model.load_state_dict(state_dict)

# Now you can access layers
print(list(model.children())[0])