"""
ConvLSTM AutoEncoder
Tensor-2-Tensor self embedding model.
Input the video tensor and reconstruction tensor itself.

512, 1024, or 2048 are common sizes for the embedding vectors
reference: http://www.cs.toronto.edu/~nitish/unsup_video.pdf
https://holmdk.github.io/2020/04/02/video_prediction.html
""" 

import torch
from torch import nn

# 3D ConvLSTM Cell
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, shrink=1):
        super(ConvLSTMCell, self).__init__()
        self.channels = 3
        self.H = 1024//shrink
        self.W = 576//shrink
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding)
        
    def forward(self, x: torch.Tensor, current_state):
        h_current, c_current = current_state
        combined = torch.cat([x, h_current], dim=1)  # dim 1 for channels

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_current + i * g
        h_next = o * torch.tanh(c_next)
        return c_next, h_next 

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_dim, self.H, self.W, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, self.H, self.W, device=self.conv.weight.device))


# ConvLSTM Autoencoder
class ConvLSTMAutoencoder(nn.Module):
    def __init__(self, hidden_dim, shrink=1, normalize=False):
        super(ConvLSTMAutoencoder, self).__init__()
        self.channels = 3
        self.hidden_dim = hidden_dim
        self.encoder_1 = ConvLSTMCell(input_dim=self.channels,hidden_dim=hidden_dim,kernel_size=(3, 3),shrink=shrink)
        self.encoder_2 = ConvLSTMCell(input_dim=hidden_dim,hidden_dim=hidden_dim,kernel_size=(3, 3),shrink=shrink)
        self.decoder_1 = ConvLSTMCell(input_dim=hidden_dim,hidden_dim=hidden_dim,kernel_size=(3, 3),shrink=shrink)
        self.decoder_2 = ConvLSTMCell(input_dim=hidden_dim,hidden_dim=hidden_dim,kernel_size=(3, 3),shrink=shrink)
        self.decoder_CNN = nn.Conv3d(in_channels=hidden_dim,out_channels=self.channels,kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.normalize = normalize
    def encoder(self, x, seq_len, h_t, c_t, h_t2, c_t2):
        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1(x=x[:, :,t, :, :], current_state=[h_t, c_t])
            h_t2, c_t2 = self.encoder_2(x=h_t, current_state=[h_t2, c_t2])

        return h_t2
    
    def decoder(self, future_step, encoder_vector, h_t3, c_t3, h_t4, c_t4):
        outputs=[]
        # decoder
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1(x=encoder_vector, current_state=[h_t3, c_t3])
            h_t4, c_t4 = self.decoder_2(x=h_t3,current_state=[h_t4, c_t4])
            encoder_vector = h_t4
            outputs += [h_t4]  # predictions
        return outputs
    
    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        # encoder_vector
        encoder_vector = self.encoder(x, seq_len, h_t, c_t, h_t2, c_t2)
        outputs = self.decoder(future_step, encoder_vector, h_t3, c_t3, h_t4, c_t4)

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Sigmoid()(outputs)
        if not self.normalize:
            outputs = outputs * 255
        return outputs
    
    def forward(self, x: torch.Tensor, future_seq=None, hidden_state=None):
        
        B, C, seq_len, H, W = x.size()

        if future_seq is None:
            future_seq = seq_len
        # initialize hidden states
        h_t, c_t = self.encoder_1.init_hidden(batch_size=B)
        h_t2, c_t2 = self.encoder_2.init_hidden(batch_size=B)
        h_t3, c_t3 = self.decoder_1.init_hidden(batch_size=B)
        h_t4, c_t4 = self.decoder_2.init_hidden(batch_size=B)

        # autoencoder forward
        outputs = self.autoencoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)

        return outputs
    
    def getembedding(self, x: torch.Tensor, future_seq=None, hidden_state=None):

        B, C, seq_len, H, W = x.size()

        if future_seq is None:
            future_seq = seq_len
        # initialize hidden states
        h_t, c_t = self.encoder_1.init_hidden(batch_size=B)
        h_t2, c_t2 = self.encoder_2.init_hidden(batch_size=B)
        
        # encoder forward
        outputs = self.encoder(x, seq_len, h_t, c_t, h_t2, c_t2)

        return outputs
