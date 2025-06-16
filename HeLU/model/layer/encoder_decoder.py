import torch
from torch import nn
from HeLU.model.layer.clones import clones
from .layer_norm import LayerNorm


class Encoder(nn.Module):
    "Core encoder is a stack of N layer"

    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers: # run forward in EncoderLayer
            #            print("Encoder:",x)
            x = layer(x, mask)
        #            print("Encoder:",x.size())
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, price_series_mask, local_price_mask):
        for layer in self.layers: # run forward in DecoderLayer
            x = layer(x, memory, price_series_mask, local_price_mask)
        return self.norm(x)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self,
                 batch_size,
                 feature_number,
                 d_model_Encoder,
                 d_model_Decoder,
                 encoder,
                 decoder,
                 price_series_pe,
                 local_context_length,
                 device="cuda"):

        super().__init__()
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.batch_size = batch_size
        self.feature_number = feature_number
        self.d_model_Encoder = d_model_Encoder
        self.d_model_Decoder = d_model_Decoder
        self.price_series_pe = price_series_pe
        self.local_context_length = local_context_length
        self.linear_out = nn.Linear(in_features=d_model_Encoder, out_features=1)
        self.bias = torch.nn.Parameter(torch.zeros([1, 1, 1]))
        self.bias2 = torch.nn.Parameter(torch.zeros([1, 1, 1]))

    def forward(self,
                price_series,
                price_series_mask,
                local_price_mask):
        # price_series:[batch, time_len, feature]
        price_series = self.price_series_pe(price_series) # price_series:[batch, time_len, feature]
        encode_out = self.encoder(price_series, price_series_mask)
        #################################padding_price=None###########################################################################
        decode_out = self.decoder(price_series, encode_out, price_series_mask, local_price_mask)# [batch, time_len, feature]
        out = torch.squeeze(decode_out, 0)      # [time_len, feature]
        ###################################  linear  ##################################################
        out = self.linear_out(out)  # [time_len, 1]
        # print(f'out: {out.shape}')
        # print(f'self.bias: {self.bias.shape}')
        bias = self.bias.repeat(out.size()[0], 1, 1)    #[time_len,1]

        out = out+bias

        return out