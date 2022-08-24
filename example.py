
"""
from utils import TokenEmbedding,PositionalEncoding,Generator
from torch import LongTensor,triu,full
from Transformer import Transformer
import torch.nn as nn

                                    Sequence2Sequence

class seq2seq(nn.Module):
    def __init__(self,vocab_size,embed_dim,n_head,n_encoder_layer,n_decoder_layer,d_ff,src_padding,tgt_padding,embed_padding,max_len:int=100,dropout: float = 0.2,device: str = 'cpu'):
        super(seq2seq,self).__init__()
        self.src_embedding = TokenEmbedding(vocab_size,embed_dim,embed_padding)
        self.trg_embedding = TokenEmbedding(vocab_size,embed_dim,embed_padding)
        self.pe = PositionalEncoding(embed_dim,dropout,max_len)
        self.device = 'device'

        self.transformer = Transformer(embed_dim,n_head,n_encoder_layer,n_decoder_layer,\
            d_ff,dropout,device = device,batch_first=True)
        
        self.generator = Generator(embed_dim,vocab_size,dropout=dropout)
        self.src_padding = src_padding
        self.tgt_padding = tgt_padding
    
    def forward(self,src,trg):
        
        n,src_len = src.shape
        n,trg_len = trg.shape

        src_embed = self.pe(self.src_embedding(src))
        trg_embed = self.pe(self.trg_embedding(trg))

        src_padding_mask = self.transformer.create_src_mask(src,self.src_padding)
        trg_padding_mask = self.transformer.create_src_mask(trg,self.tgt_padding)

        trg_mask = self.transformer.generate_square_subsequent_mask(trg_len)

        out = self.transformer(src_embed,trg_embed,src_key_padding_mask=src_padding_mask,\
                               tgt_key_padding_mask=trg_padding_mask,tgt_mask=trg_mask)
        
        gen = self.generator(out)

        return gen




inp_token = LongTensor([[1,2,3,0,0]])
out_token = LongTensor([[1,2,2,0]])


model = seq2seq(10,4,2,4,4,10,0,0,0)

out = model(inp_token,out_token)

print(out.shape)

"""