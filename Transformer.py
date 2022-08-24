
import torch.nn as nn
from utils import PositionWiseFeedforward,ResidualLayerNorm,_get_clones,_get_activation_fn
from torch import Tensor
from typing import Union,Callable,Optional,Any
import torch.nn.functional as f
from torch import triu,full




#                                      Transformer block
class Transformer(nn.Module):
    __constants__ = ['norm']
    def __init__(self, d_model: int = 512, num_head: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, d_ff: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = f.relu,
                 custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Transformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model,num_head,d_ff,dropout,activation,layer_norm_eps,batch_first,norm_first,**factory_kwargs)
            encoder_norm = nn.LayerNorm(d_model,layer_norm_eps,**factory_kwargs)

            self.encoder = TransformerEncoder(encoder_layer,num_encoder_layers,norm=encoder_norm)
        
        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model,num_head,d_ff,dropout,activation,layer_norm_eps,batch_first,norm_first,**factory_kwargs)
            decoder_norm = nn.LayerNorm(d_model,layer_norm_eps,**factory_kwargs)

            self.decoder = TransformerDecoder(decoder_layer,num_decoder_layers,norm=decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.num_head = num_head

        self.batch_first = batch_first
        self.enc_dec_attn_weight = None
        self.mask_attn_weight = None
        self.input_attn_weight = None


    def forward(self,src: Tensor,tgt: Tensor, src_mask: Optional[Tensor] = None,tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        is_batched = src.dim() == 3

        if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
                raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) !=tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
                
        if src.size(-1)!= self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")
                
        memory,avr_attn_weight = self.encoder(src,mask=src_mask,src_key_padding_mask=src_key_padding_mask)

        output,_,enc_dec_attn_weight = self.decoder(tgt,memory,tgt_mask=tgt_mask,memory_mask=memory_mask,
                                                            tgt_key_padding_mask=tgt_key_padding_mask,
                                                            memory_key_padding_mask=memory_key_padding_mask)
        self.enc_dec_attn_weight = enc_dec_attn_weight
        self.input_attn_weight = avr_attn_weight
        self.mask_attn_weight = _

        return output

    def _get_enc_dec_attn_weight(self):
        if self.enc_dec_attn_weight is None:
            raise Exception(f"empty attention weights")

        return self.enc_dec_attn_weight
    
    def _get_input_attn_weight(self):
        if self.input_attn_weight is None:
            raise Exception(f"empty attention weights")

        return self.input_attn_weight
    
    def _get_masked_attn_weight(self):
        if self.mask_attn_weight is None:
            raise Exception(f"empty attention weights")

        return self.mask_attn_weight

    @staticmethod
    def create_src_mask(src,pad_idx):
        #reshape src to shape(batch_size,src_len)
        src_mask = (src == pad_idx)
        return src_mask

    def generate_square_subsequent_mask(self,sz: int):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return triu(full((sz, sz), float('-inf')), diagonal=1)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    


#                   Encoder Block

class TransformerEncoder(nn.Module):
    __contants__ = ['norm']

    def __init__(self,encoder_layer,num_layers, norm=None):
        super(TransformerEncoder,self).__init__()

        self.layers = _get_clones(encoder_layer,num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.attn_weights = None
    
    def forward(self,src:Tensor, mask: Optional[Tensor]=None,src_key_padding_mask:Optional[Tensor]=None)->Tensor:
        inp = src

        for mod in self.layers:
            output,attn_weight = mod(inp,src_mask=mask,src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output =self.norm(output)

        self.attn_weights = attn_weight

        return output,attn_weight

    def _get_attention_weights(self):
        return self.attn_weights


#                       Decoder Block

class TransformerDecoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self,decoder_layer, num_layers, norm=None):
        super(TransformerDecoder,self).__init__()
        self.layers = _get_clones(decoder_layer,num_layers)
        self.num_layers = num_layers
        self.norm = norm

    
    def forward(self,tgt:Tensor,memory:Tensor,tgt_mask:Optional[Tensor]=None,
                memory_mask: Optional[Tensor]=None, tgt_key_padding_mask: Optional[Tensor]=None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        out = tgt

        for mod in self.layers:
            out,m_att_weight,enc_dec_attn_weight = mod(out,memory,tgt_mask=tgt_mask,memory_mask= memory_mask,tgt_key_padding_mask = tgt_key_padding_mask,
                      memory_key_padding_mask = memory_key_padding_mask)

            if self.norm is not None:
                out = self.norm(out)

            return out,m_att_weight,enc_dec_attn_weight




#                          Encoder Layer

class TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first','norm_first']

    def __init__(self,d_model,n_head,d_ff=2048, dropout = 0.1,activation=f.relu,\
                 layer_norm_eps = 1e-5,batch_first=False, norm_first: bool = False,
                  device=None,dtype=None) -> None:
        
        factory_kwargs = {'device':device, 'dtype':dtype}
        super(TransformerEncoderLayer,self).__init__()
        
        self.norm_first=norm_first
        self.self_attn = nn.MultiheadAttention(d_model,n_head,batch_first=batch_first,**factory_kwargs)
        self.norm1 = ResidualLayerNorm(d_model,eps_layer_norm=layer_norm_eps,dropout=dropout,**factory_kwargs)
        self.norm2 = ResidualLayerNorm(d_model,eps_layer_norm=layer_norm_eps,dropout=dropout,**factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.ff = PositionWiseFeedforward(d_model,d_ff,dropout=dropout,**factory_kwargs)

        # string support for activation function

        if isinstance(activation,str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate(self,state):
        if 'activation' not in state:
            state['activation'] = f.relu
        super(TransformerEncoderLayer,self).__setstate__(state)

    def forward(self,src: Tensor, src_mask: Optional[Tensor]=None,src_key_padding_mask:Optional[Tensor]=None)->Tensor:
        r'''pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer(required)
            src_mask: the mask for the src sequence(optional)
            src_key_padding_mask: the mask for the src keys per batch(optional)

        Shape:
            shape(src)==>(B,seq_len,n_dim) if batch_first else (seq_len,B,d_dim)

        '''
        
        '''
           ## I CAN CHANGE THIS CODE TO THE CURRENT CODE FOR EFFICIENCY CHECK...(SMILEE...)

        if self.norm_first:
            sa,attn_weight=self._sa_block(self.norm1(x), src_mask,src_key_padding_mask)
            x = x+sa
            x = x+self.ff(self.norm2(x))

        else:
            
            sa,attn_weight=self._sa_block(x,src_mask,src_key_padding_mask)
            x = self.norm1(x + sa)
            x = self.norm2(x + self.ff(x))

        return x,attn_weight


        '''


        x = src
        
        mha,encoder_attn = self._sa_block(x,src_mask,src_key_padding_mask)
        norm1 = self.norm1(mha,x)
        ff = self.ff(norm1)
        norm2=self.norm2(ff,norm1)

        return norm2,encoder_attn
        


    def _sa_block(self,x:Tensor,attn_mask: Optional[Tensor],key_padding_mask:Optional[Tensor])->Tensor:
        out, attn_weight = self.self_attn(x,x,x,
                                          attn_mask=attn_mask,
                                          key_padding_mask=key_padding_mask)

        return out, attn_weight




#                                   Decoder

class TransformerDecoderLayer(nn.Module):

    __constants__=['batch_first']

    def __init__(self, d_model: int, num_head: int, d_ff: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = f.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        
        factory_kwargs = {'device':device,'dtype':dtype}
        super(TransformerDecoderLayer,self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model,num_head,batch_first=batch_first,**factory_kwargs)
        self.multihead_attn = nn.MultiheadAttention(d_model,num_heads=num_head,batch_first=batch_first,**factory_kwargs)

        self.ff=PositionWiseFeedforward(d_model,d_ff,dropout=dropout,**factory_kwargs)

        self.norm1 = ResidualLayerNorm(d_model,eps_layer_norm=layer_norm_eps,dropout=dropout,**factory_kwargs)
        self.norm2 = ResidualLayerNorm(d_model,eps_layer_norm=layer_norm_eps,dropout=dropout,**factory_kwargs)
        self.norm3 = ResidualLayerNorm(d_model,eps_layer_norm=layer_norm_eps,dropout=dropout,**factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
        if isinstance(activation,str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    
    def __setstate__(self,state):
        if 'activation' not in state:
            state['activation'] = f.relu
        super(TransformerDecoderLayer,self).__setstate__(state)

    def forward(self,tgt: Tensor, memory: Tensor, tgt_mask:Optional[Tensor] = None,memory_mask: Optional[Tensor]=None,
                tgt_key_padding_mask:Optional[Tensor] = None, memory_key_padding_mask:Optional[Tensor]=None)->Tensor:
        
        x=tgt
        masked_mha, masked_attn_weights = self._sa_block(x,tgt_mask,tgt_key_padding_mask)
        norm1 = self.norm1(masked_mha,x)

        enc_dec_mha,enc_dec_attn_weights = self._mha_block(norm1,memory,memory_mask,memory_key_padding_mask)
        norm2 = self.norm2(enc_dec_mha,norm1)
        
        ff = self.ff(norm2)
        norm3 = self.norm3(ff,norm2)

        return norm3,masked_attn_weights,enc_dec_attn_weights


    def _sa_block(self,x:Tensor,attn_mask:Optional[Tensor],key_padding_mask:Optional[Tensor])->Tensor:
        out,attn_weight = self.self_attn(x,x,x,attn_mask=attn_mask,
                                         key_padding_mask=key_padding_mask)

        return self.dropout1(out),attn_weight
    

    def _mha_block(self,x:Tensor,mem:Tensor,attn_mask: Optional[Tensor],key_padding_mask:Optional[Tensor])->Tensor:
        out,attn_weight = self.multihead_attn(x,mem,mem,attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask)
        
        return self.dropout2(out), attn_weight




def _get_activation_fn(activation):
    if activation == "relu":
        return f.relu
    elif activation == "gelu":
        return f.gelu

    raise RuntimeError("activation should be relu/gelu, but not {}".format(activation))
        