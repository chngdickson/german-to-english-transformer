import torch
import torch.nn as nn
from torch.nn.modules.transformer import Transformer

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention,self).__init__()
        # Embedding size
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert embed_size % heads == 0 , "embeding size needs to be div by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)
        
    def forward(self, queries ,keys, values, mask):
        # Original Dim of embeddings are 
        """[N_batch, len, embed_size]"""
        # New dim for multiple heads are [N_batch, len, heads, head_dim]
        N_batch = queries.shape[0]
        q_len, k_len, v_len = queries.shape[1], keys.shape[1], values.shape[1]
        
        # Split Embedding into self.heads pieces
        queries = queries.reshape(N_batch, q_len, self.heads, self.head_dim)
        keys = keys.reshape(N_batch, k_len, self.heads, self.head_dim)
        values = values.reshape(N_batch, v_len, self.heads, self.head_dim)
        
        # Einsum can be thought of as a dot.product aka Multiplication
        # Desired output shape must match values
        #  [N,h, q_len, k_len]
        correlation = torch.einsum("nqhd,nkhd->nhqk", [queries,keys])
        
        if mask is not None:
            # Mask fill triangulation matrix
            correlation = correlation.masked_fill(mask == 0, float("-1e20"))
            
        attention = torch.softmax(correlation/(self.embed_size**(0.5)),dim=3)
        # Softmax dim=3 is k_len key dimension is the source sentence
        
        out = torch.einsum("nhqk,nvhd->nqhd",[attention,values])
        # attention shape : (N, heads, query_len, key_len)
        # Values shape : (N, value_len, heads, heads_dim)
        # out (N, query_len, heads, head_dim)
        
        out = out.reshape(N_batch,q_len,self.heads*self.head_dim)
        # Reshape to fc_out input (headss*self.head_dim)
        
        out = self.fc_out(out)
        # Reshape to original Embedding shape
        """[N_batch, len, embed_size]"""
        return out

class TransformerBlock(nn.Module):
    def __init__(self,embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feedforward = nn.Sequential(
            nn.Linear(embed_size,int(forward_expansion*embed_size)),
            nn.ReLU(),
            nn.Linear(int(forward_expansion*embed_size), embed_size)
            )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask):
        residual = query
        
        x = self.attention.forward(query,key,value,mask)
        x = self.dropout(self.norm1(residual+x))
        
        residual = x
        x = self.feedforward(x)
        x = self.dropout(self.norm2(residual+x))
        return x

class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size,embed_size)
        self.position_embedding = nn.Embedding(max_length,embed_size)
        
        self.trans_layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout,
                    forward_expansion
                )
            ]
        )
        # Confirmed theres only 1 transformer block
        #print(self.trans_layers)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x,mask):
        # WHAT is x?
        N_batch, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N_batch, seq_length).to(self.device)
        #print('position shape',positions.shape)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        #print('x', x)
        #print('position_input', positions)
        #print('position output', self.position_embedding(positions))
        for trans_layer in self.trans_layers:
            out = trans_layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size,heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size,heads,dropout,forward_expansion
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x,x,x,trg_mask)
        query = self.dropout(self.norm(attention+x))
        out = self.transformer_block.forward(query,key,value,src_mask)
        return out

class Decoder(nn.Module):
    def __init__(self, 
                 trg_vocab_size, 
                 embed_size, 
                 num_layers,
                 heads, 
                 forward_expansion, 
                 dropout, 
                 device, 
                 max_length):
        super(Decoder,self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size,heads,forward_expansion,dropout,device)
             for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x,enc_out, src_mask, trg_mask):
        N_batch , seq_length = x.shape
        positions = torch.arange(0,seq_length).expand(N_batch,seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        
        for decoderlayer in self.layers:
            # [Value key] = enc out, [query] = decoder_x
            decoderlayer.forward(x,enc_out,enc_out,src_mask,trg_mask)
        out = self.fc_out(x)
        del positions
        return out
    
class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size, 
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 embed_size=256,
                 num_layers = 6,
                 forward_expansion=4,
                 heads=8,
                 dropout=0,
                 device = "cuda",
                 max_length=100
                 ):
        super(Transformer,self).__init__()
        """
        # params for transformer
        src_vocab_size - encoder vocab size
        trg_vocab_size - decoder vocab size
        src_pad_idx - encoder pad index
        trg_pad_idx - decoder pad index
        embed_size - Our self-determined embed size
        num_layers - number of layers of encoder and decoder block
        forward_expansion - Feed-forward expansion
        heads -  number of attention heads
        dropout - dropout lol
        device = gpu or cpu
        max_length - maximum length of our sentence
        """
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )
        # Delete the ones that are a more than max_length, and keep the ones
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    
    def make_src_mask(self,src):
        # Masking in on itself
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N,1,1,src_len)
        return src_mask.to(self.device)
    
    def make_trg_mask(self,trg):
        # trg = [batch size, trg len]
        """
        1. Masking "<PAD>" tokens and
        2. Upper-triangular matrix"""
        
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        # trg_pad_mask =  4D [batch size, 1, 1, trg len]
        
        
        trg_len = trg.shape[-1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len))).to(self.device).bool()
        
        # trg_sub_mask = [trg len, trg len]
        
        
        trg_mask = trg_pad_mask & trg_sub_mask
        
        # trg_mask = [batch size , 1 , trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        # src = [batch_size, src_sentence_len]
        # trg = [batch_size, trg_sentence_len]
        
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # src_mask = [batch size, 1, 1      , src len]
        # trg_mask = [batch size, 1, trg len, trg len]
        
        
        enc_src = self.encoder.forward(src,src_mask)
        out = self.decoder.forward(trg,enc_src,src_mask,trg_mask)

        # enc_src = [batch size, src len, hid dim]
        # output =  [batch size, trg len, out dim] 
        
        # out dim = probability in each trg len
        return out
    
# def patch_src(src, pad_idx):
#     #src = src.transpose(0, 1)
#     return src


# def patch_trg(trg, pad_idx):
#     #trg = trg.transpose(0, 1)
#     trg = trg[:, :-1]
#     flat = trg[:, 1:].contiguous().view(-1)
#     return trg, flat

# from torch import optim,nn
# import torch.nn.functional as F
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(device)
#     eos = 3
#     sos = 1
#     src = torch.tensor(
#         [
#             [1,5,6,4,3,9,5,2,0],
#             [1,8,7,3,4,5,6,7,2]
#         ],dtype=torch.int64
#         ).to(device)

#     trg = torch.tensor(
#         [
#             [sos,7,4,3,5,9,2,eos],
#             [sos,5,6,2,4,7,6,eos]
#         ],dtype=torch.int64
#         ).to(device)

    
#     src_pad_idx ,trg_pad_idx = 0, 0
#     src_vocab_size , trg_vocab_size = 10, 10

#     model= Transformer(src_vocab_size,trg_vocab_size,src_pad_idx,trg_pad_idx).to(device)
#     criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
#     pred_trg = trg[:, :-1]
#     compare_trg = trg[:, 1:]
    
#     predicted = model(src, pred_trg)
    
#     predicted_vocab_size = predicted.shape[-1]
#     print(predicted_vocab_size)
#     print(predicted.shape)
    
#     """
#     we wanna use view aka reshape, because it uses [move] or [same_memory]
#     contiguous = same memory
#     # View always need contiguous
#     """
    
#     predicted = predicted.contiguous().view(-1, predicted_vocab_size)
#     compare_trg2 = compare_trg.contiguous().view(-1) # 1D
    
#     loss = criterion(predicted, compare_trg2)
#     print(loss)
#     print(loss.item())
#     print(compare_trg.data_ptr() == compare_trg2.data_ptr())
#     # [batch size, trg len, out dim] 
    

# #from a_self_attention import Transformer
# from attention_transformer import Transformer

# SRC_VOCAB_SIZE ,TRG_VOCAB_SIZE = len(SRC.vocab) , len(TRG.vocab)
# SRC_PAD_IDX, TRG_PAD_IDX = SRC.vocab['<PAD>'] , TRG.vocab['<PAD>']
# MAX_SENTENCE_LENGTH = 100
# EMBED_SIZE , NUM_LAYERS , FORWARD_EXPANSION , HEADS = 256, 3, 2 , 8
# DROPOUT = 0.1

# model = Transformer(
#     src_vocab_len=SRC_VOCAB_SIZE,
#     trg_vocab_len=TRG_VOCAB_SIZE,
#     src_pad_idx = SRC_PAD_IDX,
#     trg_pad_idx = TRG_PAD_IDX,
#     src_max_sentence_len = MAX_SENTENCE_LENGTH,
#     trg_max_sentence_len = MAX_SENTENCE_LENGTH,
#     hid_dim = EMBED_SIZE,
#     n_layers = NUM_LAYERS,
#     n_heads = HEADS,
#     ff_dim_multiplier = FORWARD_EXPANSION,
#     dropout = DROPOUT,
#     device = DEVICE
# ).to(DEVICE)


# from attention_transformer import Transformer
# import torch.nn as nn


# SRC_VOCAB_SIZE ,TRG_VOCAB_SIZE = len(vocab_en) , len(vocab_de)
# SRC_PAD_IDX, TRG_PAD_IDX = vocab_en['<PAD>'] , vocab_de['<PAD>']
# MAX_SENTENCE_LENGTH = 110
# EMBED_SIZE , NUM_LAYERS , FORWARD_EXPANSION , HEADS = 256, 3, 2 , 8
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DROPOUT = 0.1

# model = Transformer(
#     src_vocab_len=SRC_VOCAB_SIZE,
#     trg_vocab_len=TRG_VOCAB_SIZE,
#     src_pad_idx = SRC_PAD_IDX,
#     trg_pad_idx = TRG_PAD_IDX,
#     src_max_sentence_len = MAX_SENTENCE_LENGTH,
#     trg_max_sentence_len = MAX_SENTENCE_LENGTH,
#     hid_dim = EMBED_SIZE,
#     n_layers = NUM_LAYERS,
#     n_heads = HEADS,
#     ff_dim_multiplier = FORWARD_EXPANSION,
#     dropout = DROPOUT,
#     device = DEVICE
# ).to(DEVICE)
