import copy
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

#----------------
#工具函数
#----------------
def clones(module, N):
   "深拷贝N份相同模块-全新、独立、参数不共享，把复制好的模块装进Pytorch容器"
   return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        #所有位置共享一套参数，但是每个特征有独立的缩放/偏置参数
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self,x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class SublayerConnection(nn.Module):
    "residual connection + dropout + layernorm"
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        "sublayer is a function: sublayer(x)"
        return x + self.dropout(sublayer(self.norm(x)))#pre-norm/activate/dropout
    #pre-norm解决深层训练的梯度稳定性问题

#--------------
#attention
#--------------
def attention(query, key, value, mask=None, dropout=None):
    """计算多头注意力权重并应用于value。"""
    d_k = query.size(-1)#.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    #(barch_size, h, seq_len, seq_len)
    if mask is not None:
        #mask:(batch, 1, seq_len, seq_len)或可广播
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4) #q,k,v,out
        self.attn = None
        self.dropout = nn.Dropout(dropout)
    def forward(self, query, key, value, mask=None):
        #如果有mask，让mask能广播到多头上
        if mask is not None:
            #mask (batch, seq, seq) --> (batch, 1, seq, seq)
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        #线性变换并分头
        query, key, value = [
            lin(x) #(batch, seq, d_model)
            .view(nbatches, -1, self.h, self.d_k) #(batch, seq, h, d_k)
            .transpose(1, 2) #(batch, h, seq, d_k)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        #合并多头 x:(batch, h, seq, d_k)
        # transpose -> 不连续，不能view，transpose -> contiguous -> 连续 -> 可以view
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x) #(batch, seq, d_model)

#---------
# position-wise FFN
#---------
class PostionwiseFeedward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x)))) #(batch, seq, d_model)

#----------
# positional encoding
#----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        #(max_len,1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        #(d_model//2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)#(1, max_len, d_model)
        #将张量pe注册为模型的“缓冲区”，会随模型一起保存/加载；会自动跟模型到同一设备；不是可训练参数
        self.register_buffer("pe", pe)
    def forward(self, x):
        #x:(batch, seq, d_model)
        x = x + self.pe[:, :x.size(1)]#后面相加项的形状(1, seq, d_model)
        return self.dropout(x)

#--------------
# Encoder/Decoder层
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        #mask遮蔽padding位置，避免学习全局上下文语义的时候被无效padding干扰
        return self.sublayer[1](x, self.feed_forward)
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    #全局的LayNorm以保证每一批次训练中encoder输出在进入后续模块(decoder)前具有统一的尺度与均值
    #减少下游模块对不同训练阶段中输出尺度波动的敏感性。
    

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory #交叉注意力中需要输入的键/值 
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        #tgt_mask遮蔽tgt序列中的padding位置
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        #src_mask遮蔽src序列中的padding位置，避免decoder把注意力分配到无意义的padding上
        return self.sublayer[2](x, self.feed_forward)
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

#------------------
#完整的Transformer & 生成器 & Embedding
#------------------
class Generator(nn.Module):
    """从Decoder输出到词表概率"""
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class TransformerModel(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, N, d_model, d_ff, h, dropout, max_len=5000):
        super().__init__()
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PostionwiseFeedward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout, max_len=max_len)
        src_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
        tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))
        encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
        generator = Generator(d_model, tgt_vocab)
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        #xavier初始化
        #pytorch会递归收集所有子层参数放到self.parameters()中
        for p in self.parameters():
            #只初始化权重矩阵，不初始化偏置
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)
    #注意这里是encode不是encoder
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)



    





    




