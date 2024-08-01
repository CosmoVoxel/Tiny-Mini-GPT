from torch import nn 
import torch.utils
import torch.utils.checkpoint
from config import DecoderConfig

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed

class MLP(torch.nn.Module):
    def __init__(self,config:DecoderConfig):
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(config.emb_d,config.ff_d,config.bias),
            torch.nn.GELU(),
            torch.nn.Linear(config.ff_d,config.emb_d,config.bias),
            torch.nn.Dropout(config.ff_drop),
        )
    
    def forward(self,x):
        x = self.out(x)
        return x

class MultiHeadAttention(torch.nn.Module):
    def __init__(self,config:DecoderConfig):
        super().__init__()

        self.qkv = torch.nn.Linear(config.emb_d,config.emb_d*3,config.bias)
        self.heads = config.heads
        if config.emb_d % config.heads != 0:
            assert ValueError("Error: Number of embeding must be dividible by heads!")
        self.head_d = config.emb_d//config.heads
        self.out = torch.nn.Linear(config.emb_d, config.emb_d,config.bias)
    
    def forward(self,x:torch.Tensor):
        BATCH, SEQ_LEN , EMB = x.size()
        q,k,v = self.qkv(x).reshape(BATCH,SEQ_LEN,self.heads,self.head_d*3).permute(0,2,1,3).chunk(3,dim=-1)
        x = torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=DecoderConfig.att_drop)
        x = x.transpose(1,2).contiguous().view(BATCH,SEQ_LEN,EMB)
        x = self.out(x)
        return x
    
class DecoderLayer(torch.nn.Module):
    def __init__(self,config:DecoderConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.emb_d)
        self.norm2 = RMSNorm(config.emb_d)
        self.attn = MultiHeadAttention(config)
        self.mlp = MLP(config)
    
    def forward(self,x:torch.Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

def generate_sinusoidal_positional_encoding(num_positions, embedding_dim):
    pe = torch.zeros(num_positions, embedding_dim).cuda()
    position = torch.arange(0, num_positions).unsqueeze(1).float().cuda()
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / embedding_dim)).cuda()
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class Decoder(torch.nn.Module):
    def __init__(self,config:DecoderConfig):
        super().__init__()
        self.we = torch.nn.Embedding(config.vocab_size,config.emb_d)
        self.layers = torch.nn.ModuleList([DecoderLayer(config) for _ in range(config.layers_d)])
        self.norm1 = RMSNorm(config.emb_d)
        self.out = torch.nn.Linear(config.emb_d,config.vocab_size,bias=False)

    def forward(self,x:torch.Tensor):
        BATCH, SEQ_LEN = x.size()
        x = self.we(x)
        x = x + generate_sinusoidal_positional_encoding(SEQ_LEN,self.we.embedding_dim).unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        x = self.norm1(x)
        x = self.out(x)
        return x

def generate_tokens(model:Decoder,n_tokens = 128,config = DecoderConfig(),start_text = "War"):
    tokens =config.tokenizer.encode(start_text)
    n_tokens = n_tokens if n_tokens+len(tokens) < config.max_seq_len else config.max_seq_len-len(tokens)
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        for i in range(n_tokens):
            tokenized = torch.tensor(tokens).cuda().unsqueeze(0)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                next_token  = model(tokenized)
            next_token = torch.argmax(next_token[:, -1, :], dim=-1)
            tokens.append(next_token.item())
        tokens = config.tokenizer.decode(tokens)
    model.train()
    return tokens