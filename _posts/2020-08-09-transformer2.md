---
title: Attention is All You Need (Pytorch)
date: 2020-08-09 00:00:00 +0800
categories: [Code Excercise, Language]
tags: [attention]
seo:
  date_modified: 2020-08-09 20:07:02 +0800
---

<br/>

<img src="/assets/img/pe/transformer/transformer1.jpg">  

[Attention is All You Need (NIPS 2017)](https://chioni.github.io/posts/transformer1/) 포스트에서 `Transformer` 모델의 구조를 리뷰하였다.  

오늘은 모델의 구조를 단순히 컨셉적으로 이해함을 넘어 Pytorch로 어떻게 구현되는지 확인해보자.

워낙 유명한 모델이다 보니 Pytorch 홈페이지의 [Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)에도 잘 정리되어 있으니 이걸 보고 따라해보자.

<br/>

# <b>nn.Transformer</b>

PyToch 1.2 version 부터 Attention is All You Need 논문에 기반한 모듈을 제공해왔다. 논문 내용에서 알 수 있듯이 `nn.Transformer` 모듈은 draw global dependencies between input and output & superior in quality for many sequence-to-sequence problems라는 특징을 가지고 있다. 

<img src="/assets/img/pe/transformer/transformer2.jpg">  

<center><small>Pytorch에서 제공하는 Transformer 관련 Class들</small></center>
<br/>

논문에서는 Transformer 모델을  `machine translation tasks `를 해결하는데 사용하였지만, Tutorial에서는 그보다는 비교적 간단한 `language modeling tasks`에 적용한다. Language Modeling Task는 문장의 다음 단어가 무엇일지 예측하는 과제이다.  

Language Model의 Output을 얻는 과정은 Input과 유사한 차원 크기로 Decoding하는 것이 아니라 단순히 Linear Layer를 태우는 것이다. 따라서 논문에서 사용됬던 `Encoder-Decoder Attention` / `Masked Self Attention`과 같은 여러 기교들을 사용할 필요 없이 간단하게 모델을 구현하는 것이 가능하다. 물론 그렇다고 이것의 구현이 나한테도 간단한 일이 되지는 않았다. 침착하고 정교한 복붙을 통해 한 줄 한 줄 이해해보려 한다.  

<br/>

```python 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer
```

<center>많은 라이브러리들이 필요하지는 않다.</center>
<br/>

```python
class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(
            mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output
```

<br/>

Tutorial에서 Transformer의 구조가 논문과 다른 부분들이 조금 있는데, Task가 다르기 때문에 발생한 차이이다. 우선 Output이 `가장 등장 확률이 높은 단어 하나` 이기 때문에 Decoder의 형태를 띄지 않고 Linear Layer 한 층으로 간소화됬다.  

또한 언어 모델링 과제를 위해서는 Self - Attention 과정에서 이전 포지션의 단어들만 참조하도록 뒤의 단어들에 대한 `attention mask`가 필요하다. 



Class `TransformerModel`에는 여러 **Object Method**들이 존재하지만 한 눈에 보기에는 너무 많으니 객체 속성들과 함수들을 위에서부터 차근차근 봐보자.

- <b>TransforemrModel modules</b>
  - [PositionalEncoding(ninp, dropout)](#PositionalEncoding)
  - [TransformerEncoderLayer(ninp, nhead, nhid, dropout)](#TransformerEncoderLayer)
  - [_generate_square_subsequent_mask(self, sz)](#generate_square_subsequent_mask)
  - [init_weights](#init_weights)

<br/>

## <b>PositionalEncoding</b>

```python
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
```

<br/>

<img src="/assets/img/pe/transformer/transformer3.jpg">  

<center><small>예제에 적용되어 있는 Positional Encoding 클래스의 div_term을 조금 더 단순하게 (논문과 동일하게) 구현하여 사용했다.</small></center>
<br/>

>  Positional Encoding은 embeded된 input에 **고정된** 값을  더해주는 모듈이다.  
>
>  - max_len은 들어올 수 있는 input sequnce의 최대 길이이다.
>  - d_model은 input의 feature dimension이다.

<br/>

[transformer_architecture_positional_encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)라는 kzaemnejad 씨의 블로그 포스트에 다양한 기법들이 소개되어 있는데, Transforemer에 사용된 sin / cos 기반의 positional encoding의 장점은 (1) input dimension과 동일한 크기의 벡터로 생성 가능하고 (2) 모델이나 Input의 형태와 무관하게 고정된 값을 갖는데 있다. 또한 **div_term**을 통해서 input dimension이 매우 길었을 때, positional encoding이 모델에 너무 크게 관여하는 것을 방지한다.  

<br/>

## <b>TransformerEncoderLayer</b>

```python
class TransformerEncoderLayer(Module):
    def __init__(self, 
                 d_model, 
                 nhead, 
                 dim_feedforward=2048, 
                 dropout=0.1, 
                 activation="relu"):
        
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

	def forward(self, 
                src: Tensor, 
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2) # 1.
        src = self.norm1(src) # 2.
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src)))) # 3.
        src = src + self.dropout2(src2) # 4-1.
        src = self.norm2(src) # 4-2.
        return src
```

<br/>

**MultiheadAttention**

- positional encoding이 완료된 embeded input을 받아서 self-attention을 수행한다.

<br/>

**Feedforward Model**

1. attn layer 이후에 residual connection + droput

2. LayerNorm() 함수를 통해 d_model 차원의 data에 대해 Layer Normalization over a mini-batch.

   <img src="/assets/img/pe/transformer/transformer4.jpg"> 

3.  `(200 -> 2048 -> 200)` 순서로 Sparse AutoEncoder의 형태로 forward를 수행한다.

4. residual connection + dropout -> Layer Normalization

<br/>

## <b>generate_square_subsequent_mask</b>

```python
def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')
                                   ).masked_fill(mask == 1, float(0.0))
    return mask
  
src = ['a','b','c','d','e']

_generate_square_subsequent_mask(len(src))
```

<center><small>정사각형의 attention mask 생성</small></center>

<img src="/assets/img/pe/transformer/transformer5.jpg">

<br/>

## <b>init_weights</b>

