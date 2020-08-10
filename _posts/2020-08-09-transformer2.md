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
        self.pos_encoder = PositionalEncoding(ninp, dropout)                   # 1.
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)   # 2.
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers) # 3.
        self.encoder = nn.Embedding(ntoken, ninp)                              # 4-1.
        self.ninp = ninp 
        self.decoder = nn.Linear(ninp, ntoken)                                 # 4-2.
        
        self.init_weights()
```

Class `TransformerModel`에는 여러 **Object Method**들이 존재하지만 한 눈에 보기에는 너무 많으니 우선 객체의 initial state를 정의해주는 init이다. 객체 속성들 중, 다른 함수들을 통해 생성된 것들이 여러개 있는데 위에서부터 차근차근 봐보자.

<br/>

<b># 1. PositionalEncoding(ninp, dropout)</b>

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

예제에서 적용되어 있는 Positional Encoding 클래스를 조금 더 단순하게 구현해봤다.

<img src="/assets/img/pe/transformer/transformer3.jpg">  