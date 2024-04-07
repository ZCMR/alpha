import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class DualGCNBertClassifier(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn_model = GCNAbsaModel(bert, opt=opt)
        self.classifier = nn.Linear(opt.bert_dim*2, opt.polarities_dim)

    def forward(self, inputs):
        outputs1,pooled_output = self.gcn_model(inputs)
        final_outputs = torch.cat((outputs1, pooled_output), dim=-1)
        logits = self.classifier(final_outputs)

        penal = None
        
        return logits, penal


class GCNAbsaModel(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn = GCNBert(bert, opt, opt.num_layers)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, adj_dep, src_mask, aspect_mask = inputs
        h1,pooled_output = self.gcn(adj_dep, inputs)
        outputs1 = h1.sum(dim=1)

        return outputs1,pooled_output


class GCNBert(nn.Module):
    def __init__(self, bert, opt, num_layers):
        super(GCNBert, self).__init__()
        self.bert = bert
        self.opt = opt
        self.layers = num_layers
        self.mem_dim = opt.bert_dim // 2
        self.attention_heads = opt.attention_heads
        self.bert_dim = opt.bert_dim
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.pooled_drop = nn.Dropout(opt.bert_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        
        self.layernorm = LayerNorm(opt.bert_dim)

        self.layernorm1 = LayerNorm(opt.bert_dim)
        self.bert_drop1 = nn.Dropout(0.9)

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

        self.attn = MultiHeadAttention(opt.attention_heads, self.bert_dim)


    def forward(self, adj, inputs):
        text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, adj_dep, src_mask, aspect_mask = inputs
        # print('86',text_bert_indices.shape,bert_segments_ids.shape,attention_mask.shape,adj_dep.shape,asp_start.shape,asp_end.shape)


        res= self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids)
        sequence_output = res['last_hidden_state']
        pooled_output = res['pooler_output']

        res_term= self.bert(adj_dep, attention_mask=asp_start, token_type_ids=asp_end)
        sequence_term_output = res_term['last_hidden_state']
        pooled_term_output = res_term['pooler_output']

        # print(sequence_output.shape,pooled_output.shape,sequence_term_output.shape,pooled_term_output.shape)
        sequence_term_output = self.layernorm1(sequence_term_output)
        term_inputs = self.bert_drop(sequence_term_output)

        sequence_output = self.layernorm(sequence_output)
        gcn_inputs = self.bert_drop(sequence_output)
        pooled_output = self.pooled_drop(pooled_output)

        gcn_inputs = self.attn(gcn_inputs, term_inputs)


        return gcn_inputs,pooled_output


def attention(query, key, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.6):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key):
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        attn = attention(query, key,dropout=self.dropout)


        attn = attn.squeeze(dim=1)
        query = query.squeeze(dim=1)

        attn_wn = attn.sum(dim=-1) / attn.shape[-1]
        #print(attn_wn.shape)

        attn = attn_wn.unsqueeze(-1).repeat(1,1,768)     # mask for h
        # attn = torch.cat(torch.split(attn,1,dim=1),dim=3)
        # attn = attn.squeeze(dim=1)
        #print(attn.shape)
        # outputs1 = (h1*mask).sum(dim=1) / attn_wn
        output = attn * query
        return output