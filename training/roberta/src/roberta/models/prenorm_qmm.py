import torch.nn as nn
import torch
import math
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

def percentile_quantize_fn(inp, percentile, num_dist_vals):
    original_type = inp.dtype
    if percentile == 1:
        alpha = inp.abs().max()
    else:
        alpha = torch.topk(inp.abs().reshape(-1), k = int(inp.numel() * (1 - percentile)), dim = -1).values.min()
    if alpha.item() < 1e-20:
        alpha = 1e-20
    scale = (0.5 * num_dist_vals) / alpha
    inp = (torch.round(inp * scale).to(original_type) / scale).detach()
    return inp

class PreMMQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, config):
        return percentile_quantize_fn(A, config["percentile"], config["num_dist_vals"])

    @staticmethod
    def backward(ctx, grad_A):
        return grad_A, None

class PostMMQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, config):
        ctx._config = config
        return A

    @staticmethod
    def backward(ctx, grad_A):
        config = ctx._config
        return percentile_quantize_fn(grad_A, config["percentile"], config["num_dist_vals"]), None

def quantized_matmul(A, B, config):
    A = PreMMQuantize.apply(A, config)
    B = PreMMQuantize.apply(B, config)
    C = torch.matmul(A, B)
    C = PostMMQuantize.apply(C, config)
    return C


class QuantizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, config):
        super().__init__(in_features, out_features)
        self._config = config
        
    def forward(self, X):
        Y = quantized_matmul(X, self.weight.T, self._config)
        if self.bias is not None:
            Y = Y + self.bias
        return Y

class RobertaEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.pad_token_id = config.pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings + 2, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids, position_ids):

        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(self.pad_token_id + 1 + position_ids)

        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

    def resize_position_embeddings(self, target_length, init_weight):
        target_length = target_length + 2
        old_length, old_embedding_dim = self.position_embeddings.weight.shape
        if old_length == target_length:
            return

        new_embeddings = nn.Embedding(target_length, old_embedding_dim)

        multipier = int(target_length / old_length + 2)
        new_embeddings_data = torch.cat(
            [self.position_embeddings.weight.data] +
            [self.position_embeddings.weight.data[2:, :]] * multipier
        , dim = 0)
        new_embeddings.weight.data[:target_length, :] = new_embeddings_data[:target_length, :].detach()
        self.position_embeddings = new_embeddings
        return self.position_embeddings

    def resize_token_type_embeddings(self, target_length, init_weight):
        old_length, old_embedding_dim = self.token_type_embeddings.weight.shape
        if old_length == target_length:
            return
        self.token_type_embeddings = nn.Embedding(target_length, old_embedding_dim)
        init_weight(self.token_type_embeddings)
        return self.token_type_embeddings

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        assert config.hidden_size % config.num_attention_heads == 0

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        assert config.hidden_size == self.all_head_size
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)

        self.query = QuantizedLinear(config.hidden_size, config.hidden_size, config.quantize)
        self.key = QuantizedLinear(config.hidden_size, config.hidden_size, config.quantize)
        self.value = QuantizedLinear(config.hidden_size, config.hidden_size, config.quantize)
        self.dense = QuantizedLinear(config.hidden_size, config.hidden_size, config.quantize)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        hidden_states = self.LayerNorm(hidden_states)

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        context_layer = self.attention(query_layer, key_layer, value_layer, attention_mask)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        attention_output = self.dense(context_layer)
        return attention_output

    def attention(self, query_layer, key_layer, value_layer, attention_mask):
        attention_mask = -1000.0 * (1.0 - attention_mask[:, None, None, :].to(query_layer.dtype))
        attention_scores = quantized_matmul(query_layer, key_layer.transpose(-1, -2), self._config.quantize) / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = self.dropout(F.softmax(attention_scores, dim = -1))
        context_layer = quantized_matmul(attention_probs, value_layer, self._config.quantize)
        return context_layer

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention = Attention(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.FFN = nn.Sequential(
            nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps),
            QuantizedLinear(config.hidden_size, config.intermediate_size, config.quantize),
            nn.GELU(),
            QuantizedLinear(config.intermediate_size, config.hidden_size, config.quantize),
            nn.Dropout(config.hidden_dropout_prob)
        )

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.dropout(attention_output) + hidden_states
        layer_output = self.FFN(attention_output) + attention_output
        return layer_output

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class PrenormRobertaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = Encoder(config)

    def forward(self, input_ids, token_type_ids, position_ids, attention_mask, **kwargs):
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
        sequence_output = self.encoder(embedding_output, attention_mask)
        return sequence_output, {}, {}

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
