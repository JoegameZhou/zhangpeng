import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
import numpy as np
from mindspore import Tensor, ParameterTuple


# SeqTensor = TensorType['batch', 'seq_len', 'token_dim']
# StateTensor = TensorType['batch', 'state_len', 'state_dim']

# constants

DEFAULT_DIM_HEAD = 64
MIN_DIM_HEAD = 32

def cast_tuple(val, num = 1):
    return val if isinstance(val, tuple) else ((val,) * num)


class RotaryEmbedding(nn.Cell):
    def __init__(self, dim):
        super(RotaryEmbedding, self).__init__()
        inv_freq = 1.0 / (10000 ** (ops.arange(0, dim, 2).astype(mindspore.float32) / dim))
        self.inv_freq = mindspore.Parameter(Tensor(inv_freq), name="inv_freq")

    def construct(self, max_seq_len, device, offset=0):
        seq = ops.arange(max_seq_len, dtype=mindspore.float32) + offset
        freqs = ops.einsum('i , j -> i j', seq, self.inv_freq)
        emb = ops.concatenate((freqs, freqs), axis=-1)
        return ops.rearrange(emb, 'n d -> 1 1 n d')


def rotate_half(x):
    x = ops.transpose(x, (0, 1, 3, 2))
    x1, x2 = ops.split(x, 2, -2)
    return ops.concatenate((-x2, x1), -1)

def apply_rotary_pos_emb(t, freqs):
    seq_len, rot_dim = t.shape[-2], freqs.shape[-1]
    t, t_pass = ops.split(t, (rot_dim, -1), -1)
    t_cos = t * mnp.cos(freqs)
    t_sin = rotate_half(t) * mnp.sin(freqs)
    t = t_cos + t_sin
    return ops.concatenate((t, t_pass), -1)


class RecurrentStateGate(nn.Cell):
    """Poor man's LSTM
    """
    def __init__(self, dim: int):
        super(RecurrentStateGate, self).__init__()
        self.main_proj = nn.Dense(dim, dim)
        self.input_proj = nn.Dense(dim, dim)
        self.forget_proj = nn.Dense(dim, dim)
    
    def construct(self, x, state):
        z = ops.tanh(self.main_proj(x))
        i = ops.sigmoid(self.input_proj(x) - 1)
        f = ops.sigmoid(self.forget_proj(x) + 1)
        return state * f + z * i



def repeat(tensor, pattern):
    repeats = [1] * len(tensor.shape)
    for i, p in enumerate(pattern):
        repeats[i] = p
    return ops.tile(tensor, repeats)

def einsum(equation, *operands):
    equation = equation.replace('...', '')
    return ops.einsum(equation, *operands)

class Attention(nn.Cell):
    """Shamelessly copied from github.com/lucidrains/RETRO-pytorch
    """
    def __init__(
        self,
        dim,
        *,
        dim_head=64,
        heads=8,
        causal=False,
        dropout=0.,
        null_kv=False
    ):
        super(Attention, self).__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm((dim,))
        self.dropout = nn.Dropout(p=dropout)

        self.to_q = nn.Dense(dim, inner_dim, weight_init='normal')
        self.to_kv = nn.Dense(dim, inner_dim * 2, weight_init='normal')
        self.to_out = nn.Dense(inner_dim, dim, weight_init='normal')

        # allowing for attending to nothing (null function)
        # and to save attention from breaking if all retrieved chunks are padded out
        if null_kv:
            self.null_kv = Tensor(np.random.randn(2, inner_dim).astype(np.float32))
        else:
            self.null_kv = None

    def forward(self, x, mask=None, context=None, pos_emb=None, pos_func=None):
        b, device, h, scale = x.shape[0], x.device, self.heads, self.scale

        x = self.norm(x)
        kv_input = context if context is not None else x

        q = self.to_q(x)
        k, v = self.to_kv(kv_input).split(self.heads, dim=-1)

        # split heads
        q, k, v = q.reshape((b, -1, h, int(q.shape[-1] / h))), k.reshape((b, -1, h, int(k.shape[-1] / h))), v.reshape((b, -1, h, int(v.shape[-1] / h)))

        # scale
        q = q * scale

        # apply relative positional encoding (rotary embeddings)
        if pos_emb is not None:
            q_pos_emb, k_pos_emb = pos_emb
            q = apply_rotary_pos_emb(q, q_pos_emb)
            k = apply_rotary_pos_emb(k, k_pos_emb)

        # add null key / values
        if self.null_kv is not None:
            nk, nv = self.null_kv
            nk, nv = repeat(nk, (b, h, 1, 1)), repeat(nv, (b, h, 1, 1))
            k = ops.cat((nk, k), axis=-2)
            v = ops.cat((nv, v), axis=-2)

        # derive query key similarities
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # masking
        mask_value = -3.4028234663852886e+38

        if mask is not None:
            if self.null_kv is not None:
                mask = ops.Pad(((1, 0),), 'CONSTANT', True)(mask)

            mask = mask.reshape((b, 1, 1, -1))
            sim = ops.tensor_ops.Tensor.masked_fill(sim, ops.tensor_ops.Tensor.logical_not(mask), mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = ops.tensor_ops.ones((i, j), mindspore.bool_).triu(j - i + 1)
            sim = ops.tensor_ops.Tensor.masked_fill(sim, causal_mask, mask_value)

        if pos_func is not None:
            relative_position_bias = pos_func(sim.shape[-2], sim.shape[-1])
            pbias = broadcast_mask(relative_position_bias, sim)
            sim = sim + pbias

        # attention
        attn = ops.tensor_ops.Tensor.softmax(sim, axis=-1)
        attn = self.dropout(attn)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = out.reshape((b, -1, h * v.shape[-1]))

        # combine heads linear out
        return self.to_out(out)


class FeedForward(nn.Cell):
    def __init__(self, input_dim, hidden_dim, output_dim, activation='ReLU'):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Dense(input_dim, hidden_dim)
        self.activation = getattr(nn, activation)()
        self.linear2 = nn.Dense(hidden_dim, output_dim)

    def construct(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
    
class ContextualTransformerBlock(nn.Cell):
    def __init__(
        self,
        dim: int,
        dim_state: int,
        dim_head=DEFAULT_DIM_HEAD,
        state_len=512,
        heads=8,
        **kwargs
    ):
        super(ContextualTransformerBlock, self).__init__()
        self.scale = dim_head ** -0.5

        attn_kwargs = {}

        self.dim = dim
        self.dim_state = dim_state
        self.pos_mode = 'RoPE'
        print('PE =', self.pos_mode)
        if self.pos_mode == 'learnedemb':
            self.w_1 = nn.Dense(dim, dim, has_bias=False)
            self.w_2 = nn.Dense(dim_state, dim_state, has_bias=False)
        elif self.pos_mode == 'FoPE':
            self.FoPE = RelativeFourierPositions(None, heads, 1024, 128)
        elif self.pos_mode == 'T5PE':
            self.T5PE = T5RelativePositionBiases(None, heads, 32, 128)
        else:
            print('Using the RoPE Positional Encoding!')
        self.heads = heads
        self.causal = True
        self.state_len = state_len
        rotary_emb_dim = max(dim_head // 2, MIN_DIM_HEAD)
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim)

        self.input_self_attn = Attention(dim, heads=heads, causal=True, **attn_kwargs)
        self.state_self_attn = Attention(dim_state, heads=heads, causal=False, **attn_kwargs)

        self.state_input_cross_attn = Attention(dim_state, heads=heads, causal=False, **attn_kwargs)

        self.proj_gate = RecurrentStateGate(dim)

        self.state_proj = nn.Dense(dim + dim_state, dim, has_bias=False)

        self.input_ff = FeedForward(dim,dim,dim)

    def forward(
        self,
        x,
        state,
        mask=None,
        state_mask=None,
        pos_x_emb=None,
        num_seg=None,
    ):
        # ) -> Tuple[SeqTensor, StateTensor]:
    
        batch, seq_len, device = x.shape[0], x.shape[-2], x.device
        if not exists(state):
            state = ops.tensor_ops.zeros((batch, self.state_len, self.dim_state), dtype=mindspore.float32)

        self_attn_pos_emb, state_pos_emb = None, None
        if self.pos_mode == 'RoPE':
            self_attn_pos_emb = self.rotary_pos_emb(seq_len, device=device)
            state_pos_emb = self.rotary_pos_emb(self.state_len, device=device)
            input_attn = self.input_self_attn(x, mask=mask, pos_emb=self_attn_pos_emb)
            state_attn = self.state_self_attn(state, mask=state_mask, pos_emb=state_pos_emb)
        elif self.pos_mode == 'learnedemb':
            x = x + self.w_1(pos_x_emb[num_seg].to(device))
            if num_seg > 0:
                state = state + self.w_2(pos_x_emb[num_seg - 1].to(device))
            else:
                state = state
            input_attn = self.input_self_attn(x, mask=mask, pos_emb=None)
            state_attn = self.state_self_attn(state, mask=state_mask, pos_emb=None)
        elif self.pos_mode == 'FoPE':
            self_attn_pos_emb, state_pos_emb = self_attn_pos_emb, state_pos_emb
            input_attn = self.input_self_attn(x, mask=mask, pos_emb=self_attn_pos_emb, pos_func=self.FoPE)
            state_attn = self.state_self_attn(state, mask=state_mask, pos_emb=state_pos_emb, pos_func=self.FoPE)
        elif self.pos_mode == 'T5PE':
            self_attn_pos_emb, state_pos_emb = self_attn_pos_emb, state_pos_emb
            input_attn = self.input_self_attn(x, mask=mask, pos_emb=self_attn_pos_emb, pos_func=self.T5PE)
            state_attn = self.state_self_attn(state, mask=state_mask, pos_emb=state_pos_emb, pos_func=self.T5PE)
        else:
            self_attn_pos_emb, state_pos_emb = self_attn_pos_emb, state_pos_emb
            input_attn = self.input_self_attn(x, mask=mask, pos_emb=self_attn_pos_emb)
            state_attn = self.state_self_attn(state, mask=state_mask, pos_emb=state_pos_emb)

        state_as_q_cross_attn = self.state_input_cross_attn(state_attn, context=input_attn, mask=state_mask)

        projected_state = self.state_proj(ops.tensor_ops.concat((state_as_q_cross_attn, state_attn), axis=2))

        state_residual = self.proj_gate(projected_state, state)

        output = self.input_ff(state_as_q_cross_attn) + x

        next_state = state_residual
        
        # output = x

        return output, next_state

class NyAttention(nn.Cell):
    def __init__(self, hidden_size, attensize_size):
        super(NyAttention, self).__init__()

        self.attn = nn.Dense(hidden_size, attensize_size, activation='tanh')
        self.ctx = nn.Dense(attensize_size, 1, has_bias=False)
        self.softmax = nn.Softmax(axis=1)
    
    def construct(self, inputs):
        u = self.attn(inputs)  # [b, seq_len, hidden_size] => [b, seq_len, attention_size]
        scores = self.ctx(u)   # [b, seq_len, attention_size] => [b, seq_len, 1]
        attn_weights = self.softmax(scores)  # [b, seq_len, 1]
        inputs = ops.transpose(inputs, (0,2,1))
        out = ops.matmul(inputs, attn_weights)  # [b, seq_len, hidden_size] => [b, hidden_size, seq_len] x [b, seq_len, 1] => [b, hidden_size, 1]
        out = ops.squeeze(out, axis=-1)  # [b, hidden_size, 1] => [b, hidden_size]

        return out, attn_weights

if __name__ == '__main__':
    dim_head = 64
    seq_len = 128
    rotary_emb_dim = max(dim_head // 2, MIN_DIM_HEAD)
    rotary_pos_emb = RotaryEmbedding(rotary_emb_dim)
    pos_emb = rotary_pos_emb(seq_len, device = 'cpu')
    print(pos_emb.size())

    # q_pos_emb, k_pos_emb = cast_tuple(pos_emb, num = 2)
    # q = torch.randn(4, 8, 128, 64)
    # q = apply_rotary_pos_emb(q, q_pos_emb)
    # # k = apply_rotary_pos_emb(k, k_pos_emb)
    # print(q.size())
