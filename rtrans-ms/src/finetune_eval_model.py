# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

'''
Bert finetune and evaluation model script.
'''
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P
from mindspore import context
import numpy as np
from .bert_model import BertModel
from .utils import long_sequence_splitter
from .ctx_transformer_block import ContextualTransformerBlock, NyAttention
# import ipdb

class BlockRecurrentDecoder(nn.Cell):
    def __init__(self, max_len, window_len, num_classes, num_tokens, dim, is_lock, pretrained_model):
        super(BlockRecurrentDecoder, self).__init__()
        self.bert = pretrained_model
        self.attn = ContextualTransformerBlock(dim=dim, dim_state=dim, dim_head=64, state_len=window_len, heads=8)
        self.pos_k = []
        for k in range(max_len // window_len):
            self.pos_k.append(mindspore.Tensor(np.random.randn(window_len, dim), dtype=mstype.float32))
        
        self.mlp = nn.Dense(768, dim)
        if is_lock:
            for name, param in self.bert.cells_and_names():
                if name.startswith('pooler'):
                    continue
                else:
                    param.requires_grad = False

    def construct(self, x, state=None, state_mask=None, num_seg=0):
        input_ids, token_type_ids, attention_mask = x
        sequence_output, pooled_output, _ = self.bert(input_ids, token_type_ids, attention_mask)
        # global_attention_mask = ops.zeros_like(input_ids)
        # global_attention_mask[:, 0] = 1
        # outputs, pooled_output, _ = self.bert(input_ids, token_type_ids, attention_mask)
        # outputs, _, _ = self.bert(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask,
        #                        token_type_ids=token_type_ids, return_dict=False)
        
        outputs_mlp = self.mlp(sequence_output)
        # print('outputs size: {}'.format(outputs_mlp.shape)) # (8, 256, 512)
        
        attention_mask = attention_mask > 0
        # print('size: {}, {}, {}, {}, {}'.format(attention_mask.shape, 0, state_mask.shape, self.pos_k[0].shape, num_seg)) # (8, 256, 512)
        
        # outputs, state = self.attn(x=outputs_mlp, state=state, mask=attention_mask, state_mask=state_mask, pos_x_emb=self.pos_k, num_seg=num_seg)
        outputs = outputs_mlp
        
        return outputs_mlp, outputs, state, attention_mask



class BertCLSModel(nn.Cell):
    """
    This class is responsible for classification task evaluation, i.e. XNLI(num_labels=3),
    LCQMC(num_labels=2), Chnsenti(num_labels=2). The returned output represents the final
    logits as the results of log_softmax is proportional to that of softmax.
    """

    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False,
                 assessment_method=""):
        super(BertCLSModel, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.dtype
        self.num_labels = num_labels
        self.dense_1 = nn.Dense(config.hidden_size, self.num_labels, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        self.dense_att = nn.Dense(512, self.num_labels, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.assessment_method = assessment_method
        # self.config = config # 似乎没有生效
        dim, att_dim = 512, 512
        max_len, window_len = 1024, 256
        self.decoder = BlockRecurrentDecoder(max_len, window_len, self.num_labels, None, dim, False, self.bert)
        self.att = NyAttention(dim, att_dim)

    def construct(self, input_ids, input_mask, token_type_id):
        model_type = 'RTrans' # 'SegmentAttn' 
        window_len = 256
        state = None
        state_rev = None
        if model_type == 'SegmentAttn':
            # print('Recurrent Tansformers...')
            seg_states = []
            bs = input_ids.shape[0]
            text = {'input_ids':input_ids, 'token_type_ids':token_type_id, 'attention_mask':input_mask}
            text_rev = {'input_ids':input_ids, 'token_type_ids':token_type_id, 'attention_mask':input_mask} 
            for seg in long_sequence_splitter(text, window_len, 'bert'):
                ids, token_type_ids, attention_mask = seg
                sequence_output, pooled_output, _ = self.bert(ids, token_type_ids, attention_mask)
                # print(sequence_output.shape) # (bs, seq_len, hidden_dim) = (8, 256, 768)
                cls = self.cast(pooled_output, self.dtype)
                seg_states.append(cls)
            seg_num = len(seg_states)
            # print(f'segment num: {seg_num}') # 1024 // 256 = 4
            # 对多个cls的段的表达进行叠加
            merge_seg_states = seg_states[0]
            for i in range(seg_num-1):
                merge_seg_states += seg_states[i+1]
            logits = self.dense_1(merge_seg_states)
            logits = self.cast(logits, self.dtype)
            if self.assessment_method != "spearman_correlation":
                logits = self.log_softmax(logits)
        elif model_type == 'RTrans':
            seg_states = []
            bs = input_ids.shape[0]
            text = {'input_ids':input_ids, 'token_type_ids':token_type_id, 'attention_mask':input_mask}
            text_rev = {'input_ids':input_ids, 'token_type_ids':token_type_id, 'attention_mask':input_mask} 
            
            hidden = []
            hidden_gate = []
            seg_cls = []
            prev_hidden_states = None
            num_seg = 0
    
            state_mask = ops.zeros((bs, window_len), dtype=mindspore.bool_)
            state_rev_mask = ops.zeros((bs, window_len), dtype=mindspore.bool_)
            # print('state_mask size: ', state_mask.shape)
            
            for seg in long_sequence_splitter(text, window_len, 'bert'):
                ids, token_type_ids, attention_mask = seg
                preds_bert, preds, state, state_mask = self.decoder(seg, state, state_mask, num_seg)
                if len(hidden)>=1:
                    hidden.append(ops.unsqueeze(preds[:,-1,:], -2))
                else:
                    hidden.append(ops.unsqueeze(preds[:,-1,:], -2))
            preds_last_hidden = ops.cat(hidden, axis=-2)
            # print('preds_last_hidden size: {}'.format(preds_last_hidden.shape)) #  (8, 4, 512)
            logits, att_matrix = self.att(preds_last_hidden)
            logits = self.dense_att(logits)
            if self.assessment_method != "spearman_correlation":
                logits = self.log_softmax(logits)
        else:
            _, pooled_output, _ = self.bert(input_ids, token_type_id, input_mask)
            cls = self.cast(pooled_output, self.dtype)
            cls = self.dropout(cls)
            logits = self.dense_1(cls)
            logits = self.cast(logits, self.dtype)
            if self.assessment_method != "spearman_correlation":
                logits = self.log_softmax(logits)
        return logits
    
class RTransModel(nn.Cell):
    def __init__(self, config, is_training, dim, window_len, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False,
                 assessment_method=""):
        super(RTransModel, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.attn = ContextualTransformerBlock(dim=dim, dim_state=dim, dim_head=64, state_len=window_len, heads=8)
        
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.dtype
        self.num_labels = num_labels
        self.dense_1 = nn.Dense(config.hidden_size, self.num_labels, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.assessment_method = assessment_method

    def construct(self, input_ids, input_mask, token_type_id):
        _, pooled_output, _ = self.bert(input_ids, token_type_id, input_mask)
        cls = self.cast(pooled_output, self.dtype)
        cls = self.dropout(cls)
        logits = self.dense_1(cls)
        logits = self.cast(logits, self.dtype)
        if self.assessment_method != "spearman_correlation":
            logits = self.log_softmax(logits)
        return logits

class BertSquadModel(nn.Cell):
    '''
    This class is responsible for SQuAD
    '''

    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False):
        super(BertSquadModel, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.dense1 = nn.Dense(config.hidden_size, num_labels, weight_init=self.weight_init,
                               has_bias=True).to_float(config.compute_type)
        self.num_labels = num_labels
        self.dtype = config.dtype
        self.log_softmax = P.LogSoftmax(axis=1)
        self.is_training = is_training
        self.gpu_target = context.get_context("device_target") == "GPU"
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.shape = (-1, config.hidden_size)
        self.origin_shape = (-1, config.seq_length, self.num_labels)
        self.transpose_shape = (-1, self.num_labels, config.seq_length)

    def construct(self, input_ids, input_mask, token_type_id):
        """Return the final logits as the results of log_softmax."""
        sequence_output, _, _ = self.bert(input_ids, token_type_id, input_mask)
        sequence = self.reshape(sequence_output, self.shape)
        logits = self.dense1(sequence)
        logits = self.cast(logits, self.dtype)
        logits = self.reshape(logits, self.origin_shape)
        if self.gpu_target:
            logits = self.transpose(logits, (0, 2, 1))
            logits = self.log_softmax(self.reshape(logits, (-1, self.transpose_shape[-1])))
            logits = self.transpose(self.reshape(logits, self.transpose_shape), (0, 2, 1))
        else:
            logits = self.log_softmax(logits)
        return logits


class BertNERModel(nn.Cell):
    """
    This class is responsible for sequence labeling task evaluation, i.e. NER(num_labels=11).
    The returned output represents the final logits as the results of log_softmax is proportional to that of softmax.
    """

    def __init__(self, config, is_training, num_labels=11, use_crf=False, with_lstm=False,
                 dropout_prob=0.0, use_one_hot_embeddings=False):
        super(BertNERModel, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.dtype
        self.num_labels = num_labels
        self.dense_1 = nn.Dense(config.hidden_size, self.num_labels, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        if with_lstm:
            self.lstm_hidden_size = config.hidden_size // 2
            self.lstm = nn.LSTM(config.hidden_size, self.lstm_hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.reshape = P.Reshape()
        self.shape = (-1, config.hidden_size)
        self.use_crf = use_crf
        self.with_lstm = with_lstm
        self.origin_shape = (-1, config.seq_length, self.num_labels)

    def construct(self, input_ids, input_mask, token_type_id):
        """Return the final logits as the results of log_softmax."""
        sequence_output, _, _ = self.bert(input_ids, token_type_id, input_mask)
        seq = self.dropout(sequence_output)
        if self.with_lstm:
            batch_size = input_ids.shape[0]
            data_type = self.dtype
            hidden_size = self.lstm_hidden_size
            h0 = P.Zeros()((2, batch_size, hidden_size), data_type)
            c0 = P.Zeros()((2, batch_size, hidden_size), data_type)
            seq, _ = self.lstm(seq, (h0, c0))
        seq = self.reshape(seq, self.shape)
        logits = self.dense_1(seq)
        logits = self.cast(logits, self.dtype)
        if self.use_crf:
            return_value = self.reshape(logits, self.origin_shape)
        else:
            return_value = self.log_softmax(logits)
        return return_value
