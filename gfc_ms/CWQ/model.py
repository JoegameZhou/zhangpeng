import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import SparseTensor, COOTensor
from mindspore import Tensor, Parameter, CSRTensor
from mindspore.common.initializer import Normal
import mindspore_hub as mshub
import mindspore
import mindspore as ms
import mindspore.ops as ops
sparse_dense_matmul = mindspore.ops.SparseTensorDenseMatmul()

class GFC(nn.Cell):
    def __init__(self, args, ent2id, rel2id):
        super().__init__()
        num_relations = len(rel2id)
        self.num_ents = len(ent2id)
        self.num_steps = 2
        self.num_ways = 2
        try:
            if args.bert_name == "bert-base-uncased":
                model = "mindspore/1.9/bertbase_cnnews128"
                self.bert_encoder = mshub.load(model)
            else:
                raise ValueError("please input the right name of pretrained model")
        except ValueError as e:
            raise e
        dim_hidden = 768

        self.step_encoders = []
        for i in range(self.num_steps):
            m = nn.SequentialCell([
                nn.Dense(in_channels=dim_hidden, out_channels=dim_hidden)
            ])
            self.step_encoders.append(m)

        self.rel_classifier = nn.Dense(in_channels=dim_hidden, out_channels=num_relations)

        self.hop_att_layer = nn.SequentialCell([
            nn.Dense(in_channels=dim_hidden, out_channels=1)
        ])
        self.high_way = nn.SequentialCell([
            nn.Dense(in_channels=dim_hidden, out_channels=dim_hidden),
            nn.Sigmoid()
        ])


    def construct(self, heads, questions, answers=None, triples=None, entity_range=None, training=True):
        for item in questions:
            questions[item] = mindspore.Tensor(questions[item])
        if answers is not None:
            answers = mindspore.Tensor(answers)
        if entity_range is not None:
            entity_range = mindspore.Tensor(entity_range)
        if "attention_mask" in questions:
            questions['input_mask'] = questions.pop("attention_mask")
        q = self.bert_encoder(**questions)
        q_word_h, q_embeddings, _ = q[0], q[1], q[2] # (bsz, dim_h), (bsz, len, dim_h)
        heads = mindspore.Tensor(heads)
        heads = heads.astype(mindspore.float32)
        last_e = heads.unsqueeze(0)
        bsz = len(heads)
        word_attns = []
        rel_probs = []
        ent_probs = []
        ctx_h_list = []
        q_word_h_dist_ctx = [0]
        q_word_h_hop = q_word_h
        for t in range(self.num_steps):
            h_key = self.step_encoders[t](q_word_h_hop) # [1 64 768]
            q_logits = mindspore.ops.matmul(h_key, q_word_h.transpose((0,2,1))) # [1 64, 768] [1 768 64]  # [bsz, max_q, dim_h] * [bsz, dim_h, max_q] = [bsz, max_q, max_q]
            q_logits = q_logits.transpose((0,2,1))  # [1 64 64]

            q_dist = mindspore.nn.Softmax(axis=2)(q_logits)  # [bsz, max_q, max_q]
            q_dist = q_dist * questions['input_mask'].astype(mindspore.float32).unsqueeze(1)  # [bsz, max_q, max_q]*[bsz, max_q]
            q_dist = q_dist / (mindspore.ops.ReduceSum(keep_dims=True)(q_dist, axis=2) + 1e-6)  # [bsz, max_q, max_q]
            hop_ctx = mindspore.ops.matmul(q_dist, q_word_h_hop)  # [1 64 64] [1 64 768]
            if t == 0:
                z = 0
            else:
                z = self.high_way(q_word_h_dist_ctx[-1]) 
            if t == 0:
                q_word_h_hop = q_word_h + hop_ctx
            else:
                q_word_h_hop = q_word_h + hop_ctx + z*q_word_h_dist_ctx[-1]# [bsz, max_q, max_q]*[bsz, max_q, dim_h] = [bsz, max_q, dim_h]
            q_word_h_dist_ctx.append(hop_ctx + z*q_word_h_dist_ctx[-1])

            q_word_att = mindspore.ops.ReduceSum(keep_dims=True)(q_dist, axis=1)  # [bsz, max_q,1]
            q_word_att = mindspore.nn.Softmax(axis=2)(q_word_att)
            q_word_att = q_word_att * questions['input_mask'].float().unsqueeze(1)  # [bsz, 1, max_q]*[bsz, max_q]
            q_word_att = q_word_att / (mindspore.ops.ReduceSum(keep_dims=True)(q_word_att, axis=2) + 1e-6)  # [bsz, max_q, max_q]
            word_attns.append(q_word_att)  # bsz,1,q_max [1 1 64]
            ctx_h = mindspore.ops.matmul(q_word_h_hop.transpose((0,2,1)), q_word_att.transpose((0,2,1))).squeeze(2) # [1 768] # [bsz, dim_h, max_q] * [bsz, max_q,1]

            ctx_h_list.append(ctx_h) 
            rel_logit = self.rel_classifier(ctx_h)  # [bsz, num_relations]
            rel_dist = mindspore.ops.Sigmoid()(rel_logit)
            rel_probs.append(rel_dist)

            new_e = []
            for b in range(1):
                sub, rel, obj = triples[:, 0], triples[:, 1], triples[:, 2]
                sub_p = last_e[b:b + 1, sub]  # [1 2316] [1, #tri]
                rel_p = rel_dist[b:b + 1, rel]  # [1 2316] [1, #tri]
                obj_p = sub_p * rel_p  # [1 2316]
                obj = obj.astype(mindspore.int32)
                new_e.append(
                    ops.index_add(ops.zeros((1, self.num_ents)), obj, obj_p, axis=1))
            last_e = P.Concat(0)(new_e)

            # reshape >1 scores to 1 in a differentiable way
            # m = last_e.gt(1).float()
            # z = (m * last_e + (1 - m)).detach()
            # last_e = last_e / z

            ent_probs.append(last_e)

        hop_res = mindspore.ops.stack(ent_probs, axis=1)  # [1 2 E] [bsz, num_hop, num_ent]
        ctx_h_history = mindspore.ops.stack(ctx_h_list, axis=2)  # [1 768 2] [bsz, dim_h, num_hop]

        hop_logit = self.hop_att_layer(ctx_h_history.transpose((0,2,1))) # 1 2 1  # bsz, num_hop, 1
        hop_attn = mindspore.nn.Softmax(axis=2)(hop_logit.transpose((0,2,1))).transpose((0,2,1)) # 2 1  # bsz, num_hop, 1

        last_e = mindspore.ops.matmul(hop_attn.transpose((0,2,1)), hop_res).squeeze(1).squeeze(0)  # [bsz, num_ent]

        if not training:
            return {
                'e_score': last_e,
                'word_attns': word_attns,
                'rel_probs': rel_probs,
                'ent_probs': ent_probs,
                'hop_attn': hop_attn
            }
        else:
            weight = answers * 9 + 1
            loss = mindspore.ops.ReduceSum()(entity_range * weight * P.Pow()(last_e - answers, 2)) / mindspore.ops.ReduceSum()(entity_range * weight)

            return loss
