import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import SparseTensor, COOTensor
from mindspore import Tensor, Parameter, CSRTensor
from mindspore.common.initializer import Normal
import mindspore_hub as mshub
import mindspore
import mindspore.ops as ops

sparse_dense_matmul = mindspore.ops.SparseTensorDenseMatmul()
class GFC(nn.Cell): #
    def __init__(self, args, ent2id, rel2id, triples):
        super().__init__()
        self.args = args
        self.num_steps = 3
        self.num_relations = len(rel2id)

        self.Tsize = len(triples)
        self.Esize = len(ent2id)
        self.triples = triples
        try:
            if args.bert_name == "bert-base-uncased":
                model = "mindspore/1.9/bertfinetune_ner_cluener" # "mindspore/1.9/bertbase_cnnews128"
                self.bert_encoder = mshub.load(model)
            elif args.bert_name == "roberta-base":
                model = "mindspore/1.9/bertfinetune_ner_cluener" # "mindspore/1.9/bertbase_cnnews128"
                self.bert_encoder = mshub.load(model)
            else:
                raise ValueError("please input the right name of pretrained model")
        except ValueError as e:
            raise e
        dim_hidden = 768 #self.bert_encoder.config.hidden_size
        self.rel_classifier = nn.Dense(dim_hidden, self.num_relations)
        self.key_layer = nn.Dense(dim_hidden, dim_hidden, weight_init=Normal(0.02))
        self.hop_att_layer = nn.SequentialCell(
            [nn.Dense(dim_hidden, 1)]
        )

        self.high_way = nn.SequentialCell(
            [nn.Dense(dim_hidden, dim_hidden, weight_init=Normal(0.02)),
            nn.Sigmoid()]
        )
        self.idx = mindspore.Tensor([i for i in range(self.Tsize+1)])

    def follow(self, e, r):
        # x1 = sparse_dense_matmul(self.Msubj.indices, self.Msubj.values, self.Msubj.sparse_shape, e.transpose((1,0)))
        # x2 = sparse_dense_matmul(self.Mrel.indices, self.Mrel.values, self.Mrel.sparse_shape,  r.transpose((1,0)))
        e = e.unsqueeze(0)
        x1 = self.Msubj.mv(e.transpose((1, 0)))
        x2 = self.Mrel.mv(r.transpose((1, 0)))
        x = x1*x2
        # return sparse_dense_matmul(self.Mobj, x.transpose((1,0)))  # [bsz, Esize]
        # return self.Mobj.mv(x.transpose((1, 0)))  # [bsz, Esize]
        # output = ops.mv(x.transpose((1, 0)), self.Mobj)
        output = self.Mobj.T.mv(x)
        return output.transpose((1, 0))  # [bsz, Esize]

    def construct(self, heads, questions, answers=None, entity_range=None, training=True):
        # sparse tensor只能在cell的construct方法中
        # mindspore.ops.stack((self.idx, self.triples[:,0]))
        self.Msubj = CSRTensor(self.idx.astype(mindspore.int32), self.triples[:,0].astype(mindspore.int32), mindspore.Tensor([1] * self.Tsize, mindspore.float32), (self.Tsize, self.Esize))
        self.Mobj = CSRTensor(self.idx.astype(mindspore.int32), self.triples[:,2].astype(mindspore.int32), mindspore.Tensor([1] * self.Tsize, mindspore.float32), (self.Tsize, self.Esize))
        self.Mrel = CSRTensor(self.idx.astype(mindspore.int32), self.triples[:,1].astype(mindspore.int32), mindspore.Tensor([1] * self.Tsize, mindspore.float32), (self.Tsize, self.num_relations))
        for item in questions:
            questions[item] = mindspore.Tensor(questions[item])
        if answers is not None:
            answers = mindspore.Tensor(answers)
        if entity_range is not None:
            entity_range = mindspore.Tensor(entity_range)
        if "attention_mask" in questions:
            questions['input_mask'] = questions.pop("attention_mask")

        q = self.bert_encoder(**questions)
        q_word_h, q_embeddings, _ = q[0], q[1], q[2]  # (bsz, dim_h), (bsz, len, dim_h)
        # q_word_h.set_dstype(mindspore.float32)
        # q_word_h = q_word_h.astype(mindspore.float32)
        heads = mindspore.Tensor(heads)
        heads = heads.astype(mindspore.float32)
        # entity_range = entity_range.astype(mindspore.float32)
        # answers = answers.astype(mindspore.float32)
        last_e = heads
        word_attns = []
        rel_probs = []
        ent_probs = []
        ctx_h_list = []
        q_word_h_hop = q_word_h
        q_word_h_dist_ctx = [0]
        # last_e = last_e.unsqueeze(0)
        for t in range(self.num_steps):
            h_key = self.key_layer(q_word_h_hop)  # [bsz, max_q, dim_h]
            q_logits = mindspore.ops.matmul(h_key, q_word_h.transpose((0,2,1))) # 1 64 768 1 768 64[bsz, max_q, dim_h] * [bsz, dim_h, max_q] = [bsz, max_q, max_q]
            q_logits = q_logits.transpose((0,2,1))  # 1 64 64
            q_dist = mindspore.nn.Softmax(axis=2)(q_logits)  # [bsz, max_q, max_q]
            q_dist = q_dist * questions['input_mask'].astype(mindspore.float32).unsqueeze(1)  # 1 64 64 [bsz, max_q, max_q]*[bsz, max_q]
            q_dist = q_dist / (mindspore.ops.ReduceSum(keep_dims=True)(q_dist, axis=2) + 1e-6) #1 64 64 [bsz, max_q, max_q]
            hop_ctx = mindspore.ops.matmul(q_dist, q_word_h_hop) # 1 64 768
            if t == 0:
                z = 0
            else:
                z = self.high_way(q_word_h_dist_ctx[-1]) 
            if t == 0:
                q_word_h_hop = q_word_h + hop_ctx
            else:
                q_word_h_hop = q_word_h + hop_ctx + z*q_word_h_dist_ctx[-1]# [bsz, max_q, max_q]*[bsz, max_q, dim_h] = [bsz, max_q, dim_h]
            q_word_h_dist_ctx.append(hop_ctx + z*q_word_h_dist_ctx[-1])
            q_word_att = mindspore.ops.ReduceSum(keep_dims=True)(q_dist, axis=1)  # 1 1 64[bsz, 1, max_q]  # 2改为1
            q_word_att = mindspore.nn.Softmax(axis=2)(q_word_att) # 1 1 64
            q_word_att = q_word_att * questions['input_mask'].float().unsqueeze(1)  # [bsz, 1, max_q]*[bsz, max_q]
            q_word_att = q_word_att / (mindspore.ops.ReduceSum(keep_dims=True)(q_word_att, axis=2) + 1e-6)  #1 1 64 [bsz, max_q, max_q]
            word_attns.append(q_word_att)  # bsz,1,q_max
            ctx_h = mindspore.ops.matmul(q_word_h_hop.transpose((0,2,1)), q_word_att.transpose((0,2,1))).squeeze(2)  # 1 768[bsz, dim_h, max_q] * [bsz, max_q,1]
            ctx_h_list.append(ctx_h)
            rel_logit = self.rel_classifier(ctx_h)  # 1 1144 [bsz, num_relations]
            rel_dist = mindspore.ops.Sigmoid()(rel_logit);rel_probs.append(rel_dist);last_e = last_e
            # last_e = self.follow(last_e, rel_dist)  # faster than index_add
            # reshape >1 scores to 1 in a differentiable way
            # m = last_e.gt(1).float()
            # z = (m * last_e + (1-m)).detach()
            # last_e = last_e / z
            ent_probs.append(last_e)
        hop_res = mindspore.ops.stack(ent_probs, axis=1) # 1 2 E [bsz, num_hop, num_ent]
        ctx_h_history = mindspore.ops.stack(ctx_h_list, axis=2)  #1 768 2 [bsz, dim_h, num_hop]
        hop_logit = self.hop_att_layer(ctx_h_history.transpose((0,2,1)))  # 1 2 1 bsz, num_hop, 1
        hop_attn = mindspore.nn.Softmax(axis=2)(hop_logit.transpose((0,2,1))).transpose((0,2,1)).squeeze(0)  # bsz, num_hop, 1
        # last_e = mindspore.ops.ReduceSum()(mindspore.ops.matmul(hop_res, hop_attn), axis=1) # 1 2 E * 1 2 1[bsz, num_ent]
        last_e = mindspore.ops.matmul(hop_res, hop_attn).squeeze(1)  # 1 2 E * 1 2 1[bsz, num_ent]
        if not training:
            return {
                'e_score': last_e,
                'word_attns': word_attns,
                'rel_probs': rel_probs,
                'ent_probs': ent_probs,
                'hop_attn': hop_attn
            }
        else:
            weight = 10
            loss = mindspore.ops.ReduceSum()(P.Pow()(last_e - answers, 2))*weight / mindspore.ops.ReduceSum()(entity_range)
            return loss



