#coding=utf-8
#! /usr/bin/env python3.4

# import torch
# from torch.autograd import Variable
from cProfile import label
import numpy as np
# import torch.nn.functional as F 
import yaml
import argparse
from lm_quantum import quantum_repersent as GQLM
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor,Parameter
import mindspore

"""
### used torch operaters:

# torch: 			from_numpy, zeros, matmul, unsqueeze, sum, cat, argmax, LongTensor 
# nn: 				Module, Dropout, Embedding, Linear, ReLU, Parameter, 
# nn.functional:	relu, conv2d
# Tensor: 			view, size, cuda, copy_, requires_grad

### used mindspore operaters:

# mindspore: 		Parameter
# nn:				Dropout, Embedding, Dense, ReLU  
# ops:				Zeros, Concat, ReduceSum, Expand_Dims, Conv2D, Argmax, MatMul
# Tensor: 			from_numpy, view, copy, requires_grad

### lacked/useless torch operaters:

# l/u:				LongTensor(l)

"""



rng = np.random.RandomState(23455)

parser = argparse.ArgumentParser()
parser.add_argument(
	'--config-yml',
	default = 'pytorch_config.yml'
	)
args = parser.parse_args()
args.config_yml = 'pytorch_config.yml'
config = yaml.load(open(args.config_yml, 'rb'),Loader=yaml.FullLoader)

class QA_quantum(nn.Cell):
	def __init__(self, 
		         max_input_left, 
		         max_input_right, 
		         vocab_size,  
		         embeddings, 
		         config
		         ):
		super(QA_quantum, self).__init__()
		self.config = config
		self.vocab_size = vocab_size
		self.embeddings = embeddings
		self.max_input_left = max_input_left
		self.max_input_right = max_input_right
		self.dropout_keep_prob = config['model']['dropout_keep_prob']
		self.num_filters = config['model']['num_filters']
		self.embedding_size = config['model']['embedding_dim']
		self.overlap_needed = config['train']['overlap_needed']
		self.trainable = config['train']['trainable']
		self.filter_sizes = list(map(int, config['model']['filter_sizes'].split(",")))
		self.pooling = config['train']['pooling']
		self.position_needed = config['train']['position_needed']
		self.batch_size = config['train']['batch_size']
		self.l2_reg_lambda = config['model']['l2_reg_lambda']
		self.is_Embedding_Needed = config['train']['embedding_needed']   
		self.max_len = config['model']['max_len']
		self.dropout = nn.Dropout(1-self.dropout_keep_prob)

		# bilstm编码
		self.context_lstm =  nn.GRU(50, 50, has_bias=True, batch_first=True, bidirectional=True)

		# # 卷积运算
		self.conv = nn.Conv2d(in_channels=1, out_channels=self.num_filters, kernel_size=(1, 100),pad_mode='valid')
		self.bn = nn.BatchNorm2d(num_features=self.num_filters)
		self.fc1 = nn.Dense(self.num_filters * 100 * 2, 100 * 2)
		self.fc2 = nn.Dense(100 * 2, 2)
		self.relu = nn.ReLU()


		self.expand_dims = ops.ExpandDims()
		self.sum = ops.ReduceSum()
		# self.concat = ops.Concat()

		self.gqlm = GQLM(config)
		
		self.kernels = []
		if self.is_Embedding_Needed:
			self.words_embeddings = nn.Embedding(self.vocab_size, self.embedding_size,embedding_table=ops.Cast()(Tensor.from_numpy(self.embeddings),mindspore.float32)) 
		else: self.words_embeddings = nn.Embedding(self.vocab_size, self.embedding_size)     #定义变量self.words_embeddings作为词向量矩阵，维度为[词表大小，词向量维度]，元素是随机数
		# if self.is_Embedding_Needed:
		# 	self.words_embeddings.embedding_table.data.copy(Tensor.from_numpy(self.embeddings))     #当需要导入词向量时，往self.embedding_W导入词向量，并进行训练
		self.words_embeddings.embedding_table.requires_grad = True
		self.overlap_embeddings = nn.Embedding(3, self.embedding_size)     
		self.overlap_embeddings.embedding_table.requires_grad = True
		self.position_embeddings = nn.Embedding(self.max_len+1, self.embedding_size)    
		self.position_embeddings.embedding_table.requires_grad = True
		self.weights = nn.Dense(self.embedding_size, self.embedding_size)
		# self.weights2 = torch.nn.Linear(self.embedding_size, self.embedding_size)
		self.projection = nn.Dense(self.embedding_size*2, self.embedding_size)
		self.relu = nn.ReLU()
		linear_length = 0

		linear_length += self.embedding_size * self.embedding_size

		self.connected_layer_softmax = nn.Dense(linear_length * 2, 2)



	def concat_embedding(self, words_indice, overlap_indice, position_indice, overlap_needed, position_needed):
		# print('wi: ',words_indice.shape)
		words_embedding = self.words_embeddings(Tensor(words_indice))
		# print('mark4')
		position_embedding = self.position_embeddings(Tensor(position_indice))
		overlap_embedding = self.overlap_embeddings(Tensor(overlap_indice))
		return self.dropout(words_embedding+position_embedding+overlap_embedding)
	# 密度矩阵运算
	def density(self, q_norm, a_norm):
		q_max_len = q_norm.shape[1]
		a_max_len = a_norm.shape[1]
		self.w_q = Parameter(Tensor(np.random.randn(1,q_max_len,1,1),mindspore.float32), requires_grad=True)
		self.w_a = Parameter(Tensor(np.random.randn(1,q_max_len,1,1),mindspore.float32), requires_grad=True)
		# print("q_norm.shape:", q_norm_t.shape)
		# print("a_norm.shape:", a_norm_t.shape)
		# 反转
		q_norm_t = ops.transpose(q_norm,(0,1,3,2))
		a_norm_t = ops.transpose(a_norm,(0,1,3,2))

		# 矩阵相乘,得到测量算子(b,s,d,d)
		q_measure = ops.matmul(q_norm, q_norm_t)
		a_measure = ops.matmul(a_norm, a_norm_t)

		# print("q_measure.shape:", q_measure.shape)
		# print("a_measure.shape:", a_measure.shape)

		# 密度矩阵(b,dim,dim)
		op = ops.ReduceSum(keep_dims=True) # 求和
		q_density = op(q_measure, 1).squeeze()
		a_density = op(a_measure, 1).squeeze()
		# print("q_density:",q_density.shape)
		# exit()

		# 对问题密度矩阵扩充维度(b,config.a_max_len,dim,dim)
		q_bro = ops.BroadcastTo((a_max_len, -1, -1, -1))
		q_out = q_bro(q_density)
		q_density1 = ops.transpose(q_out,(1,0,2,3))

		# q_density1 = q_density.expand(a_max_len, -1, -1, -1).transpose(0, 1)
		# 对答案密度矩阵扩充维度(b,config.q_max_len,dim,dim)
		a_bro = ops.BroadcastTo((q_max_len, -1, -1, -1))
		a_out = a_bro(a_density)
		a_density1 = ops.transpose(a_out, (1, 0, 2, 3))
		# a_density1 = a_density.expand(q_max_len, -1, -1, -1).transpose(0, 1)
		# 对测量算子转置
		q_measure_t = ops.transpose(q_measure, (0,1,3,2))
		a_measure_t = ops.transpose(a_measure, (0,1,3,2))
		# q_measure_t = q_measure.transpose(3, 2)
		# a_measure_t = a_measure.transpose(3, 2)

		# 测量算子与密度矩阵相乘，得测量后的状态(b,s,dim,dim)
		qm = ops.matmul(ops.matmul(a_measure, q_density1), a_measure_t)
		am = ops.matmul(ops.matmul(q_measure, a_density1), q_measure_t)

		# 乘以单词概率
		qm = op(qm, 1).squeeze()
		am = op(am, 1).squeeze()

		return qm, am

	def construct(self, question, question_overlap, question_position, answer, answer_overlap, answer_position, overlap_needed, position_needed):
		# [32,30,50]
		q_emb = self.concat_embedding(question, question_overlap, question_position, overlap_needed, position_needed)
		a_emb = self.concat_embedding(answer, answer_overlap, answer_position, overlap_needed, position_needed)
		# bilstm编码
		q_lstm, _ = self.context_lstm(q_emb)
		a_lstm, _ = self.context_lstm(a_emb)

		q_lstm = self.dropout(q_lstm)
		a_lstm = self.dropout(a_lstm)
		# 扩充维度
		q_dim = self.expand_dims(q_lstm,-1)
		a_dim = self.expand_dims(a_lstm,-1)
		# 密度矩阵(b,dim,dim)
		q_density, a_density = self.density(q_dim, a_dim)
		# 扩充维度适应卷积网络（b,1,dim,dim)
		q_density_dim = self.expand_dims(q_density, 1)
		a_density_dim = self.expand_dims(a_density, 1)

		# 各自进行卷积(b,config.num_filters * dim)
		q_conv = self.bn(self.conv(q_density_dim)).view((q_density.shape[0], -1))
		a_conv = self.bn(self.conv(a_density_dim)).view((q_density.shape[0], -1))
		# print("q_conv.shape:",q_conv.shape)
		# print("a_conv.shape:",a_conv.shape)
		# 进行拼接 （b,config.num_filters * dim * 2)
		op_cat = ops.Concat(1)
		M_conv = op_cat((q_conv, a_conv))
		# 进行预测
		fc1 = self.fc1(self.dropout(M_conv))
		fc1 = self.dropout(self.relu(fc1))

		return self.fc2(fc1)

class QuantumWithLossCell(nn.Cell):
	def __init__(self, backbone, loss_fn):
		super(QuantumWithLossCell, self).__init__(auto_prefix=False)
		self._backbone = backbone
		self._loss_fn = loss_fn
	
	def construct(self, question, question_overlap, question_position, answer, answer_overlap, answer_position, overlap_needed, position_needed, label):
		out = self._backbone(question, question_overlap, question_position, answer, answer_overlap, answer_position, overlap_needed, position_needed)
		return self._loss_fn(out,label)

	@property
	def backbone_network(self):
		return self._backbone