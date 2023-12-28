#coding=utf-8
#! /usr/bin/env python3.4

import mindspore as ms
import mindspore.nn as nn
# import torch
# import torch.nn as nn
import numpy as np
# import torch.nn.functional as F
import mindspore.ops as ops
import mindspore.numpy


"""
### used torch operaters:

# torch: 			sort, matmul, randn, zeros, ones, mul, sum, eye
# nn: 				Module, Parameter, Sigmoid, ReLU, Tanh,
# nn.functional:	normalize, relu
# Tensor: 			permute, squeeze, unsqueeze, cuda, index_select, repeat

### used mindspore operaters:

# mindspore: 		Parameter
# nn:				Cell, Sigmoid, ReLu, Tanh,  
# ops:				StandardNormal, Sort, Transpose, Zeros, L2Normalize, ReduceSum, Mul, Expand_Dims, MatMul
# Tensor: 			squeeze
# numpy:			tile

### lacked/useless torch operaters:

# l/u:				cuda(u), index_select(l)

"""



class quantum_repersent(nn.Cell):
	def __init__(self, config):
		super(quantum_repersent, self).__init__()
		self.batch_size = config['train']['batch_size']
		self.feature_size = config['model']['feature_dim']
		self.max_input_sentence = config['model']['max_len']
		self.embedding_size = config['model']['embedding_dim']
		self.standardnormal = ops.StandardNormal()
		# self.weights_sentence = torch.nn.Parameter(torch.ones(1, self.max_input_sentence, 1, 1))
		# self.weights_feature = torch.nn.Parameter(torch.ones(self.feature_size))

		# self.unitary_matrix_rdm = torch.nn.Parameter(torch.zeros((self.max_input_sentence, self.feature_size, self.feature_size)))
		# self.unitary_matrix_w = torch.nn.Parameter(torch.zeros((self.max_input_sentence, self.embedding_size, self.embedding_size)))

		# self.input_len = config['model']['input_len']
		# self.embedding_size_2 = config['model']['embedding_dim_2']
		self.weights_words = ms.Parameter(self.standardnormal((1, self.max_input_sentence, 1))/np.sqrt(self.max_input_sentence))
		self.b = ms.Parameter(self.standardnormal((1, self.max_input_sentence))/np.sqrt(self.max_input_sentence))
		self.sentence_vec = ms.Parameter(self.standardnormal((1, self.embedding_size))/np.sqrt(self.embedding_size))
		# self.epsilon_sen = torch.nn.Parameter(torch.zeros(self.max_input_sentence))
		self.epsilon_wor = ms.Parameter(self.standardnormal((1, self.max_input_sentence, 1, 1))/np.sqrt(self.max_input_sentence))
		self.sig = nn.Sigmoid()
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		# self.M = torch.nn.Linear(self.embedding_size, self.embedding_size_2)

		self.tranpose = ops.Transpose()
		self.sort = ops.Sort()
		self.mul = ops.Mul()
		self.sum = ops.ReduceSum()
		# self.matmul = ops.matmul()
		self.expand_dims = ops.ExpandDims()

		


	# def density_matrix_sentence(self, question_matrix, sentence_weighted):
	# 	norm = torch.nn.functional.normalize(question_matrix, p = 2, dim = 2)
	# 	print(norm.size())
	# 	reverse_matrix = norm.permute(0, 1, 3, 2)
	# 	sentence_representation = torch.matmul(norm, reverse_matrix)
	# 	return torch.sum(torch.mul(sentence_representation, sentence_weighted), 1)

	def select_words(self, question_matrix):
		reverse_matrix = self.tranpose(question_matrix,(0, 1, 3, 2))
		_, ids = self.sort((ops.matmul(reverse_matrix, question_matrix)).squeeze(), descending = True)
		idx, _ = self.sort(self.tranpose((self.tranpose(ids,(1, 0)))[: self.max_input_sentence],(1, 0)), descending = False)
		new_matrix = ops.Zeros()(self.batch_size, self.max_input_sentence, self.embedding_size)
		for id in range(len(idx)):
			new_matrix[id] = (reverse_matrix.squeeze())[id].index_select(0, idx[id])			# needed to change
		new_matrix = self.expand_dims(new_matrix, -1)
		return new_matrix

	def words_matrix(self, question_matrix, epsilon_wor, weights_words, sentence_origin):
		# print(question_matrix.size())32,30,50,1
		norm = ops.L2Normalize(axis=2)(question_matrix)
		reverse_matrix = self.tranpose(norm,(0, 1, 3, 2))
		words_matrixs = self.mul(ops.matmul(norm, reverse_matrix), epsilon_wor) #构造酉矩阵用
		# words_matrixs = torch.matmul(norm, reverse_matrix) * 0.1
		p_w = ops.Ones()((self.batch_size, 1), mindspore.float32)
		for id in range(self.max_input_sentence):
			w_i = self.tranpose(norm,(1, 0, 2, 3))[id]
			p_w_i = self.sig(ops.ReduceSum()(self.mul(self.mul(ops.matmul(norm.squeeze(), w_i), ops.matmul(norm.squeeze(), w_i)), weights_words), 1) + (self.tranpose(self.b, (1, 0)))[id])
			p_w = self.sig(self.mul(p_w, p_w_i))
		# print(p_w)
		sentence = ops.L2Normalize(axis=1)(self.mul(ops.L2Normalize(axis=1)(sentence_origin), p_w))
		return words_matrixs, sentence
	
	def unitary_matrix_sentence(self, words_matrixs, sentence):
		identity_matrix = mindspore.numpy.tile(ops.Eye()(self.embedding_size, self.embedding_size, mindspore.float32), (self.batch_size, 1, 1))
		unitary_matrix = self.expand_dims(identity_matrix, 1) + words_matrixs
		unitary_matrix = self.tranpose(unitary_matrix,(1, 0, 2, 3))
		# print("sssss", sentence.size())
		sentence = self.expand_dims(sentence, 1)
		for id in range(self.max_input_sentence):
			sentence_i = ops.matmul(sentence, unitary_matrix[id])
			sentence = self.relu(sentence_i) + sentence * 0.01   #0.011wiki
			# sentence = torch.nn.functional.normalize(sentence_i)
		# print(sentence.size())
		reverse_sentence = self.tranpose(sentence,(0, 2, 1))
		sentence_matrix = ops.matmul(reverse_sentence, sentence)
		return sentence_matrix
		# return sentence

	def density_matrix(self, sentence_matrix):
		norm = ops.L2Normalize(axis=2)(sentence_matrix)
		reverse_matrix = self.tranpose(norm,(0, 1, 3, 2))
		single_sentence_representation = ops.matmul(norm, reverse_matrix)
		# output = torch.relu(torch.sum(single_sentence_representation, 1))
		output = self.relu(ops.ReduceSum()(single_sentence_representation, 1))		# perhaps wrong
		return output

	def construct(self, words_indice, sentence_embedding):
		# matrix = self.select_words(sentence_embedding)
		# res = self.density_matrix(matrix)
		# words_matrixs, sentence = self.words_matrix(matrix, self.epsilon_wor, self.weights_words, self.sentence_vec)
		res = self.density_matrix(sentence_embedding)
		words_matrixs, sentence = self.words_matrix(sentence_embedding, self.epsilon_wor, self.weights_words, self.sentence_vec)
		sentence_matrix = self.unitary_matrix_sentence(words_matrixs, sentence)
		# sentence = self.unitary_matrix_sentence(words_matrixs, sentence)

		# print(self.epsilon_wor)
		return sentence_matrix + res
		# return sentence

	# def feature_vector_fun(self, feature_vectors, words_indice, pos):
	# 	feature_vector = torch.mul(self.weights_feature, feature_vectors)
	# 	if pos == 0:
	# 		return feature_vector
	# 	else:
	# 		unitary_matrix_rdm = self.unitary_matrix_rdm[pos - 1]
	# 		feature_vector = torch.matmul(feature_vectors.expand(self.batch_size, self.feature_size, self.feature_size), unitary_matrix_rdm)
	# 		return feature_vector


	# def a_t(self, question_matrix, density_matrix_sentence, feature_vector, pos):
	# 	norm = torch.nn.functional.normalize(question_matrix, p = 2, dim = 2)
	# 	word = norm.permute(1, 0, 2, 3)[0]
	# 	word_T = word.permute(0, 2, 1)
	# 	w0_wj = torch.sqrt(torch.matmul(torch.matmul(word_T, density_matrix_sentence), word))
	# 	product_weight_feature_vector = torch.mul(w0_wj, feature_vector)

	# 	word_pos = norm.permute(1, 0, 2, 3)[pos]
	# 	word_pos_T = word.permute(0, 2, 1)
	# 	word_pos_T = word_pos_T.expand(self.batch_size, self.feature_size, self.embedding_size)

	# 	fv_kr_wp = torch.mul(word_pos_T.unsqueeze(3), product_weight_feature_vector.unsqueeze(2))
	# 	fv_kr_wp = torch.reshape(fv_kr_wp.permute(0, 1, 3, 2), [self.batch_size, self.feature_size, self.feature_size * self.embedding_size, 1])
	# 	fv_kr_wp_T = fv_kr_wp.permute(0, 1, 3, 2)
	# 	fv_wp_matrix = torch.sum(torch.matmul(fv_kr_wp, fv_kr_wp_T), 1)

	# 	diagonal_nums = torch.diagonal(fv_wp_matrix, dim1 = -2, dim2 = -1)
	# 	trace_p = torch.sum(diagonal_nums, 1)
	# 	prob = self.sig(trace_p)
	# 	return prob
		
	# def reduced_density_matrix(self, question_matrix, density_matrix_sentence, feature_vector, pos, prob):
	# 	norm = torch.nn.functional.normalize(question_matrix, p = 2, dim = 2)
	# 	word = norm.permute(1, 0, 2, 3)[0]
	# 	word_T = word.permute(0, 2, 1)
	# 	w0_wj = torch.sqrt(torch.matmul(torch.matmul(word_T, density_matrix_sentence), word))
	# 	product_weight_feature_vector = torch.mul(w0_wj, feature_vector)
	# 	product_weight_feature_vector_T = product_weight_feature_vector.permute(0, 2, 1)
	# 	tr_f = torch.matmul(product_weight_feature_vector, product_weight_feature_vector_T)
	# 	diagonal_nums = torch.diagonal(tr_f, dim1 = -2, dim2 = -1)
	# 	tr_f = torch.sum(diagonal_nums, 1)
	# 	pra = torch.div(tr_f, prob)

	# 	unitary_matrix_w = self.unitary_matrix_w[pos]
	# 	word_pos = norm.permute(1, 0, 2, 3)[pos]
	# 	word_pos_T = word.permute(0, 2, 1)
	# 	uw_wp = torch.matmul(unitary_matrix_w, word_pos)
	# 	uw_wp_T = uw_wp.permute(0, 2, 1)
	# 	w = torch.matmul(uw_wp, uw_wp_T)
	# 	rdm = torch.mul(w, pra.unsqueeze(1).unsqueeze(2))
	# 	return rdm

	# def density_matrix(self, sentence_matrix):
	# 	norm = torch.nn.functional.normalize(sentence_matrix, p = 2, dim = 2)
	# 	reverse_matrix = norm.permute(0, 1, 3, 2)
	# 	single_sentence_representation = torch.matmul(norm, reverse_matrix)
	# 	output = torch.relu(torch.sum(single_sentence_representation, 1))
	# 	return output

	# def forward(self, words_indice, sentence_embedding):
	# 	res = self.density_matrix(sentence_embedding)

	# 	density_matrix_sentence = self.density_matrix_sentence(sentence_embedding, self.weights_sentence)

	# 	feature_vector = torch.eye(self.feature_size).cuda()

	# 	feature_vector = self.feature_vector_fun(feature_vector, words_indice, self.max_input_sentence-2)
	# 	prob = self.a_t(sentence_embedding, density_matrix_sentence, feature_vector, self.max_input_sentence-2)
	# 	reduced_density_matrix = self.reduced_density_matrix(sentence_embedding, density_matrix_sentence, feature_vector, self.max_input_sentence-1, prob)
		
	# 	return reduced_density_matrix + res
