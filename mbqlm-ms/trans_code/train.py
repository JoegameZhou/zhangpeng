#coding=utf-8
#! /usr/bin/env python3.4
# import torch
from matplotlib.style import context
import mindspore
import numpy as np
import os
import time
import datetime


from pytorch_helper import get_overlap_dict, batch_gen_with_point_wise, load, load_trec, prepare, batch_gen_with_single, load_wiki, load_yahoo
import operator
from pytorch_QA_CNN_quantum import QA_quantum as QA_CNN_quantum_model, QuantumWithLossCell 

# import sys
# print(sys.path)

import random
from functools import wraps
import evaluation
from sklearn.model_selection import train_test_split
import yaml
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import sys
from optimization import *
from tqdm import tqdm

import mindspore.nn as nn
from mindspore import Tensor,save_checkpoint,context
import mindspore.ops as ops
import mindspore

import setproctitle
setproctitle.setproctitle("quantumn_lm_mindspore_dr0.5")

parser = argparse.ArgumentParser()
parser.add_argument(
	'--config-yml',
	default = 'pytorch_config.yml'
	)
args = parser.parse_args()
args.config_yml = 'pytorch_config.yml'
config = yaml.load(open(args.config_yml),Loader=yaml.FullLoader)

import setproctitle
setproctitle.setproctitle("qlm_ms_"+config['train']['tid'])

# if config['model']['seed'] != 0:
# 	torch.manual_seed(config['model']['seed'])
# 	torch.cuda.manual_seed(config['model']['seed'])

mindspore.set_seed(config['model']['seed'])

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
context.set_context(device_target='GPU',device_id=config['train']['gpunum'], mode=1)
 
now = int(time.time())
timeArray = time.localtime(now)
timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
timeDay = time.strftime("%Y%m%d", timeArray)
# print(timeStamp)

def log_time_delta(func):
	@wraps(func)
	def _deco(*args, **kwargs):
		start = time.time()
		ret = func(*args, **kwargs)
		end = time.time()
		delta = end - start
		# print( "%s runed %.2f seconds"% (func.__name__,delta))
		return ret
	return _deco

log_dir = 'log/'+'overlap_posi_sum'+ timeDay
if not os.path.exists(log_dir):
	os.makedirs(log_dir)
data_file = log_dir + '/test_' + config['train']['data'] + timeStamp
para_file = log_dir + '/test_' + config['train']['data'] + timeStamp + '_para'
precision = data_file + 'precise' + config["model"]["count"]

@log_time_delta
def predict(my_model, test, alphabet, batch_size, question_len, answer_len):
	scores = []
	arg_max = ops.Argmax()
	mean_func = ops.ReduceMean()
	equal_func = ops.Equal()
	cast_func = ops.Cast()
	overlap_dict = get_overlap_dict(test, alphabet, question_len, answer_len)
	# loss_function = nn.SoftmaxCrossEntropyWithLogits(reduction = 'mean', sparse=True)

	for data in batch_gen_with_single(test, alphabet, batch_size, question_len, answer_len, overlap_dict = overlap_dict): 
		score = my_model(
						question = Tensor(data[0],mindspore.int32), 
						answer = Tensor(data[1],mindspore.int32), 
						question_overlap = Tensor(data[2],mindspore.int32), 
						answer_overlap = Tensor(data[3],mindspore.int32), 
						question_position = Tensor(data[4],mindspore.int32), 
						answer_position = Tensor(data[5],mindspore.int32), 
						overlap_needed = config['train']['overlap_needed'], 
						position_needed = config['train']['position_needed']
						)

		# predictions = ops.Argmax()(score, 1)
		# loss = loss_function(score, arg_max(Tensor(data[6]), 1))
		# print('evaluation loss:  ', loss)
		# current_predictions = equal_func(predictions, arg_max(Tensor(data[6]), 1))
		# acc = mean_func(cast_func(current_predictions, mindspore.float32))
		# print('eval loss: {}, eval acc: {} ', loss, acc)

		scores.extend(score.asnumpy())

	output = np.array(scores[:len(test)])
	return output

@log_time_delta
def test_point_wise():

	train, dev, test = load_wiki(config['train']['data'], filter = config['train']['clean'])

	q_max_sent_length = config['model']['max_len']
	a_max_sent_length = config['model']['max_len']
	# q_max_sent_length = config['model']['input_len']
	# a_max_sent_length = config['model']['input_len']
	

	# print ('train question unique:{}'.format(len(train['question'].unique())))
	# print ('train length', len(train))
	# print ('test length', len(test))
	# print ('dev length', len(dev))
	alphabet, embeddings = prepare([train, test, dev], dim = config['model']['embedding_dim'], is_embedding_needed = True, fresh = True)
	# print ('alphabet:', len(alphabet))

	log = open(precision, 'w')
	s = 'embedding_dim:' + str(config['model']['embedding_dim']) + \
	'\n' + 'feature_dim:' + str(config['model']['feature_dim']) + \
	'\n' + 'filter_sizes:' + str(config['model']['filter_sizes']) + \
	'\n' + 'num_filters:' + str(config['model']['num_filters']) + \
	'\n' + 'dropout_keep_prob:' + str(config['model']['dropout_keep_prob']) + \
	'\n' + 'l2_reg_lambda:' + str(config['model']['l2_reg_lambda']) + \
	'\n' + 'learning_rate:' + str(config['model']['learning_rate']) + \
	'\n' + 'batch_size:' + str(config['train']['batch_size']) + \
	'\n' + 'trainable:' + str(config['train']['trainable']) + \
	'\n' + 'num_epochs:' + str(config['train']['num_epochs']) + \
	'\n' + 'data:' + str(config['train']['data']) + \
	'\n' + 'seed:' + str(config['model']['seed']) + \
	'\n' + 'tid:' + str(config['train']['tid']) + \
	'\n'
	log.write(str(s) + '\n')
	my_model = QA_CNN_quantum_model(
		max_input_left = q_max_sent_length,
		max_input_right = a_max_sent_length,
		vocab_size = len(alphabet),
		embeddings = embeddings,
		config = config
		)
	
	
	# scheduler = nn.ExponentialDecayLR(config['model']['learning_rate'], decay_rate=0.1, decay_steps=1)
	optimizer = nn.Adam(my_model.trainable_params(), 
						learning_rate = linear_schedule_with_warmup_iterator(config['model']['learning_rate'],
																			num_warmup_steps=int((config['train']['num_epochs'] * 136) / 100),
																			num_training_steps=config['train']['num_epochs'] * 136,
																			total_steps=config['train']['num_epochs']*config['train']['checkpoint_every']),
						weight_decay = config['model']['l2_reg_lambda'], eps=1e-08)
	# optimizer = torch.optim.SGD(my_model.parameters(), lr = config['model']['learning_rate'], weight_decay = config['model']['l2_reg_lambda'])
	# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int((config['train']['num_epochs'] * 136) / 100), num_training_steps=config['train']['num_epochs'] * 136)
	loss_function = nn.SoftmaxCrossEntropyWithLogits(reduction = 'mean', sparse = True)
	net_with_loss = QuantumWithLossCell(my_model, loss_function)
	train_network = nn.TrainOneStepCell(net_with_loss, optimizer)
	train_network.set_train()

	arg_max = ops.Argmax(axis=1)
	sum_func = ops.ReduceSum()
	equal_func = ops.Equal()
	cast_func = ops.Cast()
	map_max = config['model']['map_max']
	count = 0
	map_current_max = 0
	flagmrr = 0

	for epoch in range(config['train']['num_epochs']):
		datas = batch_gen_with_point_wise(train, alphabet, config['train']['batch_size'], overlap_dict = None, q_len = q_max_sent_length, a_len = a_max_sent_length)
		mcount = 0
		for data in datas:
			# optimizer.zero_grad() 
			# scores = my_model(question = data[0], answer = data[1], question_overlap = data[3], question_position = data[5], answer_overlap = data[4], answer_position = data[6], overlap_needed = config['train']['overlap_needed'], position_needed = config['train']['position_needed'])
			# current_predictions = equal_func(predictions,arg_max(Tensor(data[2]), 1))
			# accuracy = sum_func(cast_func(current_predictions, mindspore.float32))
			time_str = datetime.datetime.now().isoformat()
			# print(data[0])
			# loss = loss_function(scores, arg_max(Tensor(data[2]), 1))
			loss = train_network(
								Tensor(data[0],mindspore.int32), 				# question
								Tensor(data[3],mindspore.int32),				# question_overlap
								Tensor(data[5],mindspore.int32),				# question_position
								Tensor(data[1],mindspore.int32),				# answer
								Tensor(data[4],mindspore.int32), 				# answer_overlap
								Tensor(data[6],mindspore.int32), 				# answer_position
								config['train']['overlap_needed'], 				# overlap_needed
								config['train']['position_needed'], 			# position_needed
								arg_max(Tensor(data[2],mindspore.float32))		# label
								)
			
			# loss.backward()
			# optimizer.step()
			# scheduler.step()

			count += 1
			mcount += 1
			# print(loss.item())
			print("{}: batch {}, loss {} ".format(time_str, mcount, loss))
			if count % config['train']['evaluate_every'] == 0: 
				predicted = predict(my_model, dev, alphabet, config['train']['batch_size'], q_max_sent_length, a_max_sent_length)
				map_mrr_dev = evaluation.evaluationBypandas(dev, predicted[:, -1])
				predicted_test = predict(my_model, test, alphabet, config['train']['batch_size'], q_max_sent_length, a_max_sent_length)
				map_mrr_test = evaluation.evaluationBypandas(test, predicted_test[:, -1])
					
				if map_mrr_test[0] > map_current_max:
					map_current_max = map_mrr_test[0]
				else:
					flagmrr += 1

				if map_mrr_test[0] > map_max:
					map_max = map_mrr_test[0]
					folder = 'runs/' + timeDay
					out_dir = folder + '/' + config['train']['data']
					if not os.path.exists(folder):
						os.makedirs(folder)
					# state = {'net':my_model.state_dict(), 'optimizer':optimizer.state_dict()}
					save_path = save_checkpoint(my_model, out_dir+'dr0.5.ckpt')
					print ("Model saved in file: ", save_path)

				print ("{}:dev step:map mrr {}".format(count // config['train']['evaluate_every'], map_mrr_dev))
				print ("{}:test step:map mrr {}".format(count // config['train']['evaluate_every'], map_mrr_test))

				line1 = " {}:step: map_dev{}".format(count // config['train']['evaluate_every'], map_mrr_dev)
				line2 = " {}:step: map_test{}".format(count // config['train']['evaluate_every'], map_mrr_test)
				log.write(line1 + '\t' + line2 + '\n')
				log.flush()
			if flagmrr > 50:
				exit()
	log.close()

if __name__ == '__main__':
    if config['model']['loss'] == 'point_wise':
        test_point_wise()