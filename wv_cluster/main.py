# coding: utf-8

import numpy as np
import argparse, time
import data, model
import cupy
import chainer
from chainer import links as L
from chainer import iterators, optimizers, training
from chainer.training import extensions
from chainer.datasets import tuple_dataset

from sklearn.model_selection import KFold


###############
#	引数処理
###############
def get_arg():
	parser = argparse.ArgumentParser(description='converter')
	parser.add_argument('-gpu','--gpu', type=int, default=0)
	parser.add_argument('-epoch','--epoch', type=int, default=50)
	parser.add_argument('-type','--type', type=str, default='ffnn')
	parser.add_argument('-dim','--dim', type=int, default=100)
	parser.add_argument('-w2v','--w2v', type=int, default=1)
	parser.add_argument('-units','--units', type=int, default=50)
	parser.add_argument('-size','--size', type=int, default=2)	# 前後形態素取得数(2=>前後それぞれ2取得)
	parser.add_argument('-n_cluster','--n_cluster', type=int, default=4)
	parser.add_argument('-cluster','--cluster', type=int, default=1)
	args = parser.parse_args()
	return args

def writeResult(fpath, sent):
	with open(fpath, 'a') as f:
		f.write(sent+'\n')

if __name__ == '__main__':
	start = time.time()
	args = get_arg()
	chainer.using_config('use_cudnn', True)
	######### クラスタリングなし #########
	if args.cluster == 0:
		# # 曖昧性のあるMWEのall
		# sentlist = data.load('../nn/tmp/sentlist'+str(args.size)+'.pickle')
		# featlist = data.load('../nn/tmp/featlist'+str(args.size)+'.pickle')
		# marklist = data.load('../nn/tmp/marklist'+str(args.size)+'.pickle')
		# y = data.load('../nn/tmp/labels'+str(args.size)+'.pickle')
		# vocab = data.load('../nn/tmp/vocab'+str(args.size)+'.pickle')
		# fvocab = data.load('../nn/tmp/fvocab'+str(args.size)+'.pickle')
		# embeddings = data.load('../nn/tmp/initialW_'+str(args.dim)+'_size'+str(args.size)+'.pickle')

		# all
		sentlist = data.load('./result/sentlist_all.pickle')
		featlist = data.load('./result/featlist_all.pickle')
		marklist = data.load('./result/marklist_all.pickle')
		y = data.load('./result/labels_all.pickle')
		vocab = data.load('./result/vocab_all.pickle')
		fvocab = data.load('./result/fvocab_all.pickle')
		embeddings = data.load('./result/embedding_size'+str(args.dim)+'_all.pickle')

		x = []
		fmaxlen = len(featlist[0])	# featlist内の各要素は既にpadding済み
		# print(fmaxlen)	# 16
		sentlist = data.padding(sentlist, fmaxlen)
		marklist = data.padding(marklist, fmaxlen)
		for sent, feat, mark in zip(sentlist, featlist, marklist):
			# print(sent, feat, mark)
			x.append([sent, feat, mark])

		vocab_inv = {v: k for k, v in vocab.items()}
		n_labels = 2
		batch_size = 30
		total_acc = 0.0
		total_rule_acc = 0.0

		print('gpu:',args.gpu)
		print('modeltype:',args.type)
		print('dim:', args.dim)
		print('getsize:', args.size)
		print('len(vocab):', len(vocab))

		if args.gpu >= 0:
			xp = cupy
		else:
			xp = np

		x = xp.array(x, dtype=xp.int32)
		y = xp.array(y, dtype=xp.int32)
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=53)

		# trainとtestの0/1分布確認
		print('y_train:', y_train.tolist().count(0), y_train.tolist().count(1), y_train.tolist().count(1)/len(y_train.tolist()))
		print('y_test:', y_test.tolist().count(0), y_test.tolist().count(1), y_test.tolist().count(1)/len(y_test.tolist()))
		rule_acc = y_test.tolist().count(1)/len(y_test.tolist())
		if rule_acc < 0.5:
			rule_acc = 1 - rule_acc

		train_data = tuple_dataset.TupleDataset(x_train, y_train)
		test_data = tuple_dataset.TupleDataset(x_test, y_test)
		train_iter = iterators.SerialIterator(train_data, batch_size, shuffle=True)
		test_iter = iterators.SerialIterator(test_data, batch_size, repeat=False, shuffle=False)

		# Set up a model
		if args.type == 'ffnn':
			nn = L.Classifier(model.FFNN(len(vocab), args.dim, len(fvocab), args.units, n_labels, args.w2v, embeddings))
		elif args.type == 'bilstm':
			nn = L.Classifier(model.BiLSTM(len(vocab), args.dim, len(fvocab), args.units, n_labels, args.w2v, embeddings))
		if args.gpu >= 0:
			nn.to_gpu()

		# Set up an optimizer
		optimizer = optimizers.Adam(alpha=0.01, beta1=0.9, beta2=0.999, eps=1e-6)
		optimizer.setup(nn)
		optimizer.use_cleargrads()
		updater = training.StandardUpdater(train_iter, optimizer)

		# Set up a trainer
		trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='result')
		trainer.extend(extensions.LogReport())
		trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy']))
		trainer.extend(extensions.ProgressBar())
		trainer.run()

		###### Evaluate the final model ######
		print('--------------------------')
		print('test')
		eval_nn = nn.copy()
		eval_predictor = eval_nn.predictor
		eval_predictor.train = False

		# Reset the RNN state at the beginning of each evaluation
		# if args.type in ['gru', 'bilstm']:
		# 	eval_predictor.reset_state()		# BLSTMBaseのreset_state 

		evaluator = extensions.Evaluator(test_iter, eval_nn, device=args.gpu)
		result = evaluator()
		test_acc = result['main/accuracy']
		print('rule_acc', rule_acc)
		print('validation/main/accuracy, validation/main/loss:', result['main/accuracy'], result['main/loss'])
		print('--------------------------')

		# # Serialize the final model
		modelpath = './result/'+'model_'+args.type+'_dim'+str(args.dim)+'_all.npz'
		chainer.serializers.save_npz(modelpath, nn)
		# writeResult('./result/result_acc.txt', 'rule: '+'all_'+str(rule_acc))
		# writeResult('./result/result_acc.txt', 'ffnn: '+'all_'+str(test_acc))


	elif args.cluster == 1:
		######## クラスタリングあり #########	
		args = get_arg()
		print('gpu:',args.gpu)
		print('modeltype:',args.type)
		print('n_cluster:', args.n_cluster)
		print('dim:', args.dim)
		print('getsize:', args.size)

		chainer.using_config('use_cudnn', True)
		for i in range(1, args.n_cluster+1):
			print('#################################')
			vocab = data.load('./result/vocab'+str(i)+'.pickle')
			fvocab = data.load('./result/fvocab'+str(i)+'.pickle')
			embeddings = data.load('./result/embedding_size'+str(args.dim)+'_'+str(i)+'.pickle')

			# Load the Train/Test Data
			
			vocab_inv = {v: k for k, v in vocab.items()}
			n_labels = 2
			batch_size = 30
			total_acc = 0.0
			total_rule_acc = 0.0

			print('len(vocab):', len(vocab))
			if args.gpu >= 0:
				xp = cupy
			else:
				xp = np


			# # kf = KFold(n_splits=10, shuffle=True, random_state=53)
			# # for i, (train_index, test_index) in enumerate(kf.split(x, y)):
			# # 	print('i:', str(i), "TRAIN:", train_index, "TEST:", test_index)
			# # 	x_train, x_test = x[train_index], x[test_index]
			# # 	y_train, y_test = y[train_index], y[test_index]

			# # trainとtestの0/1分布確認
			# print('y_train:', y_train.tolist().count(0), y_train.tolist().count(1), y_train.tolist().count(1)/len(y_train.tolist()))
			# print('y_test:', y_test.tolist().count(0), y_test.tolist().count(1), y_test.tolist().count(1)/len(y_test.tolist()))
			# rule_acc = y_test.tolist().count(1)/len(y_test.tolist())
			# if rule_acc < 0.5:
			# 	rule_acc = 1 - rule_acc
				# print('rule_base:', y_test.tolist().count(1)/len(y_test.tolist()))
				# total_rule_acc += y_test.tolist().count(1)/len(y_test.tolist())

			# x = xp.array(x, dtype=xp.int32)
			# y = xp.array(y, dtype=xp.int32)

			train_data = tuple_dataset.TupleDataset(x_train, y_train)
			test_data = tuple_dataset.TupleDataset(x_test, y_test)
			train_iter = iterators.SerialIterator(train_data, batch_size, shuffle=True)
			test_iter = iterators.SerialIterator(test_data, batch_size, repeat=False, shuffle=False)

			# Set up a model
			if args.type == 'ffnn':
				nn = L.Classifier(model.FFNN(len(vocab), args.dim, len(fvocab), args.units, n_labels, args.w2v, embeddings))
			elif args.type == 'bilstm':
				nn = L.Classifier(model.BiLSTM(len(vocab), args.dim, len(fvocab), args.units, n_labels, args.w2v, embeddings))
			if args.gpu >= 0:
				nn.to_gpu()

			# Set up an optimizer
			optimizer = optimizers.Adam(alpha=0.01, beta1=0.9, beta2=0.999, eps=1e-6)
			optimizer.setup(nn)
			optimizer.use_cleargrads()
			updater = training.StandardUpdater(train_iter, optimizer)

			# Set up a trainer
			trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='result')
			# trainer.extend(extensions.Evaluator(test_iter, nn))
			# trainer.extend(extensions.dump_graph('main/loss'))
			trainer.extend(extensions.LogReport())
			# trainer.extend(extensions.PrintReport(	['epoch', 'main/loss', 'validation/main/loss',
												     # 'main/accuracy', 'validation/main/accuracy']))
			trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy']))
			trainer.extend(extensions.ProgressBar())
			# trainer.extend(extensions.snapshot_object(nn, 'model_'+args.type+'_size'+str(args.size)+'_w2v'+str(args.w2v)+'epoch_'+'{.updater.epoch}'))
			# trainer.extend(extensions.snapshot_object(optimizer, 'optimizer_'+args.type+'_size'+str(args.size)+'_w2v'+str(args.w2v)+'epoch_'+'{.updater.epoch}'))
			trainer.run()

			###### Evaluate the final model ######
			print('--------------------------')
			print('test')
			eval_nn = nn.copy()
			eval_predictor = eval_nn.predictor
			eval_predictor.train = False

			# Reset the RNN state at the beginning of each evaluation
			# if args.type in ['gru', 'bilstm']:
			# 	eval_predictor.reset_state()		# BLSTMBaseのreset_state 

			evaluator = extensions.Evaluator(test_iter, eval_nn, device=args.gpu)
			result = evaluator()
			test_acc = result['main/accuracy']
			print('rule_acc', rule_acc)
			print('validation/main/accuracy, validation/main/loss:', result['main/accuracy'], result['main/loss'])
			print('--------------------------')

			# Serialize the final model
			modelpath = './result/'+'model_'+args.type+'_dim'+str(args.dim)+'_i'+str(i)+'.npz'
			chainer.serializers.save_npz(modelpath, nn)

				# total_acc += result['main/accuracy']
				# respath = './result/model_'+args.type+'_w2v'+str(args.w2v)+'_mark'+str(args.mark)+'.txt'
				# writeResult(respath, 'i:'+str(i)+','+str(result['main/accuracy']))

			# writeResult(respath, 'mean_accuracy'+str(total_acc / 10.))
			writeResult('./result/result_acc.txt', 'rule: '+str(i)+'_'+str(rule_acc))
			writeResult('./result/result_acc.txt', 'ffnn: '+str(i)+'_'+str(test_acc))

		# elapsed_time = time.time() - start
		# print("elapsed_time:{0}".format(elapsed_time) + "[sec]")



