# coding: utf-8

import numpy as np
import argparse
from collections import defaultdict
import data, model
import cupy, chainer
from chainer import links as L
from chainer import functions as F
from chainer import iterators, optimizers, training, configuration
from chainer.training import extensions

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from chainer.datasets import tuple_dataset

###############
#	引数処理
###############
def get_arg():
	parser = argparse.ArgumentParser(description='converter')
	parser.add_argument('-gpu','--gpu', type=int, default=0)
	parser.add_argument('-epoch','--epoch', type=str, default=100)
	parser.add_argument('-modeltype','--modeltype', type=str, default=None)
	parser.add_argument('-posget','--posget', type=int, default=None)	# pos取得するかどうか
	parser.add_argument('-size','--size', type=int, default=None)
	parser.add_argument('-dim','--dim', type=int, default=None)
	parser.add_argument('-units','--units', type=int, default=None)
	parser.add_argument('-w2v','--w2v', type=int, default=None)
	args = parser.parse_args()
	return args

##################
#	test
##################
def error_anlz(nn, data_list, vocab_inv, fvocab_inv, senthash, reshash):
	total_acc, cnt = 0.0, 0
	y_pred, y_true = [], []
	mwe_cnt = defaultdict(lambda:defaultdict(list))
	for sentence in data_list:
		x, y = sentence
		# words = x.tolist()[0]+x.tolist()[1]

		x_0 = [int(item) for item in x[0] if int(item) is not -1]	# -1 除去
		x_1 = [int(item) for item in x[1] if int(item) is not -1]	# -1 除去

		key_0 = tuple([vocab_inv[word] for word in x_0])
		key_1 = tuple([fvocab_inv[word] for word in x_1])
		# print(key_0, key_1)
		yorei, mwe = senthash[reshash[key_0]]
		if mwe == 'をとわずに[をとわずに],を問わずに[をとわずに]':
			print('-----------------')
			print('yorei:', yorei)
			print('label: ', y)
			x = xp.array([x], dtype=xp.int32)
			y = xp.array([y], dtype=xp.int32)
			predict = nn.predictor(x)
			acc = F.accuracy(predict, y)
			print('predict, acc: ', predict, acc)
			# y_pred.append(predict.data[0])
			# y_true.append(y[0])

		# if acc.data == 1.0:
		# 	mwe_cnt[mwe]['correct'].append([key, sentence])
		# else:
		# 	mwe_cnt[mwe]['false'].append([key, sentence])


	# print(len(y_true))		# 185
	# print(y_true.count(0))	# 56
	# print(y_true.count(1))	# 129
	# y_pred = xp.array(y_pred, dtype=xp.float32)
	# y_true = xp.array(y_true, dtype=xp.int32)
	# print(len(y_pred))
	# print(len(y_true))
	# print(F.classification_summary(y_pred, y_true, label_num=2))	# (0を当てると考えたとき, 1を当てると考えたとき)

	# calc(mwe_cnt)


def calc(mwe_cnt):
	for mwe, cnt in mwe_cnt.items():
		print('-------------------')
		cor = len(cnt['correct'])
		fal = len(cnt['false'])
		acc = cor / (cor + fal)
		print(mwe ,'= ', cor, '/', (cor+fal), '=',acc)

if __name__ == '__main__':
	chainer.using_config('use_cudnn', True)
	args = get_arg()
	if args.posget == 0:
		x = data.load('./tmp/sentlist'+str(args.size)+'.pickle')
	elif args.posget == 1:
		x = []
		sentlist = data.load('./tmp/sentlist'+str(args.size)+'.pickle')
		featlist = data.load('./tmp/featlist'+str(args.size)+'.pickle')
		fmaxlen = len(featlist[0])
		sentlist = data.padding(sentlist, fmaxlen)
		for sent, feat in zip(sentlist, featlist):
			x.append([sent, feat])


	senthash = data.load('./tmp/senthash'+str(args.size)+'.pickle')
	reshash = data.load('./tmp/reshash'+str(args.size)+'.pickle')


	y = data.load('./tmp/labels'+str(args.size)+'.pickle')
	vocab = data.load('./tmp/vocab'+str(args.size)+'.pickle')
	vocab_inv = {v: k for k, v in vocab.items()}
	vocab_inv[-1] = -1
	fvocab = data.load('./tmp/fvocab'+str(args.size)+'.pickle')
	fvocab_inv = {v: k for k, v in fvocab.items()}
	fvocab_inv[-1] = -1
	embeddings = data.load('./tmp/initialW_'+str(args.dim)+'_size'+str(args.size)+'.pickle')
	n_labels = 2

	print('gpu:',args.gpu)
	print('modeltype:',args.modeltype)
	print('len(vocab):', len(vocab))	# 23229(出現形)→18935(標準形)→7940(54表現&標準形)

	if args.modeltype == 'ffnn':
		modelpath = "./model/ffnn.model"
		# model = L.Classifier(model.FFNN(len(vocab), 100, 50, 2))	# これだとlossが減少しなかった(accは上がる)
		# model = L.Classifier(model.FFNN(len(vocab), 200, 10, 2))
		nn = L.Classifier(model.FFNN(len(vocab), args.dim, len(fvocab), args.units, n_labels, args.w2v, embeddings))

	elif args.modeltype == 'cnn':
		modelpath = "./model/cnn.model"
		fil_num = 1
		nn = L.Classifier(model.CNN(fil_num, n_labels, embeddings))

	elif args.modeltype == 'bilstm':
		modelpath = "./model/bilstm.model"
		# nn = L.Classifier(model.BLSTMBase(len(vocab), args.dim, len(fvocab), args.units, n_labels, args.w2v, embeddings))
		nn = L.Classifier(model.BiLSTM(len(vocab), args.dim, len(fvocab), args.units, n_labels, args.w2v, embeddings))

	nn.to_gpu()
	xp = cupy

	# load the model and optimizer
	fname = './result/model_'+args.modeltype+'_size'+str(args.size)+'_w2v'+str(args.w2v)+'epoch_'+str(args.epoch)
	print(fname)
	chainer.serializers.load_npz(fname, nn)

	x = xp.array(x, dtype=xp.int32)
	y = xp.array(y, dtype=xp.int32)
	x_train, x_test, y_train, y_test = data.dataSplit(x, y)
	test_data = tuple_dataset.TupleDataset(x_test, y_test)
	# test_iter = iterators.SerialIterator(test_data, batch_size, repeat=False, shuffle=False)
	# test_iter = iterators.SerialIterator(test_data, batch_size=1, repeat=False, shuffle=False)

	error_anlz(nn, test_data, vocab_inv, fvocab_inv, senthash, reshash)

	###### Evaluate the final model ######
	# if args.modeltype == 'bilstm':
	# 	eval_lstm = nn.predictor

	# evaluator = extensions.Evaluator(test_iter, nn, device=args.gpu)

	# with configuration.using_config('train', False):
	# 	iterator = evaluator._iterators['main']
	# 	target = evaluator._targets['main']
	# 	eval_func = evaluator.eval_func or target
	# 	if hasattr(iterator, 'reset'):
	# 		iterator.reset()
	# 		it = iterator
	# 	else:
	# 		it = copy.copy(iterator)

	# 	for batch in it:
	# 		in_arrays = evaluator.converter(batch, evaluator.device)
	# 		print(in_arrays)
	# 		# with chainer.no_backprop_mode():
	# 		# 	if isinstance(in_arrays, tuple):
	# 		# 		eval_func(*in_arrays)
	# 		# 	elif isinstance(in_arrays, dict):
	# 		# 		eval_func(**in_arrays)
	# 		# 	else:
	# 		# 		eval_func(in_arrays)


	# # result = evaluator()
	# # print('validation/main/accuracy, main/loss:', result['main/accuracy'], result['main/loss'])

