# coding: utf-8

import numpy as np
import argparse, time
import data, model
import cupy
import chainer
from chainer import links as L
from chainer import iterators, optimizers
from chainer.training import extensions
from chainer.datasets import tuple_dataset

# from sklearn.model_selection import KFold


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
	parser.add_argument('-units','--units', type=int, default=300)	# FFNN: 9000 → 300
	####################
	parser.add_argument('-cluster','--cluster', type=int, default=1)
	####################
	args = parser.parse_args()
	return args

def writeResult(fpath, sent):
	with open(fpath, 'a') as f:
		f.write(sent+'\n')


def proc(modelname, x_test, y_test, embeddings, vocab, fvocab):
	# Set up a model
	if args.type == 'ffnn':
		nn = L.Classifier(model.FFNN(len(vocab), args.dim, len(fvocab), args.units, n_labels, args.w2v, embeddings))
	elif args.type == 'bilstm':
		nn = L.Classifier(model.BiLSTM(len(vocab), args.dim, len(fvocab), args.units, n_labels, args.w2v, embeddings))

	chainer.cuda.get_device(args.gpu).use()
	nn.to_gpu()

	# load the model and optimizer
	fname = './result/model_'+args.modeltype+'_size'+str(args.size)+'_w2v'+str(args.w2v)+'epoch_'+str(args.epoch)
	print('modelname: ', modelname)
	chainer.serializers.load_npz(modelname, nn)

	###### Evaluate the final model ######
	predict = nn.predictor(x_test)
	acc = F.accuracy(predict, y_test)
	print('predict, acc: ', predict, acc)

	return acc

def getData(i):
		x_test = load('./result/bccwj_x_test_'+str(i)+'.pickle')
		y_test = load('./result/bccwj_y_test_'+str(i)+'.pickle')
		embeddings = load('./result/embedding_'+str(i)+'_amb.pickle')
		vocab = load('../wv_cluster/result/vocab_'+str(i)+'_amb.pickle')
		fvocab = load('../wv_cluster/result/fvocab_'+str(i)+'_amb.pickle')
	return x_test, y_test, embeddings, vocab, fvocab

def calcRuleAcc(y_test):
	rule_acc = y_test.tolist().count(1)/len(y_test.tolist())
	if rule_acc < 0.5:
		rule_acc = 1 - rule_acc
	return rule_acc

if __name__ == '__main__':
	start = time.time()
	args = get_arg()
	n_labels = 2
	chainer.using_config('use_cudnn', True)

	print('args.gpu:',args.gpu)
	print('modeltype:',args.type)
	xp = cupy

	###########################
	# クラスタリングなし
	###########################
	if args.cluster == 0:
		i = 'all'
		# modelpath = './result/'+'model_'+args.type+'_dim'+str(args.dim)+'_i'+str(i)+'.npz'
		modelpath = './result/'+'wikigold_model_'+args.type+'_dim'+str(args.dim)+'_i'+str(i)+'.npz'
		x_test, y_test, embeddings, vocab, fvocab = getData(i)
		# trainとtestの0/1分布確認
		print('y_test:', y_test.tolist().count(0), y_test.tolist().count(1), y_test.tolist().count(1)/len(y_test.tolist()))
		rule_acc = calcRuleAcc(y_test)
		test_acc, nn = proc(x_test, y_test, embeddings, vocab, fvocab)
		chainer.serializers.save_npz(modelpath, nn)

		print('rule_accuracy', rule_acc)
		print('test_accuracy', test_acc)

	###########################
	# クラスタリングあり
	###########################
	elif args.cluster == 1:
		total_acc, total_rule_acc, n_cluster = 0.0, 0.0, 4
		args = get_arg()
		for i in range(1, n_cluster+1):
			if i != 2:
				print('#################################')
				print(i)
				x_test, y_test, embeddings, vocab, fvocab = getData(i)
				# # trainとtestの0/1分布確認
				print('y_test:', y_test.tolist().count(0), y_test.tolist().count(1), y_test.tolist().count(1)/len(y_test.tolist()))
				rule_acc = calcRuleAcc(y_test)
				total_rule_acc += rule_acc
				test_acc, nn = proc(x_test, y_test, embeddings, vocab, fvocab)
				total_acc += test_acc
				print('rule_acc', rule_acc)
				print('test_accuracy', test_acc)
				# modelpath = './result/'+'model_'+args.type+'_dim'+str(args.dim)+'_i'+str(i)+'.npz'
				modelpath = './result/'+'wikigold_model_'+args.type+'_dim'+str(args.dim)+'_i'+str(i)+'.npz'
				chainer.serializers.save_npz(modelpath, nn)

		# print('rule_mean_accuracy', total_rule_acc / n_cluster)
		# print('mean_accuracy', total_acc / n_cluster)
		print('rule_mean_accuracy', total_rule_acc / 3)
		print('mean_accuracy', total_acc / 3)


