# coding: utf-8

import numpy as np
import argparse
from collections import defaultdict
import model, data
import cupy, chainer
from chainer import links as L
from chainer import functions as F

###############
#	引数処理
###############
def get_arg():
	parser = argparse.ArgumentParser(description='converter')
	parser.add_argument('-gpu','--gpu', type=int, default=0)
	parser.add_argument('-dim','--dim', type=int, default=100)
	parser.add_argument('-units','--units', type=int, default=300)
	args = parser.parse_args()
	return args

##################
#	test
##################
def error_anlz(nn, x_test, y_test, vocab_inv, fvocab_inv):
	mwe_cnt = defaultdict(lambda:defaultdict(list))
	for x, y in zip(x_test, y_test):
		x = cupy.array([x], dtype=cupy.int32)
		y = cupy.array([y], dtype=cupy.int32)
		predict = nn.predictor(x)
		acc = F.accuracy(predict, y)

		### mark素性からMWEの位置を取得できそう ###
		morph = chainer.cuda.to_cpu(x[0][0])
		feature = chainer.cuda.to_cpu(x[0][1])
		mark = chainer.cuda.to_cpu(x[0][2])
		mweindex = np.where(mark == 1)
		mwe = ''.join([vocab_inv[wordid] for wordid in morph[mweindex]])

		if acc.data == 1.0:
			mwe_cnt[mwe]['correct'].append([morph, feature])
		else:
			mwe_cnt[mwe]['false'].append([morph, feature])

	return mwe_cnt


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
	chainer.cuda.get_device(args.gpu).use()
	n_cluster = 4
	for i in range(1, n_cluster+1):
		print('#################################')
		print(i)
		x_test = data.load('./result/x_test_'+str(i)+'_amb.pickle')
		y_test = data.load('./result/y_test_'+str(i)+'_amb.pickle')
		embeddings = data.load('./result/embedding_'+str(i)+'_amb.pickle')
		vocab = data.load('./result/vocab_'+str(i)+'_amb.pickle')
		fvocab = data.load('./result/fvocab_'+str(i)+'_amb.pickle')

		vocab_inv = {v: k for k, v in vocab.items()}
		fvocab_inv = {v: k for k, v in fvocab.items()}

		# load the model
		nn = L.Classifier(model.BiLSTM(len(vocab), args.dim, len(fvocab), args.units, 2, 1, embeddings))

		modelpath = './result/model_bilstm_dim'+str(args.dim)+'_i'+str(i)+'.npz'
		chainer.serializers.load_npz(modelpath, nn)
		nn.to_gpu()

		mwe_cnt = error_anlz(nn, x_test, y_test, vocab_inv, fvocab_inv)

		print(mwe_cnt)
		# calc(mwe_cnt)










