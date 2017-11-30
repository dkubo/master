import argparse, json, copy
from chainer import serializers
import data, model
import numpy as np
import cupy, chainer
from chainer import links as L
from chainer import functions as F
from chainer import iterators, optimizers, training, configuration
import MeCab
from trie import Trie

def get_arg():
	parser = argparse.ArgumentParser(description='converter')
	parser.add_argument('-gpu','--gpu', type=int, default=0)
	parser.add_argument('-epoch','--epoch', type=int, default=50)
	parser.add_argument('-posget','--posget', type=int, default=1)	# pos取得するかどうか
	parser.add_argument('-type','--type', type=str, default='ffnn')
	parser.add_argument('-dim','--dim', type=int, default=100)
	parser.add_argument('-w2v','--w2v', type=int, default=1)
	parser.add_argument('-mark','--mark', type=int, default=1)
	parser.add_argument('-units','--units', type=int, default=50)
	parser.add_argument('-size','--size', type=int, default=2)	# 前後形態素取得数(2=>前後それぞれ2取得)
	args = parser.parse_args()
	return args

def loadModel(args, vocab, fvocab, embeddings, i):
	n_labels = 2
	if args.type == 'ffnn':
		nn = L.Classifier(model.FFNN(len(vocab), args.dim, len(fvocab), args.units, n_labels, args.w2v, embeddings))
	elif args.type == 'bilstm':
		nn = L.Classifier(model.BiLSTM(len(vocab), args.dim, len(fvocab), args.units, n_labels, args.w2v, embeddings))
	nn.to_gpu()
	modelpath = './model/'+'model_'+args.type+'_size'+str(args.size)+'_w2v'+str(args.w2v)+'_dim'+str(args.dim)+'_kfold'+str(i)+'.npz'
	serializers.load_npz(modelpath, nn)
	return nn

def loadDict():
	mwelist = data.load('./tmp/mwelist.pickle')	# 日本語複合辞用例DBで、曖昧性がある機能表現のリスト
	mwedict = []
	for mwe in mwelist:
		mwe = mwe.split(',')
		for partlist in mwe:
			partlist = partlist.split('[')
			partlist[1] = partlist[1].replace(']', '')

			for part in partlist:
				m = MeCab.Tagger()
				parse = m.parse(part).split('\n')[0:-2]
				morphlist = []
				for morph in parse:
					morph = morph.split('\t')[0]
					morphlist.append(morph)

				if morphlist not in mwedict:
					mwedict.append(morphlist)

	# print(len(mwedict))	# 86
	return mwedict

def getSpan(sentlist, mwe, idxspan):
	m_frg = 0
	lenmwe = len(mwe)
	for i, sent in enumerate(sentlist):
		if sent == mwe[0]:
			startidx = i
			if sentlist[i:i+lenmwe] == mwe:
				idxspan.append([i, i+lenmwe-1])
	return idxspan

##################################
def removeHogan(idxspan):
	newspan = copy.deepcopy(idxspan)
	if len(idxspan) >= 2:
		for i in range(0, len(idxspan)-1):
			for j in range(i+1, len(idxspan)):
				if idxspan[i][0] == idxspan[j][0]:	# 開始idxを比較
					hogan_sidx = idxspan[i][0]
					remove_eidx = min(idxspan[i][1], idxspan[j][1])
					newspan.remove([hogan_sidx, remove_eidx])
	return newspan

# def removeIreko(idxspan):
# 	newspan = copy.deepcopy(idxspan)
# 	if len(idxspan) >= 2:
# 		for i in range(0, len(idxspan)-1):
# 			for j in range(i+1, len(idxspan)):
# 				if idxspan[i][1] >= idxspan[j][0]:	# 開始idxを比較
# 					# print(idxspan[i][1], idxspan[j][0])
# 	return newspan
##################################

def mark_feature(n_pre, n_mwe, n_post):
	mark = []
	for i in range(0, n_pre):
		mark.append(0)
	for i in range(0, n_mwe):
		mark.append(1)
	for i in range(0, n_post):
		mark.append(0)
	return mark

def morphParse(sentence, mwedict, getsize=2):
	sentlist, poslist, katulist = [], [], []
	spanlist, mark = [], []
	m = MeCab.Tagger()
	parse = m.parse(sentence).split('\n')[0:-2]
	for i, morph in enumerate(parse):
		morph = morph.split('\t')
		pos = '-'.join(morph[4].split("-")[0:2])	# 中分類まで抽出
		katuyokei = '-'.join(morph[6].split("-")[0:1])	# 大分類

		sentlist.append(morph[0])		# 出現系
		poslist.append(pos)
		katulist.append(katuyokei)

	return sentlist, poslist, katulist

def getInput(sentlist, poslist, katulist, idxspan):
	x, getsize = [], 2
	for span in idxspan:
		sent, feature, mark = [], [], []
		if span[0] < getsize:
			pre = sentlist[0:span[0]]
			funcmwe = sentlist[span[0]:span[1]+1]
			post = sentlist[span[1]+1:span[1]+(getsize+1)]

			sent = pre + funcmwe + post
			mark = mark_feature(len(pre), len(funcmwe), len(post))
			feature = poslist[0:span[1]+(getsize+1)]+ \
							katulist[0:span[1]+(getsize+1)]

		elif span[0] >= getsize:
			pre = sentlist[span[0]-getsize:span[0]]
			funcmwe = sentlist[span[0]:span[1]+1]
			post = sentlist[span[1]+1:span[1]+(getsize+1)]

			sent = pre + funcmwe + post
			mark = mark_feature(len(pre), len(funcmwe), len(post))
			feature = poslist[span[0]-getsize:span[1]+(getsize+1)]+ \
							katulist[span[0]-getsize:span[1]+(getsize+1)]

		x.append([sent, feature, mark])

	return x

def get_wid(vocab, words):
	idlist = []
	for word in words:
		if word in vocab:
			idlist.append(vocab[word])
		else:	# 未知語用のvocab_idを用意する必要あり？
			print(word)
	return idlist

def padding(data, maxlen):
	for i in range(0, maxlen-len(data)):
		data.append(-1)
	return data

def conv_wid(x, vocab, fvocab, maxlen):
	newx = []
	for sentlist, featlist, marklist in x:
		sentlist = get_wid(vocab, sentlist)
		featlist = get_wid(fvocab, featlist)
		sentlist = padding(sentlist, maxlen)
		featlist = padding(featlist, maxlen)
		marklist = padding(marklist, maxlen)
		newx.append([sentlist, featlist, marklist])
	return newx

def parse(nn, sentence, mwedict):
	idxspan = []
	sentlist, poslist, katulist = morphParse(sentence, mwedict, getsize=2)
	for mwe in mwedict:
		idxspan = getSpan(sentlist, mwe, idxspan)
	idxspan.sort()
	idxspan = removeHogan(idxspan)		# 包含: 長いスパンの方を採用

	# idxspan = removeIreko(idxspan)	# 入れ子: どちらの表現も用法判定を行う→どちらも機能表現だった場合、ひとまとめにする？

	x = getInput(sentlist, poslist, katulist, idxspan)
	### maxlen: # モデル学習時の最大系列数16に合わせる必要あり(機能表現の最大長さ3+前後2=8, 8*2(品詞と活用形)=16) ###
	x = conv_wid(x, vocab, fvocab, maxlen=16)
	# print(x, type(x))
	x = cupy.array(x, dtype=cupy.int32)
	# print(x, type(x))

	return nn.predictor(x), sentlist, idxspan

def concat(sentlist, pred_idx, idxspan):
	# 1のindex(=機能的用法と判断)を取り出す
	funcidx = [idxspan[i] for i, v in enumerate(pred_idx) if v == 1]
	# funcidx.sort()	# 既にidxspanがsortされてるからno need
	redu_idx = 0
	for idx in funcidx:
		mwe = ''.join(sentlist[idx[0]-redu_idx:idx[1]+1-redu_idx])
		sentlist.insert(idx[0]-redu_idx, mwe)
		del sentlist[idx[0]+1-redu_idx:idx[1]+2-redu_idx]
		redu_idx += idx[1] - idx[0]	# (idx[1]+1) - idx[0] -1
	return sentlist

if __name__ == '__main__':
	args = get_arg()
	vocab = data.load('./tmp/vocab'+str(args.size)+'.pickle')
	fvocab = data.load('./tmp/fvocab'+str(args.size)+'.pickle')
	embeddings = data.load('./tmp/initialW_'+str(args.dim)+'_size'+str(args.size)+'.pickle')
	chainer.using_config('use_cudnn', True)
	nn = loadModel(args, vocab, fvocab, embeddings, i=0)

	# mwedict = loadDict()	
			# mwelist = [[に,つい,て](出現系の形態素リスト), [に,あたり], ...]
	# data.save('./tmp/mwedict.pickle', mwedict)
	mwedict = data.load('./tmp/mwedict.pickle')
	sentence = 'その政府の代わりに応じた環境の上での問題が生じた。'	# mwe = [[代わり, に], [に, 応じ, た], [上, で], [上, で, の]]
	print('input: ', sentence)
	predict, sentlist, idxspan = parse(nn, sentence, mwedict)
	pred_idx = np.argmax(predict.data, axis=1)
	sentlist = concat(sentlist, pred_idx, idxspan)
	print('output: ', sentlist)



