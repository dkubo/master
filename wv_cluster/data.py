# coding: utf-8

import xml.etree.ElementTree as ET
import glob, sys, re, dill, argparse, collections, random
import pandas as pd
import numpy as np
import cupy
import MeCab
import json
from sklearn.cross_validation import train_test_split

###############
#	引数処理
###############
def get_arg():
	parser = argparse.ArgumentParser(description='converter')
	parser.add_argument('-size','--size', type=int, default=2)	# 前後形態素取得数(4=>前後それぞれ4取得)
	parser.add_argument('-dim','--dim', type=int, default=100)
	parser.add_argument('-n_cluster','--n_cluster', type=int, default=4)	# 4で固定
	# parser.add_argument('-cluster','--cluster', type=int, default=1)
	args = parser.parse_args()
	return args

#####################
#		 Pickle
#####################
def save(fname, cont):
	with open(fname, mode='wb') as f:
		dill.dump(cont, f)

def load(fname):
	with open(fname, mode='rb') as f:
		return dill.load(f)

def mark_feature(n_pre, n_mwe, n_post):
	mark = []
	for i in range(0, n_pre):
		mark.append(0)
	for i in range(0, n_mwe):
		mark.append(1)
	for i in range(0, n_post):
		mark.append(0)
	return mark

#####################
#		 形態素解析
#####################
def morphParse(text, span, getsize):
	sentence, sent, feature, poslist, katulist, totallen, idxspan = [], [], [], [], [], 0, []
	mark = []
	m = MeCab.Tagger()
	parse = m.parse(text).split('\n')[0:-2]

	# print(len(parse))
	for i, morph in enumerate(parse):
		morph = morph.split('\t')
		if totallen == int(span[0]):
			idxspan.append(i)

		### pos ###
		# pos = '-'.join(morph[4].split("-")[0:1])	# 大分類
		pos = '-'.join(morph[4].split("-")[0:2])	# 中分類まで抽出
		# pos = '-'.join(morph[4].split("-")[0:3])	# 小分類まで抽出

		### 活用形 ###
		katuyokei = '-'.join(morph[6].split("-")[0:1])	# 大分類
		# print(katuyokei)

		totallen += len(morph[0])	# 出現形
		if totallen == int(span[1]):
			idxspan.append(i)

		# ※※※ 出現形:morph[0], 標準形: morph[3], 活用型: morph[5], 活用形: morph[6] ※※※
		sentence.append(morph[0])		# 出現系
		# sentence.append(morph[3].split('-')[0])	# 標準形
		poslist.append(pos)
		katulist.append(katuyokei)

	if len(idxspan) == 2:
		# 機能表現の前後形態素を参照
		if idxspan[0] < getsize:
			pre = sentence[0:idxspan[0]]
			funcmwe = sentence[idxspan[0]:idxspan[1]+1]
			post = sentence[idxspan[1]+1:idxspan[1]+(getsize+1)]

			sent = pre + funcmwe + post
			mark = mark_feature(len(pre), len(funcmwe), len(post))
			feature = poslist[0:idxspan[1]+(getsize+1)]+ \
							katulist[0:idxspan[1]+(getsize+1)]

		elif idxspan[0] >= getsize:
			pre = sentence[idxspan[0]-getsize:idxspan[0]]
			funcmwe = sentence[idxspan[0]:idxspan[1]+1]
			post = sentence[idxspan[1]+1:idxspan[1]+(getsize+1)]

			sent = pre + funcmwe + post
			mark = mark_feature(len(pre), len(funcmwe), len(post))
			feature = poslist[idxspan[0]-getsize:idxspan[1]+(getsize+1)]+ \
							katulist[idxspan[0]-getsize:idxspan[1]+(getsize+1)]
	return sent, feature, mark

#####################
#	単語id割り当て
#####################
def get_wid(vocab, words):
	return [vocab[word] for word in words]

#####################
# 	パディング	
#####################
def padding(data, maxlen):
	newdata = []
	for sent in data:
		for i in range(0, maxlen-len(sent)):
			sent.append(-1)
		newdata.append(sent)
	return newdata

def getMaxlen(data):
	maxlen = 0
	for i in data:
		maxlen = max(len(i), maxlen)
	return maxlen

###############
#	w2v
###############
def trainw2v(modelpath, dim):
	from gensim.models import word2vec
	wakati = "../../../data/jawiki_wakati.txt"
	# Load the corpus 
	data = word2vec.Text8Corpus(wakati)
	# Train
	model = word2vec.Word2Vec(data, size=dim, window=5, min_count=5, workers=15)
	return model

def embedding(vocab_inv, w2vmodel, dim):
	# embedIDの初期重み作成
	initialW = []
	for i in range(0, len(vocab)):
		word = vocab_inv[i]		# wid=1から順にword取得
		try:
			w2vec = w2vmodel[word]
		except:
			w2vec = np.random.normal(scale=1.0, size=dim)
		initialW.append(w2vec)

	initialW = cupy.array(initialW, dtype=cupy.float32)
	return initialW

#####################
# 	用例DBのXMLパーズ
#####################
# def parseDB(datapath, vocab, fvocab, getsize):
def parseDB(datapath, getsize, mwelist, i):
	vocab = collections.defaultdict(lambda: len(vocab))	# 語彙用
	fvocab = collections.defaultdict(lambda: len(fvocab)) # feature用
	sidhash = {}

	for file in glob.iglob(datapath + '*.xml'):
		tree = ET.parse(file)
		root = tree.getroot()
		mwe = root.attrib["name"]
		freq = root.attrib["freq"]
		freq_total = root.attrib["total"]

		# print('mwe:', mwe)
		if mwe in mwelist:	# 用法に曖昧性がある場合
			for elm in root.findall(".//example"):
				sentence = elm.text
				sentid = elm.attrib["id"]
				label = elm.attrib["label"]
				span = elm.attrib["target"].split('-')
				sent, feature, mark = morphParse(sentence, span, getsize)
				if len(sent) != 0:
					if label in ['F', 'A', 'M']:
						label = 1
					else:
						label = 0
					sidhash[sentid] = {'sent':sent, 'feat':feature, 'mark': mark, 'label':label}
	if i != 'all':
		train_sid, test_sid = clustSidSplit(sidhash)
		x_train, y_train, vocab, fvocab = makeTrainTest(train_sid, sidhash, vocab, fvocab)
		x_test, y_test, vocab, fvocab = makeTrainTest(test_sid, sidhash, vocab, fvocab)

	else:
		test_sid = []
		# 各クラスタのテストセットの文id抽出
		for i in range(1, n_cluster+1):
			test_sid.append(load('./result/test_sid_'+str(i)+'.pickle'))	

		train_sid = list(set(sidhash.keys()) - set(test_sid))
		x_train, y_train, vocab, fvocab = makeTrainTest(train_sid, sidhash, vocab, fvocab)
		x_test, y_test, vocab, fvocab = makeTrainTest(test_sid, sidhash, vocab, fvocab)

	save('./result/vocab_'+str(i)+'.pickle', vocab)
	save('./result/fvocab_'+str(i)+'.pickle', fvocab)
	save('./result/x_train_'+str(i)+'.pickle', x_train)
	save('./result/x_test_'+str(i)+'.pickle', x_test)
	save('./result/y_train_'+str(i)+'.pickle', y_train)
	save('./result/y_test_'+str(i)+'.pickle', y_test)
	save('./result/test_sid_'+str(i)+'.pickle', test_sid)

	return vocab

def makeTrainTest(sidlist, sidhash, vocab, fvocab):
	sentlist, featlist, marklist, x, y = [], [], [], [], []
	for sentid in sidlist:
		sentlist.append((get_wid(vocab, sidhash[sentid]['sent'])))
		featlist.append((get_wid(fvocab, sidhash[sentid]['feat'])))
		marklist.append(sidhash[sentid]['mark'])
		y.append(sidhash[sentid]['label'])

	get_wid(vocab, ['unk'])		# 未知語用の語彙用意
	get_wid(fvocab, ['unk'])	# 未知語用の語彙用意

	sentlist = padding(sentlist, getMaxlen(featlist))
	featlist = padding(featlist, getMaxlen(featlist))
	marklist = padding(marklist, getMaxlen(featlist))
	for sent, feat, mark in zip(sentlist, featlist, marklist):
		x.append([sent, feat, mark])

	 return cupy.array(x, dtype=cupy.int32), cupy.array(y, dtype=cupy.int32), vocab, fvocab

def clustSidSplit(sidhash):
	return train_test_split(list(sidhash.keys()), test_size=0.1, random_state=53)

########################
#	全表現をクラスタリングした各クラスタから、曖昧性のある表現抽出する
#	積集合を求める
########################
def extractAmb(amb_mwelist, clust_mwelist):
	return list(set(amb_mwelist) & set(clust_mwelist))

#####################
#		main
#####################
if __name__ == '__main__':
	args = get_arg()
	print('getsize:', args.size)
	print('n_cluster:', args.n_cluster)
	xp = cupy

	datapath = '../../../data/MUST-dist-1.0/data/'
	mpath = '../nn/model/wiki_vector'+str(args.dim)+'.model'
	from gensim.models import word2vec
	w2vmodel = word2vec.Word2Vec.load(mpath)	# load
	amb_mwelist = load('../nn/tmp/mwelist.pickle')
	# # # vocab['unk']: 未知語用の語彙

	# # クラスタ毎に回す(1〜args.n_cluster)
	for i in range(1, args.n_cluster+1):
		print(i)
		clust_mwelist = load('./result/clusterlist_'+str(i)+'.pickle')
		mwelist = extractAmb(amb_mwelist, clust_mwelist)
		# print(len(clust_mwelist), len(amb_mwelist), len(mwelist))
		vocab = parseDB(datapath, args.size, mwelist, i)
		vocab_inv = {v: k for k, v in vocab.items()}
		initialW = embedding(vocab_inv, w2vmodel, args.dim)
		save('./result/embedding_size'+str(args.dim)+'_'+str(i)+'.pickle', initialW)

	# all
	mwelist = amb_mwelist
	vocab = parseDB(datapath, args.size, mwelist, 'all')
	vocab_inv = {v: k for k, v in vocab.items()}
	initialW = embedding(vocab_inv, w2vmodel, args.dim)
	save('./result/embedding_size'+str(args.dim)+'_all.pickle', initialW)



