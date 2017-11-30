# coding: utf-8

import xml.etree.ElementTree as ET
import glob, sys, re, dill, argparse, collections, random
import pandas as pd
import numpy as np
import MeCab
import json
from trie import Trie


### mecabのuserdicの追加方法 ###
# /home/is/daiki-ku/usr/etc/mecabrc を編集
#### 一語の機能表現は除去(Bタグだけのやつ) ####

#####################
#		 Pickle
#####################
def save(fname, cont):
	with open(fname, mode='wb') as f:
		dill.dump(cont, f)

def load(fname):
	with open(fname, mode='rb') as f:
		return dill.load(f)

#####################
# 	BIOタグ生成 ← やめた
# 	2値分類 ← ○
#####################
# def makeBIO(result, span, label):
# 	totallen = 0
# 	for i, morph in enumerate(result):
# 		if label in ['F', 'A', 'M']:
# 			if totallen == span[0]:
# 				result[i].append('B')
# 			elif span[0] < totallen < span[1]:
# 				result[i].append('I')
# 			else:
# 				result[i].append('O')
# 			totallen += len(morph[0])
# 		else:
# 			result[i].append('O')

# 	return result

#####################
# 	正解データ生成
#####################
# def makeAns(bio):
# 	labels = []
# 	for label in bio:
# 		if label == 'O':
# 			labels.append(0)
# 		elif label == 'B':
# 			labels.append(1)
# 		elif label == 'I':
# 			labels.append(2)

# 	return labels

 	# if 'B' in bio and 'I' in bio:
	# 	df = pd.get_dummies(pd.DataFrame(bio))
	# 	return to_np(df.values)
	# else:
	# 	hurei = []
	# 	for i in range(0, len(bio)):
	# 		hurei.append([0, 0, 1])
	# 	return to_np(hurei)

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

	# a = [1,2,3,4,5]
	# a[2:4] # [3,4]

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

	# else:	# マッチング上のスパンと、形態素解析の結果の境界が一致しない
	# 	print(text)

	# 前処理
	# sentence = re.sub(r'[　|◆]', '', sentence)
	# if '　' in sent:

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

#####################
# 	用例DBのXMLパーズ
#####################
def parseDB(datapath, vocab, fvocab, getsize):
	# mwelist, cnt = [], 0
	mwelist = load('./tmp/mwelist.pickle')
	sentlist, featlist, marklist, labels = [], [], [], []
	senthash, reshash = {}, {}
	mwehash = collections.defaultdict(lambda: collections.defaultdict(list))
	sid = 0

	for file in glob.iglob(datapath + '*.xml'):
		tree = ET.parse(file)
		root = tree.getroot()
		mwe = root.attrib["name"]
		freq = root.attrib["freq"] # 出現頻度: 文字列として比較したとき、完全に同一であるような複数の文が含まれていることがある。そのような文について、最初の文のみを残し、それ以外の文を取り除いた場合の、全てのターゲット文字列の出現頻度
		freq_total = root.attrib["total"] # 全出現頻度: 毎日新聞1995から、見つかった全てのターゲット文字列の出現頻度

		mwehash[mwe]['fpath'].append(file)
		# print(len(mwelist))
		# print('mwe:', mwe)
		if mwe in mwelist:	# 用法に曖昧性がある場合
			for elm in root.findall(".//example"):
				sentence = elm.text
				label = elm.attrib["label"]
				span = elm.attrib["target"].split('-')

				sent, feature, mark = morphParse(sentence, span, getsize)
				if len(sent) != 0:
					sentlist.append((get_wid(vocab, sent)))
					featlist.append((get_wid(fvocab, feature)))
					marklist.append(mark)
					senthash[sid] = [sentence, mwe]
					reshash[tuple(sent)] = sid
					sid += 1

					# 正解データ
					if label in ['F', 'A', 'M']:
						labels.append(1)
						mwehash[mwe]['func'].append(sentence)
					else:
						labels.append(0)
						mwehash[mwe]['cont'].append(sentence)

	smaxlen = getMaxlen(sentlist)
	fmaxlen = getMaxlen(featlist)

	hashlist = [senthash, reshash, mwehash]
	vocablist = [vocab, fvocab]
	return padding(sentlist, smaxlen), padding(featlist, fmaxlen), padding(marklist, smaxlen), labels, vocablist, hashlist

		# func, cont = 0, 0
		# for elm in root.findall(".//example"):
		# 	label = elm.attrib["label"]

		# 	if label in ['F', 'A', 'M']:
		# 		func += 1
		# 	else:
		# 		cont += 1

		# if int(freq) >= 50:
		# 	total = len(root.findall(".//example"))
		# 	ratio = func / total
		# 	if 0.18 < ratio < 0.83:
		# 		mwelist.append(mwe)
				# cnt += 1
				# print('mwe:', mwe, 'func ratio:', func, '/', total, '=', ratio)
	# save('./tmp/mwelist.pickle', mwelist)

###############
#	引数処理
###############
def get_arg():
	parser = argparse.ArgumentParser(description='converter')
	parser.add_argument('-size','--size', type=int, default=2)	# 前後形態素取得数(4=>前後それぞれ4取得)
	parser.add_argument('-dim','--dim', type=int, default=100)
	args = parser.parse_args()
	return args

def calcWMD(model, sent1, sent2):
	print(model.wmdistance(sent1, sent2))

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

	# print(model.vocab)
	# print(model.most_similar('巨人'))
	return model

def embedding(mpath, vocab_inv, w2vmodel, dim):
	# embedIDの初期重み作成
	initialW = []
	for i in range(0,len(vocab)):
		word = vocab_inv[i]		# wid=1から順にword取得
		try:
			w2vec = w2vmodel[word]
		except:
			w2vec = np.random.normal(scale=1.0, size=dim)
		initialW.append(w2vec)

	initialW = cupy.array(initialW, dtype=cupy.float32)
	return initialW


#####################
#		main
#####################
if __name__ == '__main__':
	vocab = collections.defaultdict(lambda: len(vocab))	# 語彙用
	fvocab = collections.defaultdict(lambda: len(fvocab)) # feature用
	cvocab = collections.defaultdict(lambda: len(cvocab))

	# import cupy
	args = get_arg()
	print('getsize:', args.size)

	datapath = '../../../data/MUST-dist-1.0/data/'
	mwelist = load('./tmp/mwelist.pickle')
	print(len(mwelist))	# 52

	# for mwe in mwelist:
	# 	print(mwe)
	# parseDB(datapath, vocab, fvocab, args.size)

	# sentlist, featlist, marklist, labels, vocablist, hashlist = parseDB(datapath, vocab, fvocab, args.size)
	# print(len(labels))
	# senthash, reshash, mwehash = hashlist
	# vocab, fvocab = vocablist

	# save('./tmp/sentlist'+str(args.size)+'.pickle', sentlist)
	# save('./tmp/featlist'+str(args.size)+'.pickle', featlist)
	# save('./tmp/marklist'+str(args.size)+'.pickle', marklist)
	# save('./tmp/labels'+str(args.size)+'.pickle', labels)
	# save('./tmp/vocab'+str(args.size)+'.pickle', vocab)
	# save('./tmp/fvocab'+str(args.size)+'.pickle', fvocab)
	# save('./tmp/senthash'+str(args.size)+'.pickle', senthash)
	# save('./tmp/reshash'+str(args.size)+'.pickle', reshash)

	# save('./tmp/mwehash.pickle', mwehash)
	# mwehash = load('./tmp/mwehash.pickle')

	# mwe = 'をとわずに[をとわずに],を問わずに[をとわずに]'
	# print('mwe: ', mwe)
	# for yoho, sentences in mwehash[mwe].items():
	# 	# print('########################')
	# 	print('-------------------')
	# 	print('用法: ', yoho)
	# 	for i, sentence in enumerate(sentences):
	# 		print(i, sentence)


	# 機能表現辞書
	# dicpath = '../../../result/tsutsuji_dic_20170202.json'
	# with open(dicpath) as f:
	# 	func_dict = json.load(f)

	# headlist = []
	# for mweid, dic in func_dict.items():
	# 	headlist.append(dic['headword'])

	# headlist = list(set(headlist))
	# headlist.sort()
	
	# trie = Trie(headlist)
	# print(dir(trie))
	# # print(trie[717])
	# for query in headlist:
	# 	print('---------------------------')
	# 	print('query: ', query)
	# 	result = trie.search(query)
	# 	print('result:', result)

	# for mweid, dic in func_dict.items():
		# print('---------------------------')
		# if dic['headword'] == 'をとわず':
			# print('headword: ', dic['headword'])
			# print(dic)
			# print(dic['variation'])



	# vocab = load('./tmp/vocab'+str(args.size)+'.pickle')
	# vocab_inv = {v: k for k, v in vocab.items()}

	# sentlist = load('./tmp/sentlist.pickle')

	# print(sentlist[0])
	# array = filter(lambda s:s != -1, sentlist[0])
	# print([vocab_inv[i] for i in array])

	mpath = './model/wiki_vector'+str(args.dim)+'.model'

	# from gensim.models import word2vec
	# # # # model = trainw2v(mpath, dim=args.dim)	# train
	# # # # model.save(mpath)		# save

	# w2vmodel = word2vec.Word2Vec.load(mpath)	# load
	# # print(w2vmodel.most_similar('巨人'))
	# # print(w2vmodel.most_similar('対し'))
	# initialW = embedding(mpath, vocab_inv, w2vmodel, args.dim)
	# save('./tmp/initialW_'+str(args.dim)+'_size'+str(args.size)+'.pickle', initialW)

	# print('len(vocab):', len(vocab))
	# print('initialW.shape:', initialW.shape)	# (len(vocab), embed_size)

	# 用法の割合カウント
	# 827
	# 1459
	# labels = load('./tmp/labels'+str(args.size)+'.pickle')
	# print(labels.count(0))
	# print(labels.count(1))

	# for sent, label in test_data:
	# 	y.append(label)
	# print(y.count(0))	# 56/185
	# print(y.count(1))	# 129/185
	# 全部1モデル: 129/185 = 0.70








		