# coding: utf-8

import MeCab
import xml.etree.ElementTree as ET
import glob, sys, re, dill, argparse, collections
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def save(fname, cont):
	with open(fname, mode='wb') as f:
		dill.dump(cont, f)

def load(fname):
	with open(fname, mode='rb') as f:
		return dill.load(f)

def get_wid(vocablist, x_input):
	vid_list = []
	for morph in x_input:
		feature_list = []
		for i, feature in enumerate(morph):
			feature_list.append(vocablist[i][feature])
		vid_list.append(feature_list)
	return vid_list

def morphParse(text, span):
	result, x_input, = [], []
	idxspan, totallen = [], 0
	markfrg = 0

	m = MeCab.Tagger()
	parse = m.parse(text).split('\n')[:-2]
	for i, morph in enumerate(parse):
		# IPADIC: [表層形, 品詞, 品詞細分類1, 品詞細分類2, 品詞細分類3, 活用型, 活用形, 原形, 読み, 発音]
		# UNIDIC: [表層形, 発音, 原形読み?, 原形, 品詞大分類-品詞中分類-品詞小分類-品詞細分類, 活用型, 活用形]
		if totallen == int(span[0]):
			idxspan.append(i)
			markfrg = 1

		morph = morph.split('\t')
		poslist = morph[4].split('-')
		# padding to 4
		for num in range(0, 4-len(poslist)):
			poslist.append('')
		for pos in poslist[::-1]:
			morph.insert(4, pos)
		morph.pop(8)
		if markfrg == 0:
			morph.append(0)
		else:
			morph.append(1)

		totallen += len(morph[0])	# 出現形		
		if totallen == int(span[1]):
			idxspan.append(i)
			markfrg = 0

		result.append(morph)

	getsize = 2
	if len(idxspan) == 2:
		# 機能表現の前後形態素を参照
		if idxspan[0] < getsize:
			pre = result[0:idxspan[0]]
			funcmwe = result[idxspan[0]:idxspan[1]+1]
			post = result[idxspan[1]+1:idxspan[1]+(getsize+1)]
			x_input = pre + funcmwe + post

		elif idxspan[0] >= getsize:
			pre = result[idxspan[0]-getsize:idxspan[0]]
			funcmwe = result[idxspan[0]:idxspan[1]+1]
			post = result[idxspan[1]+1:idxspan[1]+(getsize+1)]
			x_input = pre + funcmwe + post

	return x_input

def parseDB(datapath, vocablist, mwelist):
	labels = []
	datalist = []

	for file in glob.iglob(datapath + '*.xml'):
		tree = ET.parse(file)
		root = tree.getroot()
		mwe = root.attrib["name"]

		if mwe in mwelist:	# 用法に曖昧性がある場合
			for elm in root.findall(".//example"):
				sentence = elm.text
				label = elm.attrib["label"]
				span = elm.attrib["target"].split('-')
				x_input = morphParse(sentence, span)
				if len(x_input) != 0:
					vid_list = get_wid(vocablist, x_input)
					datalist.append(vid_list)
					if label in ['F', 'A', 'M']:
						labels.append(1)
					else:
						labels.append(0)

	return datalist, labels, vocablist

def bag_of_words(datalist, vocablist):
	new_datalist = []
	for sentence in datalist:
		bows = np.array([])
		sentence = np.array(sentence).T
		# 10個のbag_of_words作る
		for i, feature in enumerate(sentence):
			bow = np.zeros(len(vocablist[i]))
			for seq in feature:
				bow[seq] += 1
			bows = np.concatenate([bows, bow])
			# bows.append(bow)
		new_datalist.append(bows)
	return new_datalist

def get_arg():
	parser = argparse.ArgumentParser(description='converter')
	parser.add_argument('-cluster','--cluster', type=int, default=3)
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = get_arg()
	datapath = '../../../data/MUST-dist-1.0/data/'
	vocab1 = collections.defaultdict(lambda: len(vocab1))
	vocab2 = collections.defaultdict(lambda: len(vocab2))
	vocab3 = collections.defaultdict(lambda: len(vocab3))
	vocab4 = collections.defaultdict(lambda: len(vocab4))
	vocab5 = collections.defaultdict(lambda: len(vocab5))
	vocab6 = collections.defaultdict(lambda: len(vocab6))
	vocab7 = collections.defaultdict(lambda: len(vocab7))
	vocab8 = collections.defaultdict(lambda: len(vocab8))
	vocab9 = collections.defaultdict(lambda: len(vocab9))
	vocab10 = collections.defaultdict(lambda: len(vocab10))
	vocab11 = collections.defaultdict(lambda: len(vocab11))	# mark
	vocablist = [vocab1, vocab2, vocab3, vocab4, vocab5, vocab6, vocab7, vocab8, vocab9, vocab10, vocab11]

	# クラスタリングなし（曖昧性のある機能表現全部一気に）
	# mwelist = load('../nn/tmp/mwelist.pickle')
	# datalist, labels, vocablist = parseDB(datapath, vocablist, mwelist)
	# save('./tmp/vocablist.pickle', vocablist)
	# save('./tmp/datalist.pickle', datalist)
	# save('./tmp/labels.pickle', labels)
	# bow_data = bag_of_words(datalist, vocablist)
	# save('./tmp/bow_data.pickle', bow_data)

	# クラスタリングなし（曖昧性のある機能表現全部一気に）
	for clustnum in range(1, args.cluster+1):
		print(clustnum)
		mwelist = load('../wv_cluster/result/clusterlist_'+str(clustnum)+'.pickle')
		datalist, labels, vocablist = parseDB(datapath, vocablist, mwelist)
		bow_data = bag_of_words(datalist, vocablist)
		save('./tmp/vocablist_'+str(clustnum)+'.pickle', vocablist)
		save('./tmp/datalist'+str(clustnum)+'.pickle', datalist)
		save('./tmp/labels'+str(clustnum)+'.pickle', labels)
		save('./tmp/bow_data'+str(clustnum)+'.pickle', bow_data)


	# print([len(vocab) for vocab in vocablist])	# [2657, 2433, 2285, 2426, 16, 22, 10, 5, 52, 20, 2]
	# save('./tmp/vocablist.pickle', vocablist)
	# save('./tmp/datalist.pickle', datalist)
	# save('./tmp/labels.pickle', labels)

	# vocablist = load('./tmp/vocablist.pickle')
	# datalist = load('./tmp/datalist.pickle')
	# labels = load('./tmp/labels.pickle')

	# bow_data = bag_of_words(datalist, vocablist)
	# save('./tmp/bow_data.pickle', bow_data)
	# bow_data = load('./tmp/bow_data.pickle')


	# count_vectorizer = CountVectorizer()
	# feature_vectors = count_vectorizer.fit_transform(['aad bad cad bad cad dad', 'fad rad aad bad dad gad'])
	# print(feature_vectors)
	# vocabulary = count_vectorizer.get_feature_names()
	# print(vocabulary)


