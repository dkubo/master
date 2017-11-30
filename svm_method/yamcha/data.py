# coding: utf-8

import MeCab
import xml.etree.ElementTree as ET
import glob, dill, csv
import numpy as np
# from sklearn.cross_validation import train_test_split

#####################
#		 Pickle
#####################
def save(fname, cont):
	with open(fname, mode='wb') as f:
		dill.dump(cont, f)

def load(fname):
	with open(fname, mode='rb') as f:
		return dill.load(f)

def writeTSV(fname, datalist):
	with open(fname, 'w') as f:
		# writer = csv.writer(f, delimiter=' ')
		writer = csv.writer(f, delimiter='\t')
		for sentence in datalist:
			for morph in sentence:
				writer.writerow(morph)	
			writer.writerow([])

def checkSpace(morph):
	result = []
	for m in morph:
		if len(m.split(' ')) != 1:
			result.append(m.split('-')[0])
		else:
			result.append(m)
	return result

def getFeat(text, span, label):
	x_input, idxspan = [], []
	totallen, frg = 0, 0
	k, j = 0, 0
	# if label in ['F', 'A', 'M']:
	# 	label = 'FAM'
	m = MeCab.Tagger()
	parse = m.parse(text).split('\n')[:-2]
	for i, morph in enumerate(parse):
		# IPADIC: [表層形, 品詞, 品詞細分類1, 品詞細分類2, 品詞細分類3, 活用型, 活用形, 原形, 読み, 発音]
		# UNIDIC: [表層形, 発音, 原形読み?, 原形, 品詞大分類-品詞中分類-品詞小分類-品詞細分類, 活用型, 活用形]
		if totallen == int(span[0]):
			j = i
			tag = 'B-' + label
			frg = 1
			idxspan.append(i)
		elif frg == 0:
			tag = 'O'
		elif frg == 1:
			tag = 'I-' + label

		result = checkSpace(morph.split('\t'))
		poslist = result[4].split('-')
		# padding to 4
		for num in range(0, 4-len(poslist)):
			poslist.append('')
		for pos in poslist[::-1]:
			result.insert(4, pos)
		result.pop(8)
		totallen += len(result[0])	# 出現形

		if totallen == int(span[1]):
			frg = 0
			idxspan.append(i)
			k = i
		result.append(tag)

		result = [x if x is not '' else '*' for x in result]
		# if len(result) != 11:
		# 	print(result)
		x_input.append(result)
	return x_input, idxspan, j, k

def chunkFeat(x_input, j, k):
	result = []
	for i, morph in enumerate(x_input):
		morph.insert(-1, k-j+1)
		morph.insert(-1, i-j+1)
		result.append(morph)
	return result

def parseDB(datapath):
	mwelist = load('../../nn/tmp/mwelist.pickle')	# 曖昧性のあるMWE
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
				x_input, idxspan, j, k = getFeat(sentence, span, label)
				if len(idxspan) == 2:
					x_input = chunkFeat(x_input, j, k)
					datalist.append(x_input)
	return datalist

if __name__ == '__main__':
	datapath = '../../../../data/MUST-dist-1.0/data/'
	# mwelist = load('../../tmp/mwelist.pickle')
	# print(mwelist) # ['ところが[ところが]', 'ていい[ていい],てよい[てよい],て良い[てよい],でいい[でいい],でよい[でよい],で良い[でよい]', 'にせよ[にせよ]', 'にくらべ[にくらべ],に比べ[にくらべ]', 'といい[といい],とよい[とよい],と良い[とよい]', 'といえば[といえば],と言えば[といえば]', 'かと思うと[かとおもうと]', 'ことがある[ことがある]', 'うと[うと]', 'ところを[ところを]', 'ても[ても]', 'とは[とは]', 'くせに[くせに],癖に[くせに]', 'とすると[とすると]', 'につき[につき]', 'といいながら[といいながら],と言いながら[といいながら]', 'じゃ[じゃ],ちゃ[ちゃ]', 'にあって[にあって]', 'に先だつ[にさきだつ],に先立つ[にさきだつ]', 'にあたって[にあたって],に当たって[にあたって],に当って[にあたって]', 'ものを[ものを]', 'ものなら[ものなら]', 'をはじめ[をはじめ]', 'にしても[にしても]', 'と思ったら[とおもったら]', 'にあたり[にあたり],に当たり[にあたり],に当り[にあたり]', 'おりから[おりから],折から[おりから]', 'うえでの[うえでの],上での[うえでの]', 'にきまっている[にきまっている],に決まっている[にきまっている]', 'とすれば[とすれば]', 'にしろ[にしろ]', 'をもって[をもって]', 'てはいけない[てはいけない],ではいけない[ではいけない]', 'うが[うが]', 'としても[としても]', 'としたら[としたら]', 'ことだ[ことだ]', 'にしたがい[にしたがい],に従い[にしたがい]', 'に応じて[におうじて]', 'あとで[あとで],後で[あとで]', 'というと[というと],と言うと[というと]', 'として[として]', 'かわりに[かわりに],代わりに[かわりに]', 'てはならない[てはならない],ではならない[ではならない]', 'とはいえ[とはいえ]', 'にとり[にとり]', 'にかけ[にかけ]', 'といっても[といっても],と言っても[といっても]', 'うえで[うえで],上で[うえで]', 'ところで[ところで]', 'に限る[にかぎる]', 'に応じた[におうじた]']
	datalist = parseDB(datapath)
	# print(datalist)
	save('data.pickle', datalist)
	# datalist = load('data.pickle')

	# train, test = train_test_split(datalist, test_size=0.1, random_state=53)
	# writeTSV('./train.tsv', train)
	# writeTSV('./test.tsv', test)

# make CORPUS=train.tsv MODEL=svm_morph FEATURE="F:-2..2:0.. T:-2..-1" SVM_PARAM="-t1 -d2 -c1" train
# MULTI_CLASS=2
