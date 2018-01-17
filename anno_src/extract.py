# coding: utf-8

'''
アノテーションの結果から，外部データの実験データを作成する
'''
import pandas as pd
import dill as pickle
from collections import defaultdict
import MeCab
from collections import OrderedDict	# 順番を保持


def save(fname, cont):
	with open(fname, mode='wb') as f:
		pickle.dump(cont, f)

def load(fname):
	with open(fname, mode='rb') as f:
		return pickle.load(f)

def mark_feature(n_pre, n_mwe, n_post):
	mark = []
	for i in range(0, n_pre):
		mark.append(0)
	for i in range(0, n_mwe):
		mark.append(1)
	for i in range(0, n_post):
		mark.append(0)
	return mark

def get_wid(vocab, words):
	return [vocab[word] for word in words]


def morphParse(text, span, getsize):
	left, mwe, right = text
	m = MeCab.Tagger()
	left = m.parse(left).split('\n')[0:-2]
	mwe = m.parse(mwe).split('\n')[0:-2]
	right = m.parse(right).split('\n')[0:-2]

	left_word = [word.split('\t')[0] for word in left][-2:]
	mwe_word = [word.split('\t')[0] for word in mwe]
	right_word = [word.split('\t')[0] for word in right][0:2]

	left_pos = ['-'.join(word.split('\t')[4].split('-')[0:2]) for word in left][-2:]
	mwe_pos = ['-'.join(word.split('\t')[4].split('-')[0:2]) for word in mwe]
	right_pos = ['-'.join(word.split('\t')[4].split('-')[0:2]) for word in right][0:2]

	left_katsu = ['-'.join(word.split('\t')[6].split('-')[0:1]) for word in left][-2:]
	mwe_katsu = ['-'.join(word.split('\t')[6].split('-')[0:1]) for word in mwe]
	right_katsu = ['-'.join(word.split('\t')[6].split('-')[0:1]) for word in right][0:2]

	sent = left_word + mwe_word + right_word
	pos = left_pos + mwe_pos + right_pos
	katuyokei = left_katsu + mwe_katsu + right_katsu
	feature = pos + katuyokei
	mark = mark_feature(len(left_word), len(mwe_word), len(right_word))

	return sent, feature, mark

def makeXy(sidhash, vocab, fvocab, featmaxlen):
	import cupy
	sentlist, featlist, marklist, x, y = [], [], [], [], []
	for k, v in sidhash.items():
		sentlist.append((get_wid(vocab, v['sent'])))
		featlist.append((get_wid(fvocab, v['feat'])))
		marklist.append(v['mark'])
		y.append(v['label'])

	sentlist = padding(sentlist, featmaxlen)
	featlist = padding(featlist, featmaxlen)
	marklist = padding(marklist, featmaxlen)
	for sent, feat, mark in zip(sentlist, featlist, marklist):
		x.append([sent, feat, mark])

	return cupy.array(x, dtype=cupy.int32), cupy.array(y, dtype=cupy.int32)

def padding(data, maxlen):
	newdata = []
	for sent in data:
		for i in range(0, maxlen-len(sent)):
			sent.append(-1)
		newdata.append(sent)
	return newdata

if __name__ == '__main__':
	sidhash, labelhash = OrderedDict(), defaultdict(int)
	featmaxlen = 0
	file = pd.ExcelFile('20180115_MWE_output.xlsx')
	df = file.parse('output', skiprows=0, parse_cols="A,C:D,F:H,J:K,N")
	cnt = defaultdict(int)
	for index, row in df.iterrows():
		filetype, bccwj_id, sentid, left, mwe, right, start, mwelen, label = row
		if filetype == 2 and label != 4:
			span, getsize = [start, start+mwelen], 2
			sent, feature, mark = morphParse([left, mwe, right], span, getsize)
			if len(sent) != 0:
				if label in [1, 3]:
					label = 1
				elif label == 2:
					label = 0
				featmaxlen = max(featmaxlen, len(feature))
				sidhash[(bccwj_id, sentid)] = {'sent':sent, 'feat':feature, 'mark': mark, 'label':label}

	for i in [1, 2, 3, 4, 'all']:
		vocab = load('../wv_cluster/result/vocab_'+str(i)+'_amb.pickle')
		fvocab = load('../wv_cluster/result/fvocab_'+str(i)+'_amb.pickle')
		x_test, y_test = makeXy(sidhash, vocab, fvocab, featmaxlen)
		save('./result/bccwj_x_test_'+str(i)+'.pickle', x_test)
		save('./result/bccwj_y_test_'+str(i)+'.pickle', y_test)







