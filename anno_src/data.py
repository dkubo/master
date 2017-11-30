# coding: utf-8

import dill, re, csv, argparse
from collections import defaultdict, Counter

'''
全2500文
1500: 
→曖昧性のある表現を対象に、BCCWJから抽出
　→曖昧性のあるモデルに対する固定の評価データとして使える

1000: 
　→以前BCCWJ上で用法の割合を計算した時のデータ使って、用法の割合が同等の表現を対象にする
　　→出現回数が一定数以上ある表現　かつ
　　→MUSTにある表現は対象外に！
'''

#####################
#		 Pickle
#####################
def save(fname, cont):
	with open(fname, mode='wb') as f:
		dill.dump(cont, f)

def load(fname):
	with open(fname, mode='rb') as f:
		return dill.load(f)

def writeCSV(fname, datalist):
	with open(fname, 'w') as f:
		for sid, sentence in datalist.items():
			f.write(sid+','+sentence+'\n')

###############
#	引数処理
###############
def get_arg():
	parser = argparse.ArgumentParser(description='converter')
	parser.add_argument('-dtype','--dtype', type=int, default=1)
	args = parser.parse_args()
	return args


####################################################
#	mwe = といえば[といえば],と言えば[といえば]
#	mwehash[mwe] = [といえば, と言えば]

####################################################
def mweMatchHash(mwelist):
	mwehash = defaultdict()
	matchlist = []
	for mwe in mwelist:
		mwes = []
		for mwepart in mwe.split(','):
			mwepart = mwepart.split('[')
			mwepart[1] = mwepart[1].replace(']', '')
			mwes += mwepart

		mwehash[mwe] = list(set(mwes))
		matchlist += list(set(mwes))

	# matchlist = list(set(matchlist))
	return matchlist, mwehash

########################
# 機能表現の前後に記号挿入
########################
def insertKigo(sentence):
	sentence.insert(1, '★')
	sentence.insert(3, '★')
	return ''.join(sentence)


########################
# 	各表現最大50抽出
########################
def extractSent(srcdata, annolist, maxfreq, dtype):
	annoData = {}
	annocnt = 0	# アノテーション対象作業の総数
	cntsent = defaultdict(int)
	# MUST上で曖昧性のある表現が、BCCWJ_LUW上でどれだけマッチングするかカウント
	matchcnt = defaultdict(int)
	with open(srcpath, 'r') as f:
		for sentence in f:
			if dtype == 1:
				sentence = sentence.rstrip().split(",")
				sid, mwe = sentence[0], sentence[4]
				sentence = insertKigo(sentence[3::])
			elif args.dtype == 2:
				sentence = sentence.rstrip()[1:-1].split("\t")
				sid, mwe = sentence[0], sentence[4]
				sentence = insertKigo(sentence[3:-1])

			if (mwe in annolist) and (cntsent[mwe] < 50):
				if not annoData.get(sid):
					annoData[sid] = sentence
					cntsent[mwe] += 1
					annocnt += 1

	print('作業箇所: ', annocnt)
	return annoData
	

if __name__ == '__main__':
	args = get_arg()
	if args.dtype == 1:
		######### MUST(曖昧性のある表現)のリスト #########
		print('===========================')
		srcpath = "./result/matched_suw.tsv"
		annopath1 = './result/annotation_data1.csv'
		mwelist1 = load('./result/mwelist1.pickle')
		print('全機能表現数: ', len(mwelist1))
		annoData1 = extractSent(srcpath, mwelist1, 50, 1)	# 各表現/最大50文抽出
		print('全', str(len(annoData1)), '行')
		writeCSV(annopath1, annoData1)
	elif args.dtype == 2:
		# ######### BCCWJ_LUW(曖昧性のある機能表現(つつじのエントリ=長単位となるもの))のリスト #########
		print('===========================')
		srcpath = "../../../result/bccwj/bccwj_matced_0128_edited.tsv"
		annopath2 = './result/annotation_data2.csv'
		mwelist2 = load('./result/mwelist2.pickle')	# 曖昧性MWE on BCCWJ
		print('全機能表現数: ', len(mwelist2))
		annoData2 = extractSent(srcpath, mwelist2, 50, 2)	# 各表現/最大50文抽出
		print('全', str(len(annoData2)), '行')
		writeCSV(annopath2, annoData2)



