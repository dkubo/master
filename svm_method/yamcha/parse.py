# coding: utf-8

import dill, csv, os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.cross_validation import train_test_split
import subprocess



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

def openTSV(fname):
	with open(fname, 'r') as f:
		reader = csv.reader(f, delimiter="\t")
		return [row for row in reader]

def calcRec(result):
	gold, pred = [], []
	rec_under, rec_upper = 0, 0
	fam_list = ['B-F', 'B-A', 'B-M', 'I-F', 'I-A', 'I-M']
	# fam_list = ['B-FAM', 'I-FAM']
	for morph in result:
		if len(morph) != 0:
			gold_tag, pred_tag = morph[-2:]
			# goldのチャンクのインデックス取得
			if gold_tag in fam_list:	# 検出器FAM
				gold.append(gold_tag)
				pred.append(pred_tag)
			
				# if gold_tag[-1] in ['F', 'A', 'M']:
				# 	gold.append(gold_tag[0:2]+'FAM')
				# else:
				# 	gold.append(gold_tag[0:2]+'C')

				# if pred_tag[-1] in ['F', 'A', 'M']:
				# 	pred.append(pred_tag[0:2]+'FAM')
				# else:
				# 	pred.append(pred_tag[0:2]+'C')

		else:	# 文区切り
			if gold != []:
				rec_under += 1	# 評価データに存在するB-FAM,I-FAMのチャンク数
				if gold == pred:
					rec_upper += 1
			# print(gold, pred)
			gold, pred = [], []

	print('recall: ', rec_upper, '/', rec_under, '=', rec_upper/rec_under)
	return rec_upper/rec_under

def calcPrec(result):
	gold, pred = [], []
	prec_under, prec_upper = 0, 0
	# fam_list = ['B-FAM', 'I-FAM']
	fam_list = ['B-F', 'B-A', 'B-M', 'I-F', 'I-A', 'I-M']
	for morph in result:
		if len(morph) != 0:
			gold_tag, pred_tag = morph[-2:]
			if pred_tag in fam_list:	# 検出器FAM
				gold.append(gold_tag)
				pred.append(pred_tag)

				# if pred_tag[-1] in ['F', 'A', 'M']:
				# 	pred.append(pred_tag[0:2]+'FAM')
				# else:
				# 	pred.append(pred_tag[0:2]+'C')

				# if gold_tag[-1] in ['F', 'A', 'M']:
				# 	gold.append(gold_tag[0:2]+'FAM')
				# else:
				# 	gold.append(gold_tag[0:2]+'C')


		else:	# 文区切り
			if pred != []:
				prec_under += 1	# B-FAM,I-FAMとして検出されたチャンク数
				# print(gold, pred)
				if gold == pred:
					prec_upper += 1	# 実際にB-FAM, I-FAMのチャンクタグだった数
			gold, pred = [], []

	print('prec: ', prec_upper, '/', prec_under, '=', prec_upper/prec_under)
	return prec_upper/prec_under

# accuracy計算
def calcAcc(result):
	gold, pred = [], []
	under, upper = 0, 0
	# fam_list = ['B-F', 'B-A', 'B-M', 'I-F', 'I-A', 'I-M']
	fam_list = ['B', 'I']
	for morph in result:
		if len(morph) != 0:
			gold_tag, pred_tag = morph[-2:]
			if gold_tag[0] in fam_list:	# 検出器FAM
					gold.append(gold_tag)
					pred.append(pred_tag)

				# if gold_tag[-1] in ['F', 'A', 'M']:
				# 	gold.append(gold_tag[0:2]+'FAM')
				# else:
				# 	gold.append(gold_tag[0:2]+'C')

				# if pred_tag[-1] in ['F', 'A', 'M']:
				# 	pred.append(pred_tag[0:2]+'FAM')
				# else:
				# 	pred.append(pred_tag[0:2]+'C')

		else:	# 文区切り
			if gold != []:
				under += 1
				if gold == pred:
					upper += 1	# 実際にB-FAM, I-FAMのチャンクタグだった数

			gold, pred = [], []

	print('acc: ', upper, '/', under, '=', upper/under)
	return upper/under

#####################
#		main
#####################
if __name__ == '__main__':
	datalist = load('data.pickle')
	datalist = np.array(datalist)
	total_prec = 0.0
	total_rec = 0.0
	total_acc = 0.0

	#####################
	# nnのデータと比較
	#####################
	trainset, testset = train_test_split(datalist, test_size=0.1, random_state=53)
	writeTSV('./result/trainset.tsv', trainset)
	writeTSV('./result/testset.tsv', testset)

	# train
	os.system('nkf -Lu ./result/trainset.tsv > ./result/trainset_Lu.tsv')
	modelname = './result/train_svm'
	subprocess.check_output('make CORPUS=./result/trainset_Lu.tsv MODEL='+modelname+' train', shell=True)
	# test
	os.system('nkf -Lu ./result/testset.tsv > ./result/testset_Lu.tsv')
	resultname = './result/result.tsv'
	os.system('yamcha -m '+modelname+'.model < ./result/testset_Lu.tsv > '+ resultname)
	
	result = openTSV(resultname)
	rec = calcRec(result)	# 再現率計算
	prec = calcPrec(result)	# 精度計算
	acc = calcAcc(result)
	print('f-measure: ', 2*prec*rec/(prec+rec))
	#####################

	#####################
	# 10-fold
	#####################
	# kf = KFold(n_splits=10, shuffle=True, random_state=43)
# 	for i, (train_index, test_index) in enumerate(kf.split(datalist)):
# 		print(i, "TRAIN:", train_index, "TEST:", test_index)
# 		trainset = datalist[train_index]
# 		testset = datalist[test_index]
# 		writeTSV('./result/trainset.tsv', trainset)
# 		writeTSV('./result/testset.tsv', testset)

# # 		# train
# 		os.system('nkf -Lu ./result/trainset.tsv > ./result/trainset_Lu.tsv')
# 		modelname = './result/train_svm_' + str(i)
# 		subprocess.check_output('make CORPUS=./result/trainset_Lu.tsv MODEL='+modelname+' train', shell=True)

# # 		# test
# 		os.system('nkf -Lu ./result/testset.tsv > ./result/testset_Lu.tsv')
# 		resultname = './result/result_'+str(i)+'.tsv'
# 		os.system('yamcha -m '+modelname+'.model < ./result/testset_Lu.tsv > '+ resultname)
		
# 		result = openTSV(resultname)
# 		print('------------------------')
# 		print(i)
# 		rec = calcRec(result)	# 再現率計算
# 		prec = calcPrec(result)	# 精度計算
# 		acc = calcAcc(result)
# 		total_rec += rec
# 		total_prec += prec
# 		total_acc += acc

	# print('------------------------')
	# ave_rec = total_rec / 10
	# ave_prec = total_prec / 10
	# ave_acc = total_acc / 10
	# print('ave_prec: ', ave_prec, 'ave_rec: ', ave_rec)
	# print('f-measure: ', 2*ave_prec*ave_rec/(ave_prec+ave_rec))
	# print('ave_acc: ', ave_acc)
	#####################

# # 

	# for i in range(0, 10):
	# 	resultname = './result/result_'+str(i)+'.tsv'
	# 	result = openTSV(resultname)
	# 	print('------------------------')
	# 	print(i)
	# 	rec = calcRec(result)	# 再現率計算
	# 	prec = calcPrec(result)	# 精度計算
	# 	acc = calcAcc(result)
	# 	total_rec += rec
	# 	total_prec += prec
	# 	total_acc += acc


# # make CORPUS=train.tsv MODEL=svm_morph FEATURE="F:-2..2:0.. T:-2..-1" SVM_PARAM="-t1 -d2 -c1" train
# # MULTI_CLASS=2
# target_names = ['B-A', 'B-B', 'B-C', 'B-F', 'B-M', 'B-Y', 'I-A', 'I-B', 'I-C', 'I-F', 'I-M', 'I-Y']
####### 検出器FAM #######
# 精度(prec): 検出に成功したチャンク数 / 解析によって検出されたチャンク数
# 		= 解析に成功したチャンク数 / B-FAM,I-FAMとして検出されたチャンク数
# 再現率(rec):  検出に成功したチャンク数 / 評価データに存在するチャンク数
# 		= 解析に成功したチャンク数 / 評価データに存在するB-FAM,I-FAMのチャンク数
# F-値: 2 * 精度 * 再現率 / 精度 + 再現率
# 判別率: 正解したラベル数 / 全判定ラベル数
# 		= 
########################


