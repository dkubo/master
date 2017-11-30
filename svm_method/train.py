# coding: utf-8

from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cross_validation import train_test_split
# from sklearn.model_selection import KFold
from sklearn.datasets import load_digits
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import numpy as np
import svmdata
import argparse


def get_arg():
	parser = argparse.ArgumentParser(description='converter')
	parser.add_argument('-dim','--dim', type=int, default=50)
	parser.add_argument('-cluster','--cluster', type=int, default=3)
	parser.add_argument('-split','--split', type=int, default=1)
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = get_arg()
	C, kernel, degree = 1.0, 'poly', 2

	# digits = load_digits()

	##### クラスタリングなし（曖昧性のある機能表現全部一気に）#####
	if args.split == 0:
		args.dim = 50
		vocablist = svmdata.load('./tmp/vocablist.pickle')
		bow_data = svmdata.load('./tmp/bow_data.pickle')
		labels = svmdata.load('./tmp/labels.pickle')
		X = np.array(bow_data)
		y = np.array(labels)
		X_train, X_test, y_train, y_test  = train_test_split(bow_data, labels, test_size=0.1, random_state=53)
		print(y_test.count(0), y_test.count(1))
		pca = PCA(args.dim)
		pca.fit(X_train)
		E = pca.explained_variance_ratio_
		print('dim: ', args.dim, "累積寄与率", sum(E)) 	# 累積寄与率
		X_train = pca.transform(X_train)
		X_test = pca.transform(X_test)

		clf = svm.SVC(C=C, kernel=kernel, degree=degree)
		clf.fit(X_train, y_train)
		score = clf.score(X_test, y_test)
		print('accuracy: ', score)

	###### クラスタリングあり #####
	elif args.split == 1:
		total_acc = 0.0
		args.dim = 30
		for clustnum in range(1, args.cluster+1):
			print('-----------------------')
			print(clustnum)
			vocablist = svmdata.load('./tmp/vocablist_'+str(clustnum)+'.pickle')
			datalist = svmdata.load('./tmp/datalist'+str(clustnum)+'.pickle')
			labels = svmdata.load('./tmp/labels'+str(clustnum)+'.pickle')
			bow_data = svmdata.load('./tmp/bow_data'+str(clustnum)+'.pickle')
			X = np.array(bow_data)
			y = np.array(labels)
			X_train, X_test, y_train, y_test  = train_test_split(bow_data, labels, test_size=0.1, random_state=53)
			print(y_test.count(0), y_test.count(1))

			pca = PCA(args.dim)
			pca.fit(X_train)
			E = pca.explained_variance_ratio_
			print('dim: ', args.dim, "累積寄与率", sum(E)) 	# 累積寄与率
			X_train = pca.transform(X_train)
			X_test = pca.transform(X_test)

			clf = svm.SVC(C=C, kernel=kernel, degree=degree)
			clf.fit(X_train, y_train)
			score = clf.score(X_test, y_test)
			total_acc += score
			print('accuracy_', clustnum, ': ', score)
			# print('pred label', clf.predict(X_test))
		print('-----------------------')
		print('mean_accuracy', total_acc / args.cluster)
	



		# kf = KFold(n_splits=10, shuffle=True, random_state=53)
		# for train_index, test_index in kf.split(X, y):
		# 	# print("TRAIN:", train_index, "TEST:", test_index)
		# 	X_train, X_test = X[train_index], X[test_index]
		# 	y_train, y_test = y[train_index], y[test_index]





