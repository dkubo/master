# coding: utf-8

from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
import MeCab
import xml.etree.ElementTree as ET
import glob, re, argparse
import data

####################
# 1. MUST1の機能的用法の用例データ(FAM)を対象に、機能表現の分散表現を学習
# 2. クラスタリング
####################
def saveWakati(fpath, wakatilist):
	with open(fpath, 'w') as f:
		for wakati in wakatilist:
			f.write(wakati)
			f.write('\n')

def getmwe(root, mwe, wakatilist, mwelist):
	for elm in root.findall(".//example"):
		sentence = elm.text
		label = elm.attrib["label"]
		span = elm.attrib["target"].split('-')
		if label in ['F', 'A', 'M']:
			wakati = morphWakati(sentence)
			wakati = concatMWE(wakati, span)
			if wakati is not None:
				mwelist.append(mwe)
				wakati = [word for word in wakati if word not in ['　', '◇']]
				wakatilist.append(' '.join(wakati)+'\n')

	return wakatilist, mwelist

######################
#	wakatilist作る
######################
def parseDB(datapath):
	wakatilist, mwelist = [], []
	for file in glob.iglob(datapath + '*.xml'):
		tree = ET.parse(file)
		root = tree.getroot()
		mwe = root.attrib["name"]
		# 全機能表現を対象
		wakatilist, mwelist = getmwe(root, mwe, wakatilist, mwelist)
	return wakatilist, list(set(mwelist))

def concatMWE(wakati, span):
	totallen, idxspan = 0, []
	for i, word in enumerate(wakati):
		if totallen == int(span[0]):
			idxspan.append(i)

		totallen += len(word)
		if totallen == int(span[1]):
			idxspan.append(i)

	if len(idxspan) == 2:	# 1の場合は、形態素境界が合わない場合
		mwe = ''.join(wakati[idxspan[0]:idxspan[1]+1])
		wakati.insert(idxspan[0], mwe)
		del wakati[idxspan[0]+1:idxspan[1]+2]
		return wakati
	else:
		return None

def morphWakati(sentence):
	wakati = ''
	m = MeCab.Tagger("-Owakati")
	parse = m.parse(sentence).split(' ')
	return parse[0:-1]

###############
#	w2v
###############
def trainw2v(modelpath, wakatipath, dim):
	from gensim.models import word2vec
	# Load the corpus
	data = word2vec.Text8Corpus(wakatipath)
	# Train
	model = word2vec.Word2Vec(data, size=dim, window=5, min_count=0, workers=15)
	return model

def loadModel(modelpath):
	from gensim.models.keyedvectors import KeyedVectors
	return KeyedVectors.load(modelpath)
	# from gensim.models import word2vec
	# return word2vec.Word2Vec.load(modelpath)	# load

######################################
#	異表記のベクトルを平均で1つのベクトルにする(e.g.「という、と言う」)
######################################
def get_wv(mwelist, model):
	cnt, vector = 0, 0
	for mwe in mwelist:
		try:
			vector = model.wv[mwe]
			cnt += 1
		except:
			next

	if cnt != 0:
		return vector/cnt
	else:
		None

#####################
#	Elbow 法
#####################
def elbow(X, picpath):
	import matplotlib as mpl
	mpl.use('Agg')
	import matplotlib.pyplot as plt

	distortions = []
	for i  in range(1,11):
		km = KMeans(n_clusters=i, random_state=43, n_jobs=-1)
		km.fit(X)
		distortions.append(km.inertia_)	# km.fitするとkm.inertia_が得られる

	plt.figure()	# 新規ウィンドウ
	plt.plot(range(1,11), distortions,marker='o')
	plt.xlabel('Number of clusters')
	plt.ylabel('Distortion')
	plt.savefig(picpath)

#######################
#	silhouette analysis
#######################
def silhouette(X, n_clus):
	from sklearn.metrics import silhouette_samples
	import matplotlib as mpl
	mpl.use('Agg')
	import matplotlib.pyplot as plt
	plt.figure()	# 新規ウィンドウ


	from matplotlib import cm

	km = KMeans(n_clusters=n_clus, random_state=43, n_jobs=-1)
	y_km = km.fit_predict(X)

	cluster_labels = np.unique(y_km)

	# シルエット係数を計算
	silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
	y_ax_lower, y_ax_upper= 0,0
	yticks = []

	for i,c in enumerate(cluster_labels):
		c_silhouette_vals = silhouette_vals[y_km==c]
		c_silhouette_vals.sort()
		y_ax_upper += len(c_silhouette_vals)
		color = cm.jet(float(i)/n_clus)
		plt.barh(range(y_ax_lower,y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=color)
		yticks.append((y_ax_lower+y_ax_upper)/2)
		y_ax_lower += len(c_silhouette_vals)

	silhouette_avg = np.mean(silhouette_vals)
	plt.axvline(silhouette_avg,color="red",linestyle="--")
	plt.yticks(yticks,cluster_labels + 1)
	plt.ylabel('Cluster')
	plt.xlabel('silhouette coefficient')
	plt.savefig('./silhouette'+str(n_clus)+'.png')
	# plt.show()

def makeVectors(model, mwelist):
	vectors, dellist = [], []
	for tmp in mwelist:
		mwes = []
		for mwe in tmp.split(','):
			mwe = mwe.split('[')
			mwe[1] = mwe[1].replace(']', '')
			mwes += mwe

		vector = get_wv(list(set(mwes)), model)
		if vector is not None:
			vectors.append(vector)
		else:
			dellist.append(tmp)

	for delmwe in dellist:
		mwelist.remove(delmwe)

	return mwelist, vectors

#################
#	引数処理
#################
def get_arg():
	parser = argparse.ArgumentParser(description='converter')
	parser.add_argument('-dim','--dim', type=str, default=100)
	# parser.add_argument('-rstate','--rstate', type=int, default=43)
	args = parser.parse_args()
	return args

#######################
#	main
#######################
if __name__ == '__main__':
	args = get_arg()
	vecdim = args.dim
	datapath = '../../../data/MUST-dist-1.0/data/'
	vecpath = './result/mwe_vec.pickle'
	mwepath = '../nn/result/mwelist.pickle'
	modelpath = '../nn/result/mwevector.model'
	wakatipath = './result/MUST1_wakati.txt'

	# wakatilist, mwelist = parseDB(datapath)
	# saveWakati(wakatipath, wakatilist)
	# model = trainw2v(modelpath, wakatipath, vecdim)
	# model.save(modelpath)		# save
	# data.save(mwepath, mwelist)

	# mwelist, vectors = makeVectors(model, mwelist)
	# data.save(mwepath, mwelist)
	# data.save(vecpath, vectors)


	#############
	# 最適なクラスタ数を考える
	#############
	# picpath = './elbow.png'
	# vectors = data.load(vecpath)
	# elbow(vectors, picpath)
	# for i in range(2, 7):
	# 	silhouette(vectors, i)

	#############
	# k-means
	#############
	n_cluster = 4
	print('n_cluster:', n_cluster)
	mwelist = data.load(mwepath)
	vectors = data.load(vecpath)
	km = KMeans(n_clusters=n_cluster, random_state=43, n_jobs=-1)
	km.fit(vectors)

	cluster_labels = km.labels_
	cluster_to_words = defaultdict(list)
	for cluster_id, word in zip(cluster_labels, mwelist):
		cluster_to_words[cluster_id].append(word)

	for i, cluslist in cluster_to_words.items():
		cluspath = './result/clusterlist_'+str(i+1)+'.pickle'
		data.save(cluspath, cluslist)
		print('-------------------')
		print(len(cluslist))
		# for clus in cluslist:
		# 	print(clus)





