# coding: utf-8

from collections import defaultdict, Counter
import pickle

"""
・マッチしたMWEカウントを、分野ごとに整理
・マッチしたMWEが，長単位で分割されてるかどうかカウント
"""

##########################
# 中間ファイル出力
##########################
def pickleFunc(fname, senthash, type):
	if type == "save":
		with open(fname, mode='wb') as f:
			pickle.dump(senthash, f)
	elif type == "load":
		with open(fname, mode='rb') as f:
			return pickle.load(f)

##########################
# 長単位解析
##########################X
def parseLUW(luwpath):
	baseid, cnt, fieldid, label = 1, 1, "", "B"
	lemmasent, sentence, sentpos, senthash = [], [], [], {}

	with open(luwpath, 'r') as f:
		for line in f:
			l = line.rstrip().split("\t")
			# genbun = l[23] 	# この情報おかしい
			fieldid, label, lemma, pos, katuyogata, katuyokei, sentpart = l[1], l[-1], l[8], l[11], l[12], l[13], l[16]
			if cnt == 1:
				label = "I"

			if label == "I":
				lemmasent.append(lemma)
				sentence.append(sentpart)
				sentpos.append([pos, katuyogata, katuyokei])
			elif label == "B":
				sentid = fieldid + "_" + str(baseid)
				senthash[str(sentid)] = [sentence, lemmasent, sentpos]
				baseid += 1
				lemmasent, sentence = [lemma], [sentpart]
				sentpos = [[pos, katuyogata, katuyokei]]
			cnt += 1

		sentid = fieldid + "_" + str(baseid)
		senthash[str(sentid)] = [sentence, lemmasent, sentpos]
	return senthash

##########################################
# 機能的用法かどうか調べる(長単位で一つになってれば機能的用法とする)
# ※純粋に機能的用法だけど，長単位になってないものもある
##########################################
def checkUsage(buf, senthash):
	totallen = 0
	sentid, s, e = buf
	for k, v in senthash.items():
		if k == sentid:
			for (sent, pos) in zip(v[0], v[2]):
				if int(s) == (totallen + 1) and int(e) == (totallen+len(sent)):
					# print("---------------------")
					# print(sentid, s, e)
					# print(sent, pos)
					return 1
				totallen += len(sent)
	return 0

def main():
	srcpath = "../../result/bccwj/bccwj_matced_0128_edited.tsv"
	luwpath = "../../data/20161007/corpus/bccwj/core_LUW.txt"

	# usage = {}
	# # 機能的用法
	# usage["func"] = {"PB":defaultdict(int), "PM":defaultdict(int), "PN":defaultdict(int), "OW":defaultdict(int), "OC":defaultdict(int), "OY":defaultdict(int)}
	# # 内容的用法もしくは長単位ではない
	# usage["other"] = {"PB":defaultdict(int), "PM":defaultdict(int), "PN":defaultdict(int), "OW":defaultdict(int), "OC":defaultdict(int), "OY":defaultdict(int)}

	# # senthash = parseLUW(luwpath)
	# # pickleFunc("luw.pickle", senthash, "save")
	# senthash = pickleFunc("luw.pickle", None, "load")

	# with open(srcpath, 'r') as f:
	# 	for line in f:
	# 		buf = line.rstrip()[1:-1].split("\t")
	# 		pos, mwe = buf[0][0:2], buf[4]

	# 		value = checkUsage(buf[0:3], senthash)
	# 		if value == 1:	# 複合辞用法であった場合
	# 			usage["func"][pos][mwe] += 1
	# 		else:
	# 			usage["other"][pos][mwe] += 1

	# pickleFunc("usage.pickle", usage, "save")

	usagehash = pickleFunc("usage.pickle", None, "load")
	merged = {"PB":defaultdict(int), "PM":defaultdict(int), "PN":defaultdict(int), "OW":defaultdict(int), "OC":defaultdict(int), "OY":defaultdict(int)}

	# 分野ごとに頻度統合（機能的用法かどうかは無視）
	for usage, freqhash in usagehash.items():
		 for field, mwehash in freqhash.items():
		 	merged[field] = dict(Counter(merged[field]) + Counter(mwehash))	# funcとotherの頻度辞書マージ

	output = defaultdict(list)
	for field, freqhash in merged.items():
		# print(field, sum(freq.values()))	# 分野ごと頻度
		 	mwetuple = sorted(freqhash.items(), key=lambda x:x[1], reverse=True)	# 頻度降順でソート
		 	# 上位20表現抽出
		 	for mwe, freq in mwetuple[0:20]:
			 	output[field].append([mwe, usagehash["func"][field][mwe], usagehash["other"][field][mwe]])

	for k, v in output.items():
		print(k)
		print(v)

if __name__ == '__main__':
	main()

