# coding: utf-8

from collections import defaultdict, Counter
import pickle, argparse
import data

"""
・MUST上で曖昧性のある機能表現が、BCCWJの短単位データ上でマッチした	文リスト		
・マッチしたMWEカウントを、分野ごとに整理
・マッチしたMWEが，長単位で分割されてるかどうかカウント
"""


###############
#	引数処理
###############
def get_arg():
	parser = argparse.ArgumentParser(description='converter')
	parser.add_argument('-dtype','--dtype', type=int, default=1)
	args = parser.parse_args()
	return args

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
			for sent in v[0]:
			# for (sent, pos) in zip(v[0], v[2]):
				if int(s) == (totallen + 1) and int(e) == (totallen+len(sent)):
					return 1
				totallen += len(sent)
	return 0

def main():
	args = get_arg()
	if args.dtype == 1:
		# MUST上で曖昧性のある機能表現が、BCCWJの短単位データ上でマッチした	文リスト		
		srcpath = "./result/matched_suw.tsv"
		luwpath = "./result/bccwj/core_LUW.txt"

		usage = {}
		# 機能的用法
		usage["func"] = {"PB":defaultdict(int), "PM":defaultdict(int), "PN":defaultdict(int), "OW":defaultdict(int), "OC":defaultdict(int), "OY":defaultdict(int)}
		# 内容的用法もしくは長単位ではない
		usage["other"] = {"PB":defaultdict(int), "PM":defaultdict(int), "PN":defaultdict(int), "OW":defaultdict(int), "OC":defaultdict(int), "OY":defaultdict(int)}

		# senthash = parseLUW(luwpath)
		# pickleFunc("luw.pickle", senthash, "save")
		senthash = pickleFunc("./result/luw_senthash.pickle", None, "load")

		with open(srcpath, 'r') as f:
			for line in f:
				if args.dtype == 1:
					buf = line.rstrip().split(",")
					pos, mwe = buf[0][0:2], buf[4]
				elif args.dtype == 2:
					buf = line.rstrip()[1:-1].split("\t")
					pos, mwe = buf[0][0:2], buf[4]

				value = checkUsage(buf[0:3], senthash)
				if value == 1:	# 複合辞用法であった場合
					usage["func"][pos][mwe] += 1
				else:
					usage["other"][pos][mwe] += 1

		pickleFunc("./result/usage1.pickle", usage, "save")
		# アノテーションデータ1
		usagehash = pickleFunc("./result/usage1.pickle", None, "load")
	elif args.dtype ==2:
		# アノテーションデータ2
		usagehash = pickleFunc("./result/usage2.pickle", None, "load")
		must_mwe = data.load('../nn/result/mwelist.pickle')	# MUSTにあるMWEは除外する
		must_mwe, musthash = data.mweMatchHash(must_mwe)

	merged = {"PB":defaultdict(int), "PM":defaultdict(int), "PN":defaultdict(int), "OW":defaultdict(int), "OC":defaultdict(int), "OY":defaultdict(int)}
	mwefreq = defaultdict(lambda: defaultdict(int))	# mweごと用法カウント
	mwelist = []
	
	# 分野ごとに頻度統合（用法ごとの頻度をマージ）
	for usage, freqhash in usagehash.items():
		for field, mwehash in freqhash.items():
			merged[field] = dict(Counter(merged[field]) + Counter(mwehash))	# funcとotherの頻度辞書マージ

	for field, freqhash in merged.items():
		mwetuple = sorted(freqhash.items(), key=lambda x:x[1], reverse=True)
		for mwe, freq in mwetuple:
			mwefreq[mwe]['func'] += usagehash["func"][field][mwe]
			mwefreq[mwe]['other'] += usagehash["other"][field][mwe]

	for mwe, yoho in mwefreq.items():
		totalfreq = yoho["func"] + yoho["other"]
		funcratio = yoho["func"] / totalfreq
		# if totalfreq >= 50:	# 出現頻度の下限
		# if (0.1 <= funcratio <= 0.9) and (mwe not in must_mwe):
		if args.dtype == 1:
			# if (0.2 <= funcratio <= 0.8):
			mwelist.append(mwe)
		elif args.dtype == 2:
			if (0.2 <= funcratio <= 0.8) and (mwe not in must_mwe):
				mwelist.append(mwe)

	print(args.dtype, mwelist)
	if args.dtype == 1:
		# アノテーションデータ1
		data.save('./result/mwelist1.pickle', mwelist)
	if args.dtype == 2:
		# アノテーションデータ2
		data.save('./result/mwelist2.pickle', mwelist)

if __name__ == '__main__':
	main()
