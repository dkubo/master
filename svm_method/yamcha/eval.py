# coding: utf-8

# 単純なミス：用法の間違い（機能的用法と内容的用法）
# チャンクスパンによるミス：用法は当てているが、スパンが間違っている
def checkSpan(result, miss, spanmiss):
	fam_list = ['B-FAM', 'I-FAM']
	gold, pred = [], []
	span, sent = [], {}

	for i, morph in enumerate(result):
		if len(morph) != 0:
			sent[i] = morph
			gold_tag, pred_tag = morph[-2:]
			# goldのチャンクのインデックス取得
			if gold_tag in fam_list:	# 検出器FAM
				gold.append(gold_tag)
				pred.append(pred_tag)
				span.append(i)
		else:	# 文区切り
			if gold != []:
				if gold != pred:
					miss += 1
					if 'B-FAM' in pred or 'I-FAM' in pred:
						print('-------------')
						for s in range(span[0]-2, span[-1]+2):
							print(sent[s])
						print(gold)
						print(pred)
			gold, pred = [], []
			span, sent = [], {}
	return miss, spanmiss

# ミスしたやつ抽出→その中でスパンミスによるものを抽出
# スパンミス => Iの長さでミスってる?
if __name__ == '__main__':
	miss, spanmiss = 0, 0
	for i in range(0, 10):
		print('========================')
		print(i)
		resultname = './result/result_'+str(i)+'.tsv'
		result = parse.openTSV(resultname)
		miss, spanmiss = checkSpan(result, miss, spanmiss)

	# miss_ratio = spanmiss / miss
	# print('span miss = ', spanmiss, '/', miss, '=', miss_ratio)
