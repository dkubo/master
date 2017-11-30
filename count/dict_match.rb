#coding:utf-8

require './data'
# require 'natto'
require 'json'
require 'csv'

# for ITC
# MYDIC = "../../result/tsutsuji_dic_20161215.json"	# 先頭の「う」と「ん」を削除, ハイフン処理, エントリ追加をした辞書

CONST1="../const/const1_unidic.tsv"
CONST2="../const/const2.tsv"


# カンマ区切りされているデータがある⇒分割

def splitCont(mwe, start_idx, sentence)
	precont, matched, postcont = sentence[0..start_idx-1].join(), 
																sentence[start_idx..start_idx+mwe.length-1].join(), 
																sentence[start_idx+mwe.length..-1].join()
	startlen, endlen = precont.length+1, precont.length + matched.length
	return precont, matched, postcont, startlen, endlen
end

# def matching(mweid, mwe, leftconst, meaning, s_id, sentence, lemma, sentpos, consthash, outdata)
def matching(mweid, mwe, leftconst, s_id, sentence, lemma, sentpos, consthash, outdata)
	m_frg, leftconst, totallen = 0, leftconst[0].scan(/.{1,2}/), 0

	lemma.each_with_index do |lempart, idx|  	# sentence loop
		if mwe[0] == lempart 	# 文の形態素とMWEの形態素が一部マッチ
			m_frg, start_idx = 1, idx
			startlen, endlen = totallen+1, totallen
			for cnt in 0..mwe.length-1 do
				if lemma[idx+cnt] != mwe[cnt]
					m_frg = 0
					break
				else
					endlen += lemma[idx+cnt].length
				end
			end

			# 品詞等の制約を確認			
			constlist = consthash[leftconst[0]]		# leftconst[1]は全て"90"
			if m_frg == 1
				m_frg = constCheck(constlist, sentpos[start_idx-2], start_idx, m_frg)
				m_frg = constCheck(constlist, sentpos[start_idx-1], start_idx, m_frg) unless m_frg != 0
			end
			# MWEが完全にマッチしたとき (制約を満たしている)
			if m_frg == 1
				precont, matched, postcont, startlen, endlen = splitCont(mwe, start_idx, sentence)		# 出現形の文に対してのstartlen等を取得
				outdata.push([mweid, s_id, startlen.to_s, endlen.to_s, precont, matched, postcont])	# startlen, endlen: 標準形の文に対してのもの
				# outdata.push([mweid, s_id, startlen.to_s, endlen.to_s, precont, matched, postcont, meaning])	# startlen, endlen: 標準形の文に対してのもの
				m_frg = 0
			end
		end
		totallen += lempart.length
	end

	return outdata
end

# 制約リスト, 確認対象の品詞等
def constCheck(constlist, sentleft, start_idx, m_frg)	# (辞書側の制約, 文内の品詞等, マッチフラグ)
	constlist.each{|const|
		check = 1
		const = const[0..-2]	# 原形の制約は無視
		const.zip(sentleft).each{|part, pos|
			# 品詞階層の整合性をとる（辞書側の品詞等制約の階層と、実際の文内の品詞等）
			pos = pos.split("-")[0...part.split("-").length].join("-")
			if part == "*" or part == pos 	# "matched!"
				next

			elsif part == "None"	# 文頭に接続詞が来た場合
				if start_idx == 0	# matched!
					next
				else	# not matched!
					check = 0
					break
				end

			else	# not matched!
				check = 0
				break
			end
		}
		if check == 1 then
			m_frg = 1
			break
		else
			m_frg = 0
		end
	}
	return m_frg
end


def proc(sent_hash, mwelist, consthash, outdata)
	sent_hash.each{|s_id, v|
		sentence, lemma, sentpos = v
		# mwelist.each{|mweid, mwe, leftconst, meaning|

		mwelist.each{|mweid, mwe, leftconst|
			outdata = matching(mweid, mwe, leftconst, s_id, sentence, lemma, sentpos, consthash, outdata)
			# outdata = matching(mweid, mwe, leftconst, meaning, s_id, sentence, lemma, sentpos, consthash, outdata)
		}
	}
	return outdata
end

def getpath()
	return ARGV[0], ARGV[1].chomp
end

def main()
	@corptype, todict = getpath()
	data = ProcData.new()
	outdata = []

	# 制約のリスト取得
	consthash = data.getConst(CONST1, CONST2)

	# 各辞書からMWEのリストを取得
	mwelist = data.getmwe(todict)	# [[mweid, mwe, leftconst, meaning], [], ...]
	# p mwelist.length		# 3609

	# open the corpus
	if @corptype == "-ud"
		tocorp = "../../result/ud/parser/mwes.conll"
		sent_hash = data.splitSentence(tocorp, "ud")	# train: 6039, test: , dev: 
		for type in ["train", "test", "dev"] do
			p type
			tocorp = "../../data/20161007/corpus/ud/ja_ktc-ud-#{type}-merged.conll"
			sent_hash = data.splitSentence(tocorp, "ud")	# train: 6039, test: , dev: 
			outdata = proc(sent_hash, mwelist, consthash, outdata)
			result = "../../result/ud/ud_matced_#{type}_1227.csv"
			outdata = data.writeCSV(result, outdata)
		end

	elsif @corptype == "-bccwj"
		tocorp = "../../data/20161007/corpus/bccwj/core_SUW.txt"
		sent_hash = data.splitSentence(tocorp, "bccwj")	# 59432 
		outdata = proc(sent_hash, mwelist, consthash, outdata)
		p outdata
		p outdata.length
		# result = "../../result/bccwj/bccwj_matced_0128.csv"
		# outdata = data.writeCSV(result, outdata)
	end
end

main()