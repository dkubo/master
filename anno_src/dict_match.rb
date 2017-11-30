#coding:utf-8

require './data'
require 'natto'
require 'json'
require 'csv'


# カンマ区切りされているデータがある⇒分割

def splitCont(mwe, start_idx, sentence)
	precont, matched, postcont = sentence[0..start_idx-1].join(), 
									sentence[start_idx..start_idx+mwe.length-1].join(), 
									sentence[start_idx+mwe.length..-1].join()
	startlen, endlen = precont.length+1, precont.length + matched.length
	return precont, matched, postcont, startlen, endlen
end

def matching(mwe, s_id, sentence, outdata)
	m_frg, totallen = 0, 0
	sentence.each_with_index do |sent, idx|  	# sentence loop
		if mwe[0] == sent 	# 文の形態素とMWEの形態素が一部マッチ
			m_frg, start_idx = 1, idx
			startlen, endlen = totallen+1, totallen
			for cnt in 0..mwe.length-1 do
				if sentence[idx+cnt] != mwe[cnt]
					m_frg = 0
					break
				else
					endlen += sentence[idx+cnt].length
				end
			end

			if m_frg == 1
				precont, matched, postcont, startlen, endlen = splitCont(mwe, start_idx, sentence)
				outdata.push([s_id, startlen.to_s, endlen.to_s, precont, matched, postcont])	# startlen, endlen: 標準形の文に対してのもの
				m_frg = 0
			end
		end
		totallen += sent.length
	end

	return outdata
end

def proc(sent_hash, mwelist, outdata)
	nm = Natto::MeCab.new
	for mwe in mwelist
		mwe = mweParse(nm, mwe)
		if mwe.length >= 2
			sent_hash.each{|s_id, v|
				sentence, lemma, sentpos = v
				outdata = matching(mwe, s_id, sentence, outdata)
			}
		end
	end
	return outdata
end


# 機能表現を形態素解析
def mweParse(nm, mwe)
	mlist = []
	nm.parse(mwe) do |n|
		mlist.push(n.surface)
	end
	return mlist[0..-2]
end


def main()
	data = ProcData.new()
	outdata = []
	tocorp = "./bccwj/core_SUW.txt"
	result = "./result/matched_suw.tsv"

	mwelist = CSV.read('./result/mwelist.csv')
	mwelist = data.mweMatchHash(mwelist)
# 	# open the corpus
	sent_hash = data.splitSentence(tocorp, "bccwj")	# 59432 
	outdata = proc(sent_hash, mwelist, outdata)
	outdata = data.writeCSV(result, outdata)
end

main()







