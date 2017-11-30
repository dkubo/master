#coding: utf-8
require 'json'
require 'csv'

class ProcData
	def initialize()
		@mwelist = Array.new()
		@const_hash = Hash.new()
	end

	def getmwe(todict)
		data_hash = JSON.parse(File.read(todict))
		data_hash.each{|mweid, value|
			# @mwelist.push([mweid, value["suw_lemma"], value["left"], value["meaning"]])
			@mwelist.push([mweid, value["suw_lemma"], value["left"]])
			for mwe in value["variation_lemma"]
				# mwelist.push([mweid, mwe, value["left"], value["meaning"]])
				@mwelist.push([mweid, mwe, value["left"]])
			end
		}
		@mwelist.uniq!
		return @mwelist
	end

	def makeArray(v)
		sentlist, lemlist, poslist, vari1, vari2, sentence, lemma, sentpos = [], [], [], [], [], [], [], []

		v.each{|line|
			sentlist.push(line[8].split(",", -1))
			lemlist.push(line[9].split(",", -1))
			poslist.push(line[10].split(",", -1))
			if line[11] == "" or line[11].split(",", -1).length == 1
				vari1.push(line[11])
			else
					vari1 += line[11].split(",", -1)
			end
			if line[12] == "" or line[12].split(",", -1).length == 1
				vari2.push(line[12])
			else
				vari2 += line[12].split(",", -1)
			end
		}

		sentlist.flatten!
		lemlist.flatten!
		poslist.flatten!

		for i in 0..lemlist.length-1
			if /-.*/ =~ sentlist[i]
				sentence.push($`)
			else
				sentence.push(sentlist[i])
			end
			if /-.*/ =~ lemlist[i]
				lemma.push($`)
			else
				lemma.push(lemlist[i])
			end
			sentpos.push([poslist[i], vari1[i], vari2[i]])
		end

		return sentence, lemma, sentpos
	end


	# コーパスを一文ずつに分解する
	def splitSentence(tocorp, type)
		sent_hash, lasthash = Hash.new(), Hash.new()
		suw_list, pre_sentid, sentid = [], "", ""
		lemmasent, sentence, sentpos = [], [], []
		corpus = open(tocorp, 'r')

		if type == "ud"
			corpus.each_line{|l|
				l = l.chomp.split("\t")
				if /# SENT-ID: / =~ l[0]
					sentid = $'
					if suw_list != []
						sent_hash[pre_sentid] = suw_list
						suw_list = []
					end
					pre_sentid = sentid
				else
					suw_list.push(l) unless l == []
				end
			}
			sent_hash[pre_sentid] = suw_list
			sent_hash.each{|s_id, v|
				sentence, lemma, sentpos = makeArray(v)
				lasthash[s_id] = [sentence, lemma, sentpos]
			}
		elsif type == "bccwj"
			baseid, cnt, fieldid, label = 1, 1, "B"
			corpus.each_line{|l|
				l = l.chomp.split("\t")
				# genbun = l[23] 	# この情報おかしい
				fieldid, label, lemma, pos, katuyogata, katuyokei, sentpart = l[1], l[9], l[12], l[16], l[17], l[18], l[22]
				if cnt == 1
					label = "I"
				end
				if label == "I"
					lemmasent.push(lemma)
					sentence.push(sentpart)
					sentpos.push([pos, katuyogata, katuyokei])
				elsif label == "B"
					sentid = fieldid + "_" + baseid.to_s
					lasthash[sentid.to_s] = [sentence, lemmasent, sentpos]
					baseid += 1
					lemmasent, sentence = [lemma], [sentpart]
					sentpos = [[pos, katuyogata, katuyokei]]
				end
				cnt += 1
			}
			sentid = fieldid + "_" + baseid.to_s
			lasthash[sentid.to_s] = [sentence, lemmasent, sentpos]
		end
		return lasthash
	end

def mweMatchHash(mwelist)
	matchlist = []
	for mwe in mwelist
		mwes = []
		for mwepart in mwe
			mwepart = mwepart.split('[')
			mwepart[1] = mwepart[1].gsub(']', '')
			mwes += mwepart
		end
		mwes.uniq!
		matchlist += mwes
	end
	matchlist.uniq!
	return matchlist
end

	# MWEID, 文ID, 開始文字位置, 終了文字位置, 前文脈, 対象表現, 後文脈, MEANING
	def	writeCSV(fname, outdata)
		file = CSV.open(fname, 'w')
		outdata.each{|data| file.puts data }
		file.close
		outdata = []
		return outdata
	end

end