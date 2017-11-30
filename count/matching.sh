# 辞書(1語除去済み)でマッチング
# ruby dict_match.rb -ud ../../result/tsutsuji_dic_20161215.json
# 入れ子排除，ソート
python check_result.py -ud 
# 品詞集約(8→4)，内容語のみを抽出
# python extract_result.py -ud