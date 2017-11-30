# coding: utf-8

if __name__ == '__main__':
	with open('train_Lu.tsv', 'r') as f:
		for line in f:
			if len(line.split('\t')) not in [1, 11]:
				print(line)