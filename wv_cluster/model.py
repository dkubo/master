
from chainer import Chain, Variable, cuda
from chainer import links as L
from chainer import functions as F
from chainer.training import extensions
import chainer
import numpy as np
import cupy

###############
#	FFNN
###############
class FFNN(chainer.Chain):
	def __init__(self, n_vocab, embed_size, n_fvocab, n_units, n_labels, w2vfrg, embeddings):
		if w2vfrg == 0:
			embeddings = None
		# n_vocab, embed_size = embeddings.shape
		super(FFNN, self).__init__(
			embed = L.EmbedID(n_vocab, embed_size, initialW=embeddings, ignore_label=-1),
			f_embed = L.EmbedID(n_fvocab, 50, ignore_label=-1),
			# f_embed = L.EmbedID(n_fvocab, embed_size, ignore_label=-1),
			m_embed = L.EmbedID(2, 10, ignore_label=-1),
			l1 = L.Linear(None,  n_units),
			l2 = L.Linear(None, 50),
			l3 = L.Linear(None, n_labels)
		)

	def __call__(self, x):
		# print('x.shape: ', x.shape)		# (batchsize, 3(morph, feat, mark), seqlen)
		x = F.transpose(x, axes=(1, 0, 2))
		# print('x[0].shape: ', x[0].shape)
		h0 = F.concat((self.embed(x[0]), self.f_embed(x[1]), self.m_embed(x[2])), axis=2)
		# print('h0.shape: ', h0.shape)
		h1 = self.l1(h0)
		# print('h1.shape: ', h1.shape)
		h2 = self.l2(h1)
		# predict = self.l2(h1)
		predict = self.l3(h2)
		return predict

###############
#	BiLSTM
###############
class BiLSTM(Chain):
	def __init__(self, n_vocab, embed_size, n_fvocab, n_units, n_labels, w2vfrg, embeddings):
		# n_vocab, embed_size = embeddings.shape
		if w2vfrg == 0:
			embeddings = None
		super(BiLSTM, self).__init__(
			embed = L.EmbedID(n_vocab, embed_size, initialW=embeddings, ignore_label=-1),
			f_embed = L.EmbedID(n_fvocab, embed_size, ignore_label=-1),
			m_embed = L.EmbedID(2, 10, ignore_label=-1),
			bilstm = L.NStepBiLSTM(1, embed_size, embed_size, dropout=1),
			# bigru = L.NStepGRU(1, embed_size, embed_size, dropout=1),
			l2 = L.Linear(None, n_units),
			l3 = L.Linear(None, 50),
			l4 = L.Linear(None, n_labels)
		)
		self.reset_state()

	# def reset_state(self):
	# 	self.bilstm.reset_state()
	def reset_state(self):
		self.cx = self.hx = None

	def __call__(self, xs):
		# self.reset_state()
		h0_f, ys_f = [], []
		xs = F.transpose(xs, axes=(1, 0, 2))
		# h0 = F.concat((self.embed(xs[0]), self.f_embed(xs[1])), axis=1)
		h0 = F.concat((self.embed(x[0]), self.f_embed(x[1]), self.m_embed(x[2])), axis=2)
		# print(h0.shape)			# (batchsize, seqsize*2, embedsize)
		for x in h0:
			h0_f.append(x)

		hy, cy, ys = self.bilstm(self.hx, self.cx, h0_f)
		# hy, ys = self.bigru(self.hx, h0_f)

		# self.hx, self.cx = hy.to_gpu(), cy.to_gpu()
		self.hx = hy.to_gpu()

		for ys_s in ys:
			ys_f.append(ys_s.data)

		ys_f = self.xp.array(ys_f, dtype=self.xp.float32)

		h1 = self.l2(ys_f)
		h2 = self.l3(h1)
		predict = self.l4(h2)
		return predict




####################
#	CNN
####################
class CNN(Chain):
	def __init__(self, fil_num, n_label, embeddings):
		n_vocab, embed_size = embeddings.shape

		super(CNN, self).__init__(
			embed = L.EmbedID(n_vocab, embed_size, initialW=embeddings, ignore_label=-1),
			conv1 = L.Convolution2D(1, fil_num, ksize=(3, 1)),	# (window size, 1)
			l1    = L.Linear(None,  n_label),
		)
    
	def __call__(self, x):
		# print('---------------')
		# print(x.shape)	# (batchsize, seqsize)
		h0 = self.embed(x)
		# print(h0.shape)	# (batchsize, seqsize, embedsize)
		# (nsample, channel, height(=maxlen), width(=embed_size)) の4次元テンソルに変換
		h0 = h0.reshape(h0.shape[0], 1, h0.shape[1], h0.shape[2])
		# print(h0.shape)	# (batchsize, 1, seqsize, embedsize)
		h1 = self.conv1(h0)
		# print(h1.shape)	# (batchsize, fil_num, convsize, 1)
		h2 = F.max_pooling_2d(h1, ksize=(h1.shape[2], 1))	# (convsize, 1)
		# print(h2.shape)		# (batchsize, 1, 1, embedsize)
		y = self.l1(h2)
		# print(y.shape)		# (1, 2)
		return y




