import torch
import time
import os
from batch_loader import get_loader
from torch.autograd import Variable
import numpy as np
import random
from build_vocab import Vocabulary
import pickle
from model import DecoderRNN
import torch.nn as nn
from config import opt
import json
from torch.nn.utils.rnn import pack_padded_sequence
import sys
from pycocoevalcap.eval2 import COCOEvalCap

LOG_DIR = 'log'
MODEL_DIR = 'model'
if not os.path.exists(LOG_DIR):
	os.mkdir(LOG_DIR)
if not os.path.exists(MODEL_DIR):
	os.mkdir(MODEL_DIR)

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

with open(opt.vocab_path, 'rb') as f:
	vocab = pickle.load(f)

bias_init = np.loadtxt(opt.bias_init_path)

batch_size = opt.batch_size
lr_init = opt.lr_init
model = DecoderRNN(len(vocab), bias_init)
model = model.cuda()

criterion = nn.CrossEntropyLoss(reduce=False)
f = lambda x: Variable(x).cuda()
	
train_data = get_loader(opt.train_feats_dir, opt.train_id_path, opt.train_dense_caption, opt.caption_path, vocab, batch_size, True, 8)
test_data = get_loader(opt.test_feats_dir, opt.test_id_path, opt.test_dense_caption, opt.caption_path, vocab, 1, False, 8)

lr = lr_init
img_ids = json.load(open(opt.test_id_path, 'r'))

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def train(epoch):
	model.train()
	num_batch = len(train_data)
	filename = 'log/epoch_{}_train.txt'.format(epoch + 1)
	start = time.time()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	for i, (feats, sent_state, targets, mask, adj_mat, visrep, rel_mat) in enumerate(train_data):
		sent_state, targets, mask = f(sent_state), f(targets), f(mask)
		feats = f(feats)
		adj_mat = f(adj_mat)
		visrep = f(visrep)
		rel_mat = f(rel_mat)
		state, outputs, tr = model(targets, feats, adj_mat, visrep, rel_mat)
		targets, mask = targets[:, :, 1:], mask[:, :, 1:]

		sent_loss = criterion(state.view(-1, 2), sent_state.view(-1))
		outputs = outputs.contiguous().view(-1, len(vocab))
		targets = targets.contiguous().view(-1)
		mask = mask.contiguous().view(-1)

		word_loss = criterion(outputs, targets)
		sent_loss = torch.sum(sent_loss) / batch_size
		word_loss = torch.sum(word_loss * mask) / batch_size

		loss = 5*sent_loss + word_loss + tr
		optimizer.zero_grad()
		loss.backward()
		clip_gradient(optimizer, 0.1)
		optimizer.step()
		if (i+1)%10 == 0:
			print('epoch {}: {}/{}, loss: {}'.format(epoch+1, i+1, num_batch, loss.item()))
			fo = open(filename, 'a+')
			fo.write('==> epoch {}: {}/{},  loss: {}\n'.format(epoch + 1, i + 1, num_batch, loss.item()))
			fo.close()
		
	fo = open(filename, 'a+')
	fo.write('time: {}\n'.format(time.time() - start))
	fo.close()


def validate(beam_size, tag, epoch):
	model.eval()
	sample = random.sample(range(len(img_ids)), 200)
	data = {}
	for i, (feats, sent_state, targets, mask, adj_mat, visrep, rel_mat) in enumerate(test_data):
		if tag == 'val':
			if i not in sample:
				continue
		else:
			print(i)
		img = img_ids[i]
		feats = f(feats)	
		adj_mat = f(adj_mat)
		visrep = f(visrep)
		rel_mat = f(rel_mat)
		state, outputs = model.sample_beam(feats, beam_size, adj_mat, visrep, rel_mat)
		_, state = state[0].max(1)
		state = state.data.cpu().numpy()
		outputs = outputs.cpu().numpy()
		sampled_caption = []
		for j in range(opt.S_max):
			for word_id in outputs[j]:
				word = vocab.idx2word[word_id]

				if word == '<end>':
					sampled_caption.append('.')
					break
				else:
					sampled_caption.append(word)
			if state[j] == 1:
				break
		sentence = ' '.join(sampled_caption)
		data[img] = []
		data[img].append({'image_id':img, 'caption':sentence})
		

	jsObj = json.dumps(data)
	if tag == 'val':
		resFile = 'result.json'
	else:
		resFile = f'epoch_{epoch+1}_res/result-{beam_size}.json'
	fileObject = open(resFile, 'w')
	fileObject.write(jsObj)
	fileObject.close()
	cocoEval = COCOEvalCap(opt.ans_file, resFile)
	return cocoEval.evaluate()

	
for i in range(25):
	train(i)
	if (i+1) % 5 == 0:
		for beam in [2, 5, 7]:
			res_dir = f'epoch_{i+1}_res'
			if not os.path.exists(res_dir):
				os.makedirs(res_dir)
			recent_bleu1 = validate(beam, 'test', i)
		torch.save(model.state_dict(), f'model_{i+1}_{recent_bleu1}.pth')

