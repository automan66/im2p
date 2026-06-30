import torch
# import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
import h5py
import json
import pickle
import sys

from config import opt

S_max = opt.S_max
N_max = opt.N_max
adj_lable = [-1, 1, 1, 1, 1, 3, 1, 2, 1, 2, 1,
3, 3, 3, 3, 3, 4, 4, 3, 3, 2,
3, 1, 1, 3, 3, 3, 4, 3, 1, 2,
1, 1, 1, 3, 3, 2, 3, 3, 3, 3,
3, 1, 1, 3, 3, 3, 3, 2, 2, 2] #-1 for background, 4 classes
def load_info(info_file):
    """
    Loads the file containing the visual genome label meanings
    :param info_file: JSON
    :return: ind_to_classes: sorted list of classes
             ind_to_predicates: sorted list of predicates
    """
    info = json.load(open(info_file, 'r'))
    info['label_to_idx']['__background__'] = 0
    info['predicate_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])

    return ind_to_classes, ind_to_predicates

class VGDataset(data.Dataset):
	"""docstring for VGDataset"""
	def __init__(self, rcnn_feats, id_path, feat_path, caption_path, vocab):

		super(VGDataset, self).__init__()
		self.rcnn_feats = rcnn_feats
		self.feat_path = feat_path
		self.img_ids = json.load(open(id_path, 'r'))
		self.para = pickle.load(open(caption_path, 'rb'))
		self.vocab = vocab

	def __getitem__(self, index):
		'''Returns one data pair (image features and caption list).'''
		vocab = self.vocab
		img_name = self.img_ids[index]
		feats = torch.zeros(opt.num_boxes, 4096)
		total_data = h5py.File(self.rcnn_feats+str(img_name)+'.h5', 'r')
		tmp_feat = total_data['obj_fmap'][:]
		feats = torch.FloatTensor(tmp_feat)
		obj = total_data['objs'][:]
		# dc_feat = h5py.File(self.feat_path+str(img_name)+'.h5', 'r')
		# dc_feat = torch.FloatTensor(dc_feat['feats'][:])
		rels = total_data['rels'][:]
		pred = total_data['pred_ind'][:]
		visrep = total_data['vis_rep'][:]
		both_rel_id = total_data['both_rel_id'][:]
		rels_pred = np.concatenate((rels, pred[:, np.newaxis]), axis=1)

		rels = rels[both_rel_id]
		rels_pred = rels_pred[both_rel_id]
		adj_mat = np.zeros([4, opt.num_boxes, opt.num_boxes])
		# adj_mat[rels] = 1
		tmp_pred = rels_pred[:, 2]
		rels_pred[:, 2] = [adj_lable[k] for k in tmp_pred]
		for t in range(4):
			lable_set = rels_pred[:, 2]==(t+1)
			if lable_set.sum()>0:
				rels_t = (rels[lable_set].T).tolist()
				adj_mat[t][rels_t] = 1

		adj_mat = torch.FloatTensor(adj_mat)

		rel_num = int(len(rels)/2)
		rels = rels[:rel_num]
		rel_mat = np.zeros([rel_num, rel_num])
		for i in range(rel_num):
			for j in range(i+1, rel_num):
				if rels[i,0]==rels[j,0] or rels[i,0]==rels[j,1] or rels[i,1]==rels[j,0] or rels[i,1]==rels[j,1]:
					rel_mat[i, j] = 1
					rel_mat[j, i] = 1
		rel_mat = torch.FloatTensor(rel_mat)

		rels = torch.LongTensor(rels)
		visrep = torch.FloatTensor(visrep[:rel_num])
		prod_rep = feats[rels[:, 0]] * feats[rels[:, 1]]
		prod_rep = prod_rep * visrep  #(40, 4096)

		sent_len = self.para[img_name][0]
		captions = self.para[img_name][1]
		# print self.para[img_name]
		sent_state = torch.zeros(S_max)
		if sent_len > S_max:
			sent_len = S_max
		sent_state[(sent_len-1):] = 1
		cap2num = []
		lens = []
		# Convert caption (string) to word ids.
		for i in range(sent_len):
			tokens = nltk.tokenize.word_tokenize(str(captions[i]).lower()) 
			cap_tmp = []
			cap_tmp.append(vocab('<start>'))
			cap_tmp.extend([vocab(token) for token in tokens])
			cap_tmp.append(vocab('<end>'))
			cap2num.append(torch.Tensor(cap_tmp))
			lens.append(len(cap_tmp))

		return feats, sent_state, lens, max(lens), cap2num, adj_mat, prod_rep, rel_mat

	def __len__(self):
		return len(self.img_ids)


def collate_fn(data):  
	feats, sent_state, lens, max_sent_len, cap2num, adj_mat, prod_rep, rel_mat = zip(*data)
	rel_mat = torch.stack(rel_mat)
	prod_rep = torch.stack(prod_rep)
	adj_mat = torch.stack(adj_mat, 0)
	feats = torch.stack(feats, 0)
	batch_size = feats.size(0)
	batch_sent_state = torch.stack(sent_state, 0).long()
	# print batch_sent_state
	targets = torch.zeros(batch_size, S_max, N_max).long()
	cap_mask = torch.zeros(batch_size, S_max, N_max)
	for i, cap in enumerate(cap2num):
		for j, sent in enumerate(cap):
			end = min(N_max,lens[i][j])
			targets[i, j, :end] = sent[:end]
			cap_mask[i, j, :end] = 1
	
	return feats, batch_sent_state, targets, cap_mask, adj_mat, prod_rep, rel_mat


def get_loader(rcnn_feats, id_path, feat_path, caption_path, vocab, batch_size, shuffle, num_workers):
	im2p_data = VGDataset(rcnn_feats, id_path, feat_path, caption_path, vocab)
	dataloader = torch.utils.data.DataLoader(dataset=im2p_data, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
	return dataloader
