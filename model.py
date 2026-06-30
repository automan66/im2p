#!/usr/bin/env python
# coding=utf-8

__author__ = "Xianglu.Zhu"

import os
import sys
import time
# import matplotlib.pyplot as plt
import pickle
import numpy as np
# import pandas as pd
import random

# import ipdb
from functools import reduce
#import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import opt
from torch.nn.init import kaiming_normal_
# ------------------------------------------------------------------------------------------------------
# Initialization class
#  1. Pooling the visual features into a single dense feature
#  2. Then, build sentence LSTM, word LSTM
# ------------------------------------------------------------------------------------------------------

def kaiming_w(a):
    return nn.Parameter(kaiming_normal_(torch.FloatTensor(a, a)))

class DecoderRNN(nn.Module):
    def __init__(self, n_words, bias_init):
        super(DecoderRNN, self).__init__()
        self.n_words = n_words
        self.num_boxes = opt.num_boxes
        self.feats_dim = 4096
        self.project_dim = 1024
        self.S_max = opt.S_max
        self.N_max = opt.N_max
        self.seq_length = self.N_max
        self.word_embed_dim = 300 # 512?

        self.sentRNN_lstm_dim = 512
        self.sentRNN_FC_dim = 1024
        self.wordRNN_lstm_dim = 512
        # self.beam_size = opt.beam_size

        # regionPooling_W shape: 4096 x 1024 (DxP)
        # regionPooling_b shape: 1024 (P)
        self.regionPooling = nn.Linear(self.feats_dim, self.project_dim)
        self.regionPooling2 = nn.Linear(self.feats_dim, self.project_dim)

        # sentence LSTM
        self.sent_LSTM_1 = nn.LSTM(self.project_dim, self.sentRNN_lstm_dim,batch_first=True)
        self.sent_LSTM = nn.LSTM(self.project_dim, self.sentRNN_lstm_dim,batch_first=True)

        # logistic classifier
        self.logistic = nn.Linear(self.sentRNN_lstm_dim, 2)

        # fc1_W: 512 x 1024, fc1_b: 1024
        # fc2_W: 1024 x 1024, fc2_b: 1024
        self.fc1 = nn.Linear(self.sentRNN_lstm_dim, self.sentRNN_FC_dim)
        self.fc2 = nn.Linear(self.sentRNN_FC_dim, 1024)

        self.fuse_fc1 = nn.Linear(self.project_dim+self.sentRNN_lstm_dim, self.project_dim)
        self.fuse_fc2 = nn.Linear(self.project_dim+self.sentRNN_lstm_dim, self.project_dim)
        self.fuse_fc3 = nn.Linear(self.project_dim*2, self.project_dim)

        # word LSTM
        self.word_LSTM = nn.LSTM(self.word_embed_dim, 512, 2, batch_first=True)
        self.embed = nn.Embedding(self.n_words, self.word_embed_dim)
        embed_init = torch.load(opt.embed_init)
        self.embed.weight.data = embed_init.clone()
        self.emb_w = nn.Parameter(nn.init.uniform_(torch.FloatTensor(self.n_words, self.wordRNN_lstm_dim), -0.1, 0.1))
        self.emb_b = nn.Parameter(torch.FloatTensor(bias_init))
        self.gnn_w = nn.ParameterList([kaiming_w(self.project_dim)]*10)
        self.gnn_b = nn.ParameterList([nn.Parameter(torch.randn(self.project_dim))]*10)

        self.topic_w = kaiming_w(self.project_dim)
        self.topic_b = nn.Parameter(torch.randn(self.project_dim))
        self.topic_w_1 = kaiming_w(self.project_dim)

        att_dim = 512
        self.att_dim = att_dim
        self.w_a = nn.Linear(att_dim, 1)
        self.w_f = nn.Linear(self.project_dim, att_dim)
        self.w_h = nn.Linear(self.sentRNN_lstm_dim, att_dim)

        self.f_beta = nn.Linear(self.sentRNN_lstm_dim, self.project_dim)

    def max_feature(self, feats, adj, rel_mat, visrep): #feats: B x num_boxes x 4096, B*4*num_box*num_box
        tmp_feats = feats.view(-1, self.feats_dim)
        project_vec_all = self.regionPooling(tmp_feats)
        project_vec_all = project_vec_all.view(feats.size(0), opt.num_boxes, self.project_dim)
        

        h_new = []
        for i in range(opt.num_boxes):
            v_i = torch.mm(project_vec_all[:, i, :], self.gnn_w[4]) + self.gnn_b[4]
            for t in range(4):
                project_vec_all = project_vec_all.view(-1, self.project_dim).contiguous()
                g_i = F.sigmoid(torch.mm(project_vec_all, self.gnn_w[t+5]) + self.gnn_b[t+5])
                project_vec_all = project_vec_all.view(-1, opt.num_boxes, self.project_dim).contiguous()
                g_i = g_i.view(project_vec_all.size())
                adj_i = adj[:, t, i].unsqueeze(2).expand(project_vec_all.size())
                tmp = torch.sum(project_vec_all*adj_i*g_i, 1)
                v_i = v_i + torch.mm(tmp, self.gnn_w[t]) + self.gnn_b[t]
            h_new.append(F.relu(v_i))    
        project_vec_all = torch.stack(h_new, 1)

        rep_all = self.regionPooling2(visrep)

        rep_gcn = []
        for j in range(rep_all.size(1)):
        	b = rep_all[:, j, :]
        	mask = rel_mat[:, j]
        	mask_sum = mask.sum(1)
        	mask = mask/mask_sum.unsqueeze(1)
        	a = torch.sum(rep_all*mask.unsqueeze(2), 1)
        	a_ = torch.mm(a, self.topic_w) + self.topic_b + torch.mm(b, self.topic_w_1)
        	rep_gcn.append(F.relu(a_))
        rep_all = torch.stack(rep_gcn, 1)
        rep = torch.mean(rep_all, 1)
        return torch.mean(project_vec_all, 1), project_vec_all, rep

    def topic_net(self, feats, adj, rel_mat, visrep):
        project_vec_0, project_vec_all, rep = self.max_feature(feats, adj, rel_mat, visrep)
        
        project_vec = self.fuse_fc3(torch.cat((project_vec_0, rep), 1))

        inputs = project_vec.unsqueeze(1)

        sent_hid = []
        att_total = []
        h1, c1, h2, c2 = [Variable(torch.zeros(1, project_vec.size(0), self.sentRNN_lstm_dim)).cuda()]*4
        v = self.w_f(project_vec_all)
        for t in range(self.S_max):
            inp1 = self.fuse_fc1(torch.cat((project_vec, h2.squeeze(0)), 1))
            _, (h1, c1) = self.sent_LSTM_1(inp1.unsqueeze(1), (h1, c1))

            topic_i = self.w_h(h1.squeeze(0))
            att = self.w_a(F.relu(v + topic_i.unsqueeze(1))).squeeze(2)

            att = F.softmax(att, dim=1)
            att_total.append(att)

            topic_feat = torch.sum(project_vec_all*att.unsqueeze(2), dim=1)
            gate = F.sigmoid(self.f_beta(h1.squeeze(0)))
            topic_feat = gate * topic_feat
            inp2 = self.fuse_fc2(torch.cat((topic_feat, h1.squeeze(0)), 1))
            sent_hid_t, (h2, c2) = self.sent_LSTM(inp2.unsqueeze(1), (h2, c2)) #b x S x 512
            sent_hid.append(sent_hid_t)

        att_total = torch.stack(att_total, 1)
        att_total = ((1 - att_total.sum(1))**2).sum(1)
        sent_hid = torch.cat(sent_hid, dim=1)
        hidden1 = self.fc1(sent_hid)
        sent_topic_vec = self.fc2(hidden1) # batch_size x S x 1024
        state = self.logistic(sent_hid) #batch_size x S x 2
        return state, sent_topic_vec, att_total.mean()

    def forward(self, captions, feats, adj, visrep, rel_mat):
        captions = captions[:, :, :-1]
        outputs = []
        batch_size = captions.size(0)
        sent_num = captions.size(1)
        
        state, sent_topic_vec, tr = self.topic_net(feats, adj, rel_mat, visrep)

        for i in range(sent_num):          
            embeddings = self.embed(captions[:, i, :])
            top_value = torch.stack((sent_topic_vec[:, i, 0:512], sent_topic_vec[:, i, 512:]), 0)
            word_hid, _ = self.word_LSTM(embeddings, (top_value, top_value))
            outputs.append(F.linear(word_hid, self.emb_w, self.emb_b))

        outputs = torch.stack(outputs, 1)
        return state, outputs, tr# b x S x 2, b x S x (N-1) x vocab_len(remove <start>)


    def sample_beam(self, feats, beam_size, adj, visrep, rel_mat):
        sent_state, sent_topic_vec, tr = self.topic_net(feats, adj, rel_mat, visrep)

        batch_size = feats.size(0)
        sent_size = self.S_max # here original batch_size is 1
        seq = torch.LongTensor(self.seq_length, sent_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, sent_size)
        done_beams = [[] for _ in range(sent_size)]
        total_seq = []
        for k in range(sent_size):
            vec_1 = sent_topic_vec[0, k, 0:512].expand(beam_size, 512)
            vec_2 = sent_topic_vec[0, k, 512:].expand(beam_size, 512)
            top_value = torch.stack((vec_1, vec_2), 0)
            initial_state = (top_value, top_value)
            # it = torch.ones([beam_size], dtype=torch.long).cuda()
            it = Variable(torch.LongTensor([1]*beam_size)).cuda()
            logprobs, state = self.get_logprobs_state(it, initial_state)
            done_beams[k] = self.beam_search(beam_size, state, logprobs)
            seq[:, k] = done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = done_beams[k][0]['logps']

        return sent_state,  seq.transpose(0, 1)# b x S x 2, b x S x N x vocab_len
        
    def get_logprobs_state(self, it, state):
        xt = self.embed(it)
        xt = xt.unsqueeze(1)
        output, new_state = self.word_LSTM(xt, state)
        probs = F.linear(output.squeeze(1), self.emb_w, self.emb_b)
        logprobs = F.log_softmax(probs, dim=1)
        return logprobs, new_state

    def beam_search(self, beam_size, init_state, init_logprobs, *args, **kwargs):

        # function computes the similarity score to be augmented
        def add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobsf = logprobsf.clone()
            for prev_choice in range(divm):
                prev_decisions = beam_seq_table[prev_choice][local_time]
                for sub_beam in range(bdash):
                    for prev_labels in range(bdash):
                        logprobsf[sub_beam][prev_decisions[prev_labels]] = logprobsf[sub_beam][prev_decisions[prev_labels]] - diversity_lambda
            return unaug_logprobsf

        # does one step of classical beam search

        def beam_step(logprobsf, unaug_logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            #INPUTS:
            #logprobsf: probabilities augmented after diversity
            #beam_size: obvious
            #t        : time instant
            #beam_seq : tensor contanining the beams
            #beam_seq_logprobs: tensor contanining the beam logprobs
            #beam_logprobs_sum: tensor contanining joint logprobs
            #OUPUTS:
            #beam_seq : tensor containing the word indices of the decoded captions
            #beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            #beam_logprobs_sum : joint log-probability of each beam

            ys,ix = torch.sort(logprobsf,1,True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols): # for each column (word, essentially)
                for q in range(rows): # for each beam expansion
                    #compute logprob of expanding beam q with word in (sorted) position c

                    local_logprob = ys[q,c].item()
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    local_unaug_logprob = unaug_logprobsf[q,ix[q,c]]
                    candidates.append({'c':ix[q,c], 'q':q, 'p':candidate_logprob, 'r':local_unaug_logprob})
            candidates = sorted(candidates,  key=lambda x: -x['p'])
            
            new_state = [_.clone() for _ in state]
            #beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1:
            #we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                #fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                #rearrange recurrent states
                for state_ix in range(len(new_state)):
                #  copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']] # dimension one is time step
                #append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c'] # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r'] # the raw logprob here
                beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam
            state = new_state
            return beam_seq,beam_seq_logprobs,beam_logprobs_sum,state,candidates

        # Start diverse_beam_search
        group_size = 1
        diversity_lambda = 0.5
        decoding_constraint = 0
        max_ppl = 0
        bdash = beam_size // group_size # beam per group

        # INITIALIZATIONS
        beam_seq_table = [torch.LongTensor(self.seq_length, bdash).zero_() for _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(self.seq_length, bdash).zero_() for _ in range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(bdash) for _ in range(group_size)]

        # logprobs # logprobs predicted in last time step, shape (beam_size, vocab_size+1)
        done_beams_table = [[] for _ in range(group_size)]
        state_table = [list(torch.unbind(_)) for _ in torch.stack(init_state).chunk(group_size, 2)]
        logprobs_table = list(init_logprobs.chunk(group_size, 0))
        # END INIT

        # Chunk elements in the args
        args = list(args)
        args = [_.chunk(group_size) if _ is not None else [None]*group_size for _ in args]
        args = [[args[i][j] for i in range(len(args))] for j in range(group_size)]

        for t in range(self.seq_length + group_size - 1):
            for divm in range(group_size): 
                if t >= divm and t <= self.seq_length + divm - 1:
                    # add diversity
                    logprobsf = logprobs_table[divm].data.float()
                    # suppress previous word
                    if decoding_constraint and t-divm > 0:
                        logprobsf.scatter_(1, beam_seq_table[divm][t-divm-1].unsqueeze(1).cuda(), float('-inf'))
                    # suppress UNK tokens in the decoding
                    logprobsf[:,logprobsf.size(1)-1] = logprobsf[:, logprobsf.size(1)-1] - 1000  
                    # diversity is added here
                    # the function directly modifies the logprobsf values and hence, we need to return
                    # the unaugmented ones for sorting the candidates in the end. # for historical
                    # reasons :-)
                    unaug_logprobsf = add_diversity(beam_seq_table,logprobsf,t,divm,diversity_lambda,bdash)

                    # infer new beams
                    beam_seq_table[divm],\
                    beam_seq_logprobs_table[divm],\
                    beam_logprobs_sum_table[divm],\
                    state_table[divm],\
                    candidates_divm = beam_step(logprobsf,
                                                unaug_logprobsf,
                                                bdash,
                                                t-divm,
                                                beam_seq_table[divm],
                                                beam_seq_logprobs_table[divm],
                                                beam_logprobs_sum_table[divm],
                                                state_table[divm])

                    # if time's up... or if end token is reached then copy beams
                    for vix in range(bdash):
                        if beam_seq_table[divm][t-divm,vix] == 0 or t == self.seq_length + divm - 1:
                            final_beam = {
                                'seq': beam_seq_table[divm][:, vix].clone(), 
                                'logps': beam_seq_logprobs_table[divm][:, vix].clone(),
                                'unaug_p': beam_seq_logprobs_table[divm][:, vix].sum(),#.item(),
                                'p': beam_logprobs_sum_table[divm][vix]#.item()
                            }
                            if max_ppl:
                                final_beam['p'] = final_beam['p'] / (t-divm+1)
                            done_beams_table[divm].append(final_beam)
                            # don't continue beams from finished sequences
                            beam_logprobs_sum_table[divm][vix] = -1000

                    # move the current group one step forward in time
                    
                    it = Variable(beam_seq_table[divm][t-divm])
                    logprobs_table[divm], state_table[divm] = self.get_logprobs_state(it.cuda(), *(args[divm] + [state_table[divm]]))

        # all beams are sorted by their log-probabilities
        done_beams_table = [sorted(done_beams_table[i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]
        done_beams = reduce(lambda a,b:a+b, done_beams_table)
        return done_beams
