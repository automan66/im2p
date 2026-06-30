"""
Configuration file!
"""
from argparse import ArgumentParser

data_path = "im2p-data/"

parser = ArgumentParser(description="Configuration")

parser.add_argument('-train_id_path', default=data_path+'new_train_id.json')
parser.add_argument('-test_id_path', default=data_path+'new_test_id.json')

parser.add_argument('-caption_path', default=data_path+'img2paragraph')
parser.add_argument('-vocab_path', default=data_path+'new_vocab.pkl')
parser.add_argument('-bias_init_path', default=data_path+'new_bias.txt')

parser.add_argument('-batch_size', default=128)
parser.add_argument('-lr_init', default=0.001)
parser.add_argument('-S_max', default=6)
parser.add_argument('-N_max', default=30)
parser.add_argument('-num_boxes', default=50)
parser.add_argument('-beam_size', default=2)
parser.add_argument('-train_feats_dir', default=data_path+'train-rep-feats-20/')
parser.add_argument('-test_feats_dir', default=data_path+'test-rep-feats-20/')
parser.add_argument('-train_dense_caption', default=data_path+"train_dense_caption/")
parser.add_argument('-test_dense_caption', default=data_path+"test_dense_caption/")
parser.add_argument('-embed_init', default=data_path+"embedding_glove.pt")
parser.add_argument('-ans_file', default=data_path+'para_whole.json')
opt = parser.parse_args()
