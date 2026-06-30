import torch
import time
import os
import argparse
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
from pycocoevalcap.eval2 import COCOEvalCap

# 命令行参数
parser = argparse.ArgumentParser(description="Image Caption Test Script")
parser.add_argument("--checkpoint", type=str, required=True, help="path to trained model checkpoint (.pth)")
parser.add_argument("--beam_size", type=int, default=7, help="beam search size")
parser.add_argument("--sample_num", type=int, default=-1, help="sample test images, -1 for all")
args = parser.parse_args()

# 全局工具函数
def f(x):
    return Variable(x).cuda()

# 加载词表
with open(opt.vocab_path, 'rb') as f:
    vocab = pickle.load(f)
vocab_size = len(vocab)

# 初始化模型并加载权重
bias_init = np.loadtxt(opt.bias_init_path)
model = DecoderRNN(vocab_size, bias_init)
model = model.cuda()

# 加载训练好的checkpoint
print(f"Loading checkpoint from: {args.checkpoint}")
ckpt = torch.load(args.checkpoint)
model.load_state_dict(ckpt)
model.eval()

# 加载测试数据集
test_data = get_loader(
    opt.test_feats_dir, opt.test_id_path, opt.test_dense_caption,
    opt.caption_path, vocab, batch_size=1, shuffle=False, num_workers=8
)
img_ids = json.load(open(opt.test_id_path, 'r'))
total_img = len(img_ids)
print(f"Total test images: {total_img}")

def test_evaluate(beam_size, sample_num=-1):
    data = {}
    sample_idx_list = list(range(total_img))
    if sample_num > 0 and sample_num < total_img:
        sample_idx_list = random.sample(sample_idx_list, sample_num)
        print(f"Random sample {sample_num} images for evaluation")
    
    with torch.no_grad():
        for idx, (feats, sent_state, targets, mask, adj_mat, visrep, rel_mat) in enumerate(test_data):
            if idx not in sample_idx_list:
                continue
            img_id = img_ids[idx]
            # 数据cuda
            feats = f(feats)
            adj_mat = f(adj_mat)
            visrep = f(visrep)
            rel_mat = f(rel_mat)
            
            # beam search生成caption
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
                    sampled_caption.append(word)
                if state[j] == 1:
                    break
            sentence = ' '.join(sampled_caption)
            data[img_id] = [{"image_id": img_id, "caption": sentence}]
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx+1}/{len(sample_idx_list)} images")
    
    # 保存生成结果json
    save_dir = "test_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    res_file = os.path.join(save_dir, f"beam_{beam_size}_result.json")
    with open(res_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Generated captions saved to {res_file}")

    # COCO指标评估
    cocoEval = COCOEvalCap(opt.ans_file, res_file)
    metrics = cocoEval.evaluate()
    return metrics


if __name__ == "__main__":
    start_time = time.time()
    metrics_result = test_evaluate(beam_size=args.beam_size, sample_num=args.sample_num)
    print("="*50)
    print("Final Evaluation Metrics:")
    for k, v in metrics_result.items():
        print(f"{k}: {v:.4f}")
    print("="*50)
    print(f"Total test time: {time.time() - start_time:.2f}s")