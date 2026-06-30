# im2p
Code for "Comprehensive Relation Modelling for Image Paragraph Generation"

This project focuses on the **image paragraph generation** task, which aims to generate coherent multi-sentence descriptions for images, making it more challenging than traditional single-sentence image captioning. The proposed approach constructs a **scene graph** to represent objects and their relationships, employs **Graph Neural Networks (GNNs)** to model relational information, and incorporates an **attention mechanism** to generate topic vectors for paragraph generation. Experimental results demonstrate that the method achieves performance comparable to existing state-of-the-art (SOTA) approaches.

- Paper: https://www.mi-research.net/en/article/doi/10.1007/s11633-022-1408-2


## Requirements

- PyTorch >= 1.4
- torchvision >= 0.4
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV

## Dataset

The preprocessed dataset is available from **Baidu Netdisk**:

https://pan.baidu.com/s/1au5kFaPwL1NqRcZtnXJJXA?pwd=httt

After downloading and extracting the archive, place the dataset into the `im2p_data` directory.

If you would like to preprocess the dataset from scratch, download the original **Visual Genome** dataset:

- Part 1: https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
- Part 2: https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

Then preprocess the dataset using the following toolkit:

https://github.com/SHTUPLUS/PySGG

## Evaluation

To reproduce the results reported in the paper, first download the pretrained model:

https://pan.baidu.com/s/1RC8kc08vND1VtI6pfYIvKw?pwd=2iyg

Place the checkpoint file into the `checkpoint` directory and run:

```bash
python test.py --checkpoint model_best.pth --test_feats_dir 'im2p_data/test-rep-feats-20/'
```

## Training

To train the model from scratch, run:

```bash
python train.py --train_feats_dir 'im2p_data/train-rep-feats-20/'
```
