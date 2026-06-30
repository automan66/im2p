__author__ = 'tylin'
import sys
sys.path.append('/mnt/filsystem1/xianglu/para/pycocoevalcap/')
from .tokenizer.ptbtokenizer import PTBTokenizer
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
# from spice.spice import Spice
import json

class COCOEvalCap:
    def __init__(self, Gt_json, Res_json):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        with open(Gt_json, 'r') as f:
            self.gts = json.load(f)
        with open(Res_json, 'r') as g:
            self.res = json.load(g)


    def evaluate(self):
        gts = {}
        res = {}
        for key, value in self.res.items():
            gts[int(key)] = self.gts[key]
            res[int(key)] = self.res[key]

        # =================================================
        # Set up scorers
        # =================================================
        # print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        # print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Meteor(),"METEOR"),
            # (Rouge(), "ROUGE_L"),
            # (Cider(), "CIDEr")
            # (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        alone_score = []
        for scorer, method in scorers:
            # print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.5f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.5f"%(method, score))
                alone_score.append(score)
        self.setEvalImgs()
        return score[0]

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]