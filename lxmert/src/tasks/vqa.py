# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections
import json
import numpy as np
import pickle

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.vqa_model import VQAModel
from tasks.vqa_data import VQADataset, VQATorchDataset, VQAEvaluator

from tasks.vqa_classifier_data import VQAClassifierDataset, VQAClassifierTorchDataset, VQAClassifierEvaluator
from tasks.vqa_classify import PREDICT_SOFTMAX_PATH, PREDICT_LOGIT_PATH

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

QUESTION_TYPE_LABELS_PATH = "data/vqa/trainval_questype2labels.json"
CLASSIFIER_LABEL2ANS_PATH = "data/vqa/classifier_label2ans.json"

PRED_QID2QUESTYPE_PATH = 'snap/vqa/vizwiz_classify_result/pred_qid2questype.json'

MAIN_LOGITS_PATH = 'snap/vqa/vizwiz_all_results/logit_result.pt'


FUSION_USE_LOGIT = False
FUSION_USE_SOFTMAX = False
QUESTYPE_USE_PRED = False


def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    print(splits)
    dset = VQADataset(splits)
    tset = VQATorchDataset(dset)
    evaluator = VQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class VQA:
    def __init__(self):
        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, bs=1024,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None
        
        # Model
        self.model = VQAModel(self.train_tuple.dataset.num_answers)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)
        
        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

        #question type labels
        self.classifier_label2ans = json.load(open(CLASSIFIER_LABEL2ANS_PATH))

        self.questype2labels = json.load(open(QUESTION_TYPE_LABELS_PATH))
        self.pred_qid2questype = json.load(open(PRED_QID2QUESTYPE_PATH))

        #classify logits and softmax results
        with open(PREDICT_SOFTMAX_PATH, 'rb') as f:
          self.quesid2softmax = pickle.load(f)
        with open(PREDICT_LOGIT_PATH, 'rb') as f:
          self.quesid2logit = pickle.load(f)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target, quesType) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2
                loss = self.bce_loss(logit, target)
                loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            # ques_id, feats, boxes, sent = datum_tuple[:4]   # Avoid seeing ground truth
            ques_id, feats, boxes, sent, target, quesType = datum_tuple
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                
                qids = []
                if FUSION_USE_LOGIT:
                  for qid, l in zip(ques_id, logit.cpu().numpy()):
                    classify_logit = self.quesid2logit[qid]
                    highest_label = np.where(np.amax(classify_logit) == classify_logit)[0][0]
                    highest_questype = self.classifier_label2ans[highest_label]
                    
                    classify_map_logit = np.zeros(l.size)
                    for label in range(0,len(self.classifier_label2ans)):
                      ques_type = self.classifier_label2ans[label]
                      questype_labels = self.questype2labels[ques_type]
                      classify_map_logit[questype_labels] = classify_logit[label]
                    
                    
                    if highest_questype == "unanswerable":
                      qids.append(qid)

                    #   print(classify_map_logit)
                    fusion_logit = l
                    # classify_map_logit if highest_questype  == "unanswerable" else 
                    score = np.amax(fusion_logit)
                    label = np.where(fusion_logit == score)[0][0]

                    label = 28574 if highest_questype == "unanswerable" else label
                
                    ans = dset.label2ans[label]
                    quesid2ans[qid] = ans    
                
                elif FUSION_USE_SOFTMAX:
                  for qid, l in zip(ques_id, logit):
                    m = nn.Softmax()
                    l_softmax = m(l)

                    l_softmax = l_softmax.cpu().numpy()
                    l = l.cpu().numpy()

                    classify_softmax = self.quesid2softmax[qid]
                    # highest_label = np.where(np.amax(classify_logit) == classify_logit)[0][0]
                    # highest_questype = self.classifier_label2ans[highest_label]
                    
                    classify_map_softmax = np.zeros(l.size)
                    for label in range(0,len(self.classifier_label2ans)):
                      ques_type = self.classifier_label2ans[label]
                      questype_labels = self.questype2labels[ques_type]
                      classify_map_softmax[questype_labels] = classify_softmax[label]
                    
                    fusion_softmax = l_softmax + classify_map_softmax
                    
                    score = np.amax(fusion_softmax)
                    label = np.where(fusion_softmax == score)[0][0]
                
                    ans = dset.label2ans[label]
                    quesid2ans[qid] = ans  

                elif QUESTYPE_USE_PRED:
                  for qid, l in zip(ques_id, logit.cpu().numpy()):
                    pred_questype = self.pred_qid2questype[qid]
                    if pred_questype in ['other','unanswerable']:
                      pred_questype_labels = self.questype2labels[pred_questype]
                      min_logit = np.min(l) - 1
                      mask = np.ones(l.size, dtype=bool)
                      mask[pred_questype_labels] = False
                      l[mask] = min_logit

                    score = np.amax(l)
                    label = np.where(l == score)[0][0]
                
                    ans = dset.label2ans[label]
                    quesid2ans[qid] = ans
                else:
                  qid2logit = {}
                  for qid, l, qType in zip(ques_id, logit.cpu().numpy(), quesType):
                      #correct_type_labels = self.questype2labels[qType]
                      #min_logit = np.min(l) - 1
                      #mask = np.ones(l.size, dtype=bool)
                      #mask[correct_type_labels] = False
                      #l[mask] = min_logit

                      score = np.amax(l)
                      label = np.where(l == score)[0][0]
                  
                      ans = dset.label2ans[label]
                      quesid2ans[qid] = ans

                      #save logits results
                      qid2logit[qid] = l
                  with open(MAIN_LOGITS_PATH, 'wb') as outfile:
                    pickle.dump(qid2logit, outfile)
                  print("logits results saved")

                # score, label = logit.max(1)

                # for qid, l in zip(ques_id, label.cpu().numpy()):
                #     ans = dset.label2ans[l]
                #     quesid2ans[qid] = ans

        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target, quesType) in enumerate(loader):
            print(i)
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    # Build Class
    vqa = VQA()

    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        vqa.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'test' in args.test:
            vqa.predict(
                get_data_tuple(args.test, bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
        elif 'val' in args.test:    
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = vqa.evaluate(
                get_data_tuple('valid', bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'valid_predict.json')
            )
            print(result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', vqa.train_tuple.dataset.splits)
        if vqa.valid_tuple is not None:
            print('Splits in Valid data:', vqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        vqa.train(vqa.train_tuple, vqa.valid_tuple)


