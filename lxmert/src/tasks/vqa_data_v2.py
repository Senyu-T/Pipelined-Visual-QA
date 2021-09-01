# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv, load_obj_pkl, prepare_questions, prepare_answers, encode_question, encode_answers

import os.path

import h5py
import torch
import torch.utils.data as data

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

# The path to data and image features.
VQA_DATA_ROOT = 'data/vqa/'
MSCOCO_IMGFEAT_ROOT = 'data/mscoco_imgfeat/'
VIZWIZ_IMGFEAT_ROOT = 'data/vizwiz_imgfeat/'
# SPLIT2NAME = {
#     'train': 'train2014',
#     'valid': 'val2014',
#     'minival': 'val2014',
#     'nominival': 'val2014',
#     'test': 'test2015',
# }

SPLIT2NAME = {
    'train': '36_vizwiz_train',
    'valid': '36_vizwiz_val'
    # 'train': '36_vizwiz_train_tiny',
    # 'valid': '36_vizwiz_val_tiny'
}

# PREDICTED_QUUES_TYPE_PATH = "snap/vqa/vizwiz_classify_result/valid_predict.json"

class VQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """
    def __init__(self, splits: str, config=None):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open("data/vqa/%s.json" % split)))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open("data/vqa/trainval_ans2label.json"))
        self.label2ans = json.load(open("data/vqa/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)

        # process data for lstm+cnn
        if config is not None:

            with open(config['annotations']['path_vocabs'], 'r') as fd:
                vocabs = json.load(fd)

            annotations_dir = config['annotations']['dir']

            path_ann = os.path.join(annotations_dir, split + ".json")
            with open(path_ann, 'r') as fd:
                self.annotations = json.load(fd)

            self.max_question_length = config['annotations']['max_length']
            self.split = split

            # vocab
            self.vocabs = vocabs
            self.token_to_index = self.vocabs['question']
            self.answer_to_index = self.vocabs['answer']

            # pre-process questions and answers
            self.questions = prepare_questions(self.annotations)
            self.questions = [encode_question(q, self.token_to_index, self.max_question_length) for q in
                              self.questions]  # encode questions and return question and question lenght

            if self.split != 'test':
                self.answers = prepare_answers(self.annotations)
                self.answers = [encode_answers(a, self.answer_to_index) for a in
                                self.answers]  # create a sparse vector of len(self.answer_to_index) for each question containing the occurances of each answer

            if self.split == "train" or self.split == "trainval":
                self._filter_unanswerable_samples()

            # load image names in feature extraction order
            with h5py.File(config['images']['path_features'], 'r') as f:
                img_names = f['img_name'][()]
            self.name_to_id = {name: i for i, name in enumerate(img_names)}

            # names in the annotations, will be used to get items from the dataset
            self.img_names = [s['image'] for s in self.annotations]
            # load features
            self.features = FeaturesDataset(config['images']['path_features'], config['images']['mode'])

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)

    @property
    def num_tokens(self):
        return len(self.token_to_index) + 1  # add 1 for <unknown> token at index 0

    def _filter_unanswerable_samples(self):
        """
        Filter during training the samples that do not have at least one answer
        """
        a = []
        q = []
        annotations = []
        for i in range(len(self.answers)):
            if len(self.answers[i].nonzero()) > 0:
                a.append(self.answers[i])
                q.append(self.questions[i])

                annotations.append(self.annotations[i])
        self.answers = a
        self.questions = q
        self.annotations = annotations


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class VQATorchDataset(Dataset):
    def __init__(self, dataset: VQADataset):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        # Loading detection features to img_data
        img_data = []
        for split in dataset.splits:
            # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
            # It is saved as the top 5K features in val2014_***.tsv
            load_topk = 5000 if (split == 'minival' and topk is None) else topk
            # img_data.extend(load_obj_tsv(
            #     os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[split])),
            #     topk=load_topk))
            img_data.extend(load_obj_pkl(
                os.path.join(VIZWIZ_IMGFEAT_ROOT, '%s.pkl' % (SPLIT2NAME[split])),
                topk=load_topk))

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # item for lstm+cnn model
        out_item = {}
        out_item['question'], out_item['q_length'] = self.raw_dataset.questions[item]
        if self.raw_dataset.split != 'test':
            out_item['answer'] = self.raw_dataset.answers[item]
        img_name = self.raw_dataset.img_names[item]
        feature_id = self.raw_dataset.name_to_id[img_name]
        out_item['img_name'] = self.raw_dataset.img_names[item]
        out_item['visual'] = self.raw_dataset.features[feature_id]
        # collate_fn sorts the samples in order to be possible to pack them later in the model.
        # the sample_id is returned so that the original order can be restored during when evaluating the predictions
        out_item['sample_id'] = item

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                if ans in self.raw_dataset.ans2label:
                    target[self.raw_dataset.ans2label[ans]] = score
            # questionType
            quesType = datum["question_type"]
            return out_item, ques_id, feats, boxes, ques, target, quesType
        else:
            return out_item, ques_id, feats, boxes, ques


class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


class FeaturesDataset(data.Dataset):

    def __init__(self, features_path, mode):
        self.path_hdf5 = features_path

        assert os.path.isfile(self.path_hdf5), \
            'File not found in {}, you must extract the features first with images_preprocessing.py'.format(
                self.path_hdf5)

        self.hdf5_file = h5py.File(self.path_hdf5, 'r')
        self.dataset_features = self.hdf5_file[mode]  # noatt or att (attention)

    def __getitem__(self, index):
        return torch.from_numpy(self.dataset_features[index].astype('float32'))

    def __len__(self):
        return self.dataset_features.shape[0]

