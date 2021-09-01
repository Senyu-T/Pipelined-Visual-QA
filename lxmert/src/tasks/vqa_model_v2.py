# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn

from param import args
from lxrt.fuse_entry import LXRTEncoder
from lxrt.fuse_modeling import BertLayerNorm, GeLU

# hugging face transformers for Classifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoModel, BertTokenizerFast, BertTokenizer, AutoTokenizer, BertModel, \
    BertForSequenceClassification

from tasks.lstmcnn_model import LSTMCNNModel
from torch.autograd import Variable

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 20


class VQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()

        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim

        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.lxrt_encoder(sent, (feat, pos))
        logit = self.logit_fc(x)

        return logit


'''
MultiLoss Module
1. CE loss for classifier
2. BCE loss for main VQA model
L_{cls} + L{VQA}
'''


class MultiLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(MultiLoss, self).__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        self.vqa_loss = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, cls_output, vqa_output, quesType, target):
        L_cls = self.cls_loss(cls_output, quesType) / vqa_output.shape[0]
        L_vqa = self.vqa_loss(vqa_output, target) * vqa_output.size(1)
        return self.alpha * L_cls + self.beta * L_vqa


'''
Classifier Modules
BERT + BiLSTM class
'''


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=False,
                            bidirectional=True)
        self.fc1 = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, embeddings):
        self.lstm.flatten_parameters()
        lstm_output, _ = self.lstm(embeddings)
        output = lstm_output[:, -1, :]
        output = self.fc1(output)
        final_logits = self.relu(output)
        output = self.fc2(final_logits)
        return output, final_logits


class BertBiLSTM(nn.Module):
    def __init__(self, num_layers, num_classes, embedding_dim=768, hidden_dim=128):
        super(BertBiLSTM, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.classifier = BiLSTM(self.embedding_dim, hidden_dim, num_layers, num_classes)

    def forward(self, input_ids, attention_mask):
        text_embeddings = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = text_embeddings[0]

        output, final_logits = self.classifier(text_embeddings)
        return output, final_logits


'''
Integrated Module for VQA + Classifier.

Fusion location:
1. after language encoder, before cross-modality encoder
2. after cross modality encoder, before final logit_fc layer
'''


# Try option 2 first
class FusionAfterCrossVQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()

        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim

        # Classifier model, can be changed to LSTM_CNN
        self.cls_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        # self.classifier = BertBiLSTM(args.cls_numlayer, args.cls_numclasses)
        self.classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                                        num_labels=5,
                                                                        output_attentions=True,
                                                                        output_hidden_states=True)
        # self.classifier = VQAModel()
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size
        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :return: (b, num_answer) The VQA logit of each answers.
             (b, num_quesType) The CLS logit for each type
        """
        x = self.lxrt_encoder(sent, (feat, pos))
        # get BERT classifier work
        encoded_sent = self.cls_tokenizer.batch_encode_plus(sent, add_special_tokens=True,
                                                            max_length=64,
                                                            padding='max_length',
                                                            return_attention_mask=True,
                                                            truncation=True, return_tensors='pt')

        input_id = encoded_sent.get('input_ids').cuda()
        mask = encoded_sent.get('attention_mask').cuda()
        cls_out = self.classifier(input_id, mask)
        pooled_output = cls_out.hidden_states[-1][:, 0, :]
        # @TODO: try other fusion method
        # print(pooled_output.shape)
        z = x * pooled_output  # subject to change like MLP
        # print(z.shape)
        logit = self.logit_fc(z)

        return logit, cls_out.logits


'''
Single LXMERT, both for type-prediction and answer-prediction
use a MLP as cls classifier on top of the cross-encoder output
'''


class JointVQAModel(nn.Module):
    def __init__(self, num_answers, num_answer_types):
        super().__init__()

        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim

        self.cls_forward = nn.Linear(hid_dim, hid_dim)
        self.cls_layer_norm = BertLayerNorm(hid_dim, eps=1e-12)
        self.cls_fc = nn.Linear(hid_dim, num_answer_types)

        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.lxrt_encoder(sent, (feat, pos))

        cls_out = self.cls_forward(x)
        cls_out = F.gelu(cls_out)
        cls_out = self.cls_layer_norm(cls_out)
        cls_logit = self.cls_fc(cls_out)

        vqa_logit = self.logit_fc(x * cls_out)

        return vqa_logit, cls_logit



'''
Fusion before Cross Modality Encoder
'''


class FusionBeforeCrossVQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()

        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim

        # Classifier model, can be changed to LSTM_CNN
        self.cls_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        # self.classifier = BertBiLSTM(args.cls_numlayer, args.cls_numclasses)
        self.classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                                        num_labels=5,
                                                                        output_attentions=True,
                                                                        output_hidden_states=True)
        # self.classifier = VQAModel()
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size
        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :return: (b, num_answer) The VQA logit of each answers.
             (b, num_quesType) The CLS logit for each type
        """

        # get BERT classifier work
        encoded_sent = self.cls_tokenizer.batch_encode_plus(sent, add_special_tokens=True,
                                                            max_length=64,
                                                            padding='max_length',
                                                            return_attention_mask=True,
                                                            truncation=True, return_tensors='pt')

        input_id = encoded_sent.get('input_ids').cuda()
        mask = encoded_sent.get('attention_mask').cuda()
        cls_out = self.classifier(input_id, mask)
        # print(cls_out.hidden_states[-1][:,:20,:].shape)
        cls_lang_rep = cls_out.hidden_states[-1][:, :20, :]
        # pooled_output = cls_out.hidden_states[-1][:,0,:]
        x = self.lxrt_encoder(cls_lang_rep, sent, (feat, pos))
        # @TODO: try other fusion method
        # print(pooled_output.shape)
        # z = x * pooled_output  # subject to change like MLP
        # print(z.shape)
        # logit = self.logit_fc(z)
        logit = self.logit_fc(x)
        torch.cuda.empty_cache()
        return logit, cls_out.logits


class LSTMCNNCrossVQAModel(nn.Module):
    def __init__(self, config, num_answers):
        super().__init__()

        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim

        # 3892 hardcode, train_loader.dataset.num_tokens
        self.classifier = nn.DataParallel(LSTMCNNModel(config, 3892))
        # self.classifier = BertBiLSTM(args.cls_numlayer, args.cls_numclasses)

        # self.classifier = VQAModel()
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

        self.cls_fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(768, 5)
        )


    def forward(self, item, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size
        :param item: for lstmcnn
        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :return: (b, num_answer) The VQA logit of each answers.
             (b, num_quesType) The CLS logit for each type
        """
        x = self.lxrt_encoder(sent, (feat, pos))

        # get LSTMCNN classifier work
        v = item['visual']
        q = item['question']
        a = item['answer']
        q_length = item['q_length']

        v = Variable(v.cuda())
        q = Variable(q.cuda())
        a = Variable(a.cuda())
        q_length = Variable(q_length.cuda())

        cls_out = self.classifier(v, q, q_length)
        # @TODO: try other fusion method

        print(x.shape)
        print(cls_out.shape)
        # z = torch.cat((x, cls_out), 1)  # subject to change like MLP
        z = x * cls_out
        print(z.shape)
        logit = self.logit_fc(z)
        cls_logits = self.cls_fc(cls_out)

        return logit, cls_logits


class LSTMCNNFusionBeforeCrossVQAModel(nn.Module):
    def __init__(self, config, num_answers):
        super().__init__()

        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim

        # 3892 hardcode, train_loader.dataset.num_tokens
        self.classifier = nn.DataParallel(LSTMCNNModel(config, 3892))
        # self.classifier = BertBiLSTM(args.cls_numlayer, args.cls_numclasses)

        # self.classifier = VQAModel()
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

        self.cls_fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(768, 5)
        )
        # self.cls_att = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(1, 20)
        # )

    def forward(self, item, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size
        :param item: for lstmcnn
        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :return: (b, num_answer) The VQA logit of each answers.
             (b, num_quesType) The CLS logit for each type
        """

        # get LSTMCNN classifier work
        v = item['visual']
        q = item['question']
        a = item['answer']
        q_length = item['q_length']

        v = Variable(v.cuda())
        q = Variable(q.cuda())
        a = Variable(a.cuda())
        q_length = Variable(q_length.cuda())

        cls_out, q = self.classifier(v, q, q_length)
        # @TODO: try other fusion method
        q = q.unsqueeze(1).repeat(1, 20, 1)
        x = self.lxrt_encoder(q, sent, (feat, pos))
        # z = torch.cat((x, cls_out), 1)  # subject to change like MLP

        logit = self.logit_fc(x)
        cls_logits = self.cls_fc(cls_out)

        torch.cuda.empty_cache()
        return logit, cls_logits


class LSTMCNNFusionBothCrossVQAModel(nn.Module):
    def __init__(self, config, num_answers):
        super().__init__()

        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim

        # 3892 hardcode, train_loader.dataset.num_tokens
        self.classifier = nn.DataParallel(LSTMCNNModel(config, 3892))
        # self.classifier = BertBiLSTM(args.cls_numlayer, args.cls_numclasses)

        # self.classifier = VQAModel()
        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

        self.cls_fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(768, 5)
        )
        # self.cls_att = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(1, 20)
        # )

    def forward(self, item, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size
        :param item: for lstmcnn
        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :return: (b, num_answer) The VQA logit of each answers.
             (b, num_quesType) The CLS logit for each type
        """

        # get LSTMCNN classifier work
        v = item['visual']
        q = item['question']
        a = item['answer']
        q_length = item['q_length']

        v = Variable(v.cuda())
        q = Variable(q.cuda())
        a = Variable(a.cuda())
        q_length = Variable(q_length.cuda())

        cls_out, q = self.classifier(v, q, q_length)
        # @TODO: try other fusion method
        q = q.unsqueeze(1).repeat(1, 20, 1)
        x = self.lxrt_encoder(q, sent, (feat, pos))
        z = x * cls_out  # subject to change like MLP
        logit = self.logit_fc(z)

        cls_logits = self.cls_fc(cls_out)

        torch.cuda.empty_cache()
        return logit, cls_logits


# '''
# Fusion before Cross Modality Encoder
# '''
#
#
# class FusionBeforeCrossVQAModel(nn.Module):
#     def __init__(self, num_answers):
#         super().__init__()
#
#         # Build LXRT encoder
#         self.lxrt_encoder = LXRTEncoder(
#             args,
#             max_seq_length=MAX_VQA_LENGTH
#         )
#         hid_dim = self.lxrt_encoder.dim
#
#         # Classifier model, can be changed to LSTM_CNN
#         self.cls_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
#         # self.classifier = BertBiLSTM(args.cls_numlayer, args.cls_numclasses)
#         self.classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased',
#                                                                         num_labels=5,
#                                                                         output_attentions=True,
#                                                                         output_hidden_states=True)
#         # self.classifier = VQAModel()
#         # VQA Answer heads
#         self.logit_fc = nn.Sequential(
#             nn.Linear(hid_dim, hid_dim * 2),
#             GeLU(),
#             BertLayerNorm(hid_dim * 2, eps=1e-12),
#             nn.Linear(hid_dim * 2, num_answers)
#         )
#         self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)
#
#     def forward(self, feat, pos, sent):
#         """
#         b -- batch_size, o -- object_number, f -- visual_feature_size
#         :param feat: (b, o, f)
#         :param pos:  (b, o, 4)
#         :param sent: (b,) Type -- list of string
#         :return: (b, num_answer) The VQA logit of each answers.
#              (b, num_quesType) The CLS logit for each type
#         """
#
#         # get BERT classifier work
#         encoded_sent = self.cls_tokenizer.batch_encode_plus(sent, add_special_tokens=True,
#                                                             max_length=64,
#                                                             padding='max_length',
#                                                             return_attention_mask=True,
#                                                             truncation=True, return_tensors='pt')
#
#         input_id = encoded_sent.get('input_ids').cuda()
#         mask = encoded_sent.get('attention_mask').cuda()
#         cls_out = self.classifier(input_id, mask)
#         # print(cls_out.hidden_states[-1][:,:20,:].shape)
#         cls_lang_rep = cls_out.hidden_states[-1][:, :20, :]
#         # pooled_output = cls_out.hidden_states[-1][:,0,:]
#         x = self.lxrt_encoder(cls_lang_rep, sent, (feat, pos))
#         # @TODO: try other fusion method
#         # print(pooled_output.shape)
#         # z = x * pooled_output  # subject to change like MLP
#         # print(z.shape)
#         # logit = self.logit_fc(z)
#         logit = self.logit_fc(x)
#         torch.cuda.empty_cache()
#         return logit, cls_out.logits