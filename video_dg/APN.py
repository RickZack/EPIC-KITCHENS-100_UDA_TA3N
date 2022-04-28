import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from video_dg.att_modules import *
from opts import parser as args
from video_dg.hyper import Hyperparams as hp





class RelationModuleMultiScale(torch.nn.Module):

    def __init__(self, img_feature_dim, num_frames):
        super(RelationModuleMultiScale, self).__init__()
        self.subsample_num = 3
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)]  

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(
                relations_scale)))  

        self.num_frames = num_frames
        num_bottleneck = 256
        self.fc_fusion_scales = nn.ModuleList()  
        self.transformer = AttModel(hp, 10000, 10000)


    def forward(self, input):
        act_all = input[:, self.relations_scales[0][0], :]
        act_all = self.transformer(act_all, act_all) # R global -> RIII
        act_block = act_all # costant R global
        other = act_all # RII
        adv_feature = act_all


        for scaleID in range(1, len(self.scales)):
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]),
                                                          self.subsample_scales[scaleID], replace=False)
            idx1, _, _ = idx_relations_randomsample
            act_relation1 = input[:, self.relations_scales[scaleID][idx1], :]

            act_relation1 = self.transformer(act_relation1, act_relation1)

            temp_1 = self.transformer(act_relation1, act_block)

            if scaleID == 1:
                other = temp_1
            else:
                other  =  other + temp_1 # Accumulate for RII
            act_all = act_all + temp_1 # Accumulate for RIII

        other = other.view((-1, other.size(1) * other.size(2)))
        # adv_result = self.final(other) # logits of RII

        act_feature = act_all.view((-1, act_all.size(1) * act_all.size(2))) # RIII reshaped


        # act_all_result = self.final(act_feature) # logits for RIII
        return act_feature, other


    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))





class AttModel(nn.Module):
    def __init__(self, hp_, enc_voc, dec_voc):
        super(AttModel, self).__init__()
        self.hp = hp_

        self.enc_voc = enc_voc
        self.dec_voc = dec_voc

        self.enc_dropout = nn.Dropout(self.hp.dropout_rate)
        for i in range(self.hp.num_blocks):
            self.__setattr__('enc_self_attention_%d' % i, multihead_attention(num_units=self.hp.hidden_units,
                                                                              num_heads=self.hp.num_heads,
                                                                              dropout_rate=self.hp.dropout_rate,
                                                                              causality=False))
            self.__setattr__('enc_feed_forward_%d' % i, feedforward(self.hp.hidden_units,
                                                                    [4 * self.hp.hidden_units,
                                                                     self.hp.hidden_units]))

        self.dec_dropout = nn.Dropout(self.hp.dropout_rate)
        for i in range(self.hp.num_blocks):
            self.__setattr__('dec_self_attention_%d' % i,
                             multihead_attention(num_units=self.hp.hidden_units,
                                                 num_heads=self.hp.num_heads,
                                                 dropout_rate=self.hp.dropout_rate,
                                                 causality=True))
            self.__setattr__('dec_vanilla_attention_%d' % i,
                             multihead_attention(num_units=self.hp.hidden_units,
                                                 num_heads=self.hp.num_heads,
                                                 dropout_rate=self.hp.dropout_rate,
                                                 causality=False))
            self.__setattr__('dec_feed_forward_%d' % i, feedforward(self.hp.hidden_units,
                                                                    [4 * self.hp.hidden_units,
                                                                     self.hp.hidden_units]))

        self.logits_layer = nn.Linear(self.hp.hidden_units, self.dec_voc)
        self.label_smoothing = label_smoothing()

    def forward(self, x, y):
        self.enc = x
        for i in range(self.hp.num_blocks):
            self.enc = self.__getattr__('enc_self_attention_%d' % i)(self.enc, self.enc, self.enc)
            self.enc = self.__getattr__('enc_feed_forward_%d' % i)(self.enc)
        self.dec = y
        for i in range(self.hp.num_blocks):
            self.dec = self.__getattr__('dec_self_attention_%d' % i)(self.dec, self.dec, self.dec)
            self.dec = self.__getattr__('dec_vanilla_attention_%d' % i)(self.dec, self.enc, self.enc)
            self.dec = self.__getattr__('dec_feed_forward_%d' % i)(self.dec)

        self.dec = self.dec_dropout(self.dec)

        return self.dec
