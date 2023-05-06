# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
import torch
from torch import nn
from itertools import product
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from ..attribute_head.roi_attribute_feature_extractors import make_roi_attribute_feature_extractor
from ..box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_relation_feature_extractors import make_roi_relation_feature_extractor
from .roi_relation_predictors import make_roi_relation_predictor
from .inference import make_roi_relation_post_processor
from .loss import make_roi_relation_loss_evaluator
from .sampling import make_roi_relation_samp_processor

from collections import Counter
class ROIRelationHead(torch.nn.Module):
    """
    Generic Relation Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIRelationHead, self).__init__()
        self.cfg = cfg.clone()
        # same structure with box head, but different parameters
        # these param will be trained in a slow learning rate, while the parameters of box head will be fixed
        # Note: there is another such extractor in uniton_feature_extractor
        self.union_feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)
        if cfg.MODEL.ATTRIBUTE_ON:
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=True)
            self.att_feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels, half_out=True)
            feat_dim = self.box_feature_extractor.out_channels * 2
        else:
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
            feat_dim = self.box_feature_extractor.out_channels
        self.predictor = make_roi_relation_predictor(cfg, feat_dim)
        self.post_processor = make_roi_relation_post_processor(cfg)
        self.loss_evaluator = make_roi_relation_loss_evaluator(cfg)
        self.samp_processor = make_roi_relation_samp_processor(cfg)

        # parameters
        self.use_union_box = self.cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION

    def forward(self, features, proposals, targets=None, logger=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes. Note: it has been post-processed (regression, nms) in sgdet mode
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        #import pdb; pdb.set_trace()
        if self.training and not self.cfg.MODEL.WEAKLY_ON:
          # relation subsamples and assign ground truth label during training
            with torch.no_grad():
                if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
                    proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.gtbox_relsample(proposals, targets)
                else:
                    proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.detect_relsample(proposals, targets)
                    #import pdb; pdb.set_trace()
        else:
            #import pdb; pdb.set_trace()
            if self.training and self.cfg.MODEL.WEAKLY_ON:
                rel_binarys = None
                rel_labels  = [target.get_field('relation') for target in targets]
                rel_pair_idxs = self.samp_processor.prepare_test_pairs(features[0].device, proposals)
                
                #rel_pair_idxs = []
                #for proposal in proposals:
                #    t = proposal.get_field('pred_scores')
                #    size_t = t.size(dim=0)
                #    if t.size(dim=0) <10:
                #        print("proposal number is less than 10")
                #        _, indices = torch.topk(t,int(size_t),dim=0)
                #    else: 
                #        _, indices = torch.topk(t,10,dim=0)
                #    indices = indices.tolist()
                    #indices.sort()
                #    tmp = list(product(indices,indices))
                #    tgts = [list(k) for k in tmp if not k[0]==k[1]]
                #    rel_pair_idxs.append(torch.tensor(tgts, device=features[0].device))
                #rel_pair_idxs = []                
                

                '''
                for proposal,target in zip(proposals,targets):
                    filtered_labels = []
                    pred_labe   = proposal.get_field('pred_labels').tolist()
                    tgt_label   = target.get_field('labels').tolist()
                    pred_scores = proposal.get_field('pred_scores')
                    

                    for i,label in enumerate(pred_labe):
                        if label in tgt_label:
                            filtered_labels.append(label)
                        else:
                            filtered_labels.append(0)

                    if not filtered_labels:
                        filtered_labels = pred_labe
                    #import pdb;pdb.set_trace()
                    #filtered_labels = torch.tensor(filtered_labels, device=features[0].device)
                    #rel_indexes = filtered_labels.nonzero().squeeze(-1).tolist()
                    #if rel_indexes and not len(rel_indexes)==1:
                    #    tmp = list(product(rel_indexes,rel_indexes))
                    #else:
                    #    tmp = [k for k in range(len(pred_labe))]
                    #    tmp = list(product(tmp,tmp))
                    #tgts = [list(k) for k in tmp if not k[0]==k[1]]
                    #rel_pair_idxs.append(torch.tensor(tgts, device=features[0].device, dtype=torch.int64))

                    #proposal.add_field("filtered_labels",filtered_labels)
                    #print("filtered labels = ",filtered_labels,"pred_labels = ",pred_labe,"target = ", tgt_label,"gt_labels=",proposal.get_field('labels').tolist())
                    count = Counter(tgt_label)
                    keep1 = []
                    for obj in count:
                        indexes = [j for j, x in enumerate(filtered_labels) if x == obj]
                        if not indexes:
                            continue
                        top_scores = pred_scores[indexes]
                        sort_indexes = sorted(range(len(top_scores)),key=top_scores.__getitem__)
                        sort_indexes.reverse()
                        idx = torch.tensor(indexes,device=features[0].device)

                        keep1.append(idx[sort_indexes[:count[obj]]].tolist())

                    keep  = [item for sublist in keep1 for item in sublist]

                    keep = torch.tensor(keep,device=features[0].device).unique()
                    
                    for j,k in enumerate(filtered_labels):
                        if j not in keep:
                            filtered_labels[j] = 0
                    
                    filtered_labels1 = torch.tensor(filtered_labels, device=features[0].device)
                    proposal.add_field("filtered_labels",filtered_labels1)
                    
                    indices = []
                    for j,k in enumerate(filtered_labels):
                        if int(k) != 0:
                            indices.append(j)
                    
                    if indices:
                        if len(indices) == 1:
                            print('len(indices)=1')
                            _, indices = torch.topk(pred_scores,10,dim=0) 
                            indices = indices.tolist()
                            indices.sort()
                            tmp = list(product(indices,indices))
                            tgts = [list(k) for k in tmp if not k[0]==k[1]]
                            rel_pair_idxs.append(torch.tensor(tgts, device=features[0].device, dtype=torch.long))
                        else:    
                            tmp = list(product(indices,indices))
                            tgts = [list(k) for k in tmp if not k[0]==k[1]]
                            rel_pair_idxs.append(torch.tensor(tgts, device=features[0].device, dtype=torch.long))   
                    else:
                        size_t = pred_scores.size(dim=0)
                        if pred_scores.size(dim=0) <10:
                            print("proposal number is less than 10")
                            _, indices = torch.topk(pred_scores,int(size_t),dim=0)
                        else:
                            print("proposal number is more than 10")
                            _, indices = torch.topk(pred_scores,10,dim=0)
                        indices = indices.tolist()
                        indices.sort()
                        tmp = list(product(indices,indices))
                        tgts = [list(k) for k in tmp if not k[0]==k[1]]
                        rel_pair_idxs.append(torch.tensor(tgts, device=features[0].device, dtype=torch.long))
                '''
                '''
                    rel = target.get_field('relation')
                    rel_pair_idxs_tgts = torch.nonzero(rel).tolist()
                    gt_labels_tgts = target.get_field('labels').tolist()
                    
                    gt_obj_classes = []
                    for tgt_pairs in rel_pair_idxs_tgts:
                        head_idx = tgt_pairs[0]
                        tail_idx = tgt_pairs[1]
                        new_element = []
                        new_element.append(gt_labels_tgts[head_idx])
                        new_element.append(gt_labels_tgts[tail_idx])
                        gt_obj_classes.append(new_element)

                    gt_labels = proposal.get_field('filtered_labels').tolist()
                    # import pdb; pdb.set_trace()
                    new_rel_pair = []
                    for j,label1 in enumerate(gt_labels):
                        if label1 == 0:
                            continue 
                        for k,label2 in enumerate(gt_labels):
                            if label2 == 0:
                                continue
                            for t, obj_classes in enumerate(gt_obj_classes):
                                if [label1, label2] == obj_classes:
                                    new_rel_pair.append([j,k])
                    
                    res = []
                    [res.append(x) for x in new_rel_pair if x not in res and x[0]!=x[1]]

                    if not res:
                        rel_indexes = gt_labels.nonzero().squeeze(-1).tolist()
                        if rel_indexes and not len(rel_indexes)==1:
                            tmp = list(product(rel_indexes,rel_indexes))
                        else:
                            tmp = [k for k in range(len(gt_labels))]
                            tmp = list(product(tmp,tmp))
                        tgts = [list(k) for k in tmp if not k[0]==k[1]]
                        rel_pair_idxs[i] = torch.tensor(tgts, device=features[0].device, dtype=torch.long)
                    else:
                        rel_pair_idxs[i] = torch.tensor(res,  device=features[0].device, dtype=torch.long)

                #import pdb; pdb.set_trace()
                '''
            else:
                rel_labels, rel_binarys = None, None
                rel_pair_idxs = self.samp_processor.prepare_test_pairs(features[0].device, proposals)
        # use box_head to extract features that will be fed to the later predictor processing
        roi_features = self.box_feature_extractor(features, proposals)
        if self.cfg.MODEL.ATTRIBUTE_ON:
            att_features = self.att_feature_extractor(features, proposals)
            roi_features = torch.cat((roi_features, att_features), dim=-1)

        if self.use_union_box:
            union_features = self.union_feature_extractor(features, proposals, rel_pair_idxs)
        else:
            union_features = None
        
        # final classifier that converts the features into predictions
        # should corresponding to all the functions and layers after the self.context class
        refine_logits, relation_logits, add_losses = self.predictor(proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger)
        #import pdb; pdb.set_trace()
        # for test
        if not self.training:
            if self.cfg.MODEL.BASE_ONLY:
                result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, proposals)
            else:
                result = (relation_logits, refine_logits, rel_pair_idxs, proposals)
            return roi_features, result, {}

        loss_relation, loss_refine = self.loss_evaluator(proposals, rel_labels, relation_logits, refine_logits)
        #import pdb; pdb.set_trace()

        if self.cfg.MODEL.ATTRIBUTE_ON and isinstance(loss_refine, (list, tuple)):
            output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine[0], loss_refine_att=loss_refine[1])
        else:
            if self.cfg.MODEL.WEAKLY_ON:
                output_losses = dict(loss_rel=loss_relation)
            else:
                output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine)

        output_losses.update(add_losses)

        return roi_features, (relation_logits, refine_logits, rel_pair_idxs, proposals), output_losses


def build_roi_relation_head(cfg, in_channels):
    """
    Constructs a new relation head.
    By default, uses ROIRelationHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIRelationHead(cfg, in_channels)
