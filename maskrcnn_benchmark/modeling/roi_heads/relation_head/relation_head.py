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
from maskrcnn_benchmark.modeling.energy_head.utils import find_bg_fg_pairs
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
                features, proposals, targets = self.postprocess_proposals(features, proposals, targets)
                rel_labels  = [target.get_field('relation') for target in targets]
                rel_pair_idxs = self.samp_processor.prepare_test_pairs(features[0].device, proposals)
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
    
    def postprocess_proposals(self, features, proposals, targets):
        """
        Removes problematic images for weak supervision. Image is problematic if:
        1) Detector cannot find any object in targets.
        2) If detector only finds 1 object from targets. -> 1 obj basically means no relation since there is no other obj.
        3) If the detected object pairs has no fg relation according to GT rels - rel matrix
        """
        # Find filtered labels.
        features_new = []
        proposals_new = []
        targets_new = []
        for i,proposal in enumerate(proposals):
            pred_labels = proposal.get_field("pred_labels")
            tgt_labels  = targets[i].get_field("labels")
            filtered_labels = []
            for p,label in enumerate(pred_labels):
                if label in tgt_labels:
                    filtered_labels.append(label)
                else:
                    filtered_labels.append(0)
            keep_indices = []
            deleted_idxs = []
            for label_idx,label in enumerate(filtered_labels): #  not in GT relations
                if int(label) != 0:
                    keep_indices.append(label_idx)
                else:
                    deleted_idxs.append(label_idx)
            rel_matrix = targets[i].get_field('relation')
            gt_labels_tgts = targets[i].get_field('labels').tolist()
            new_rel_pair, _ = find_bg_fg_pairs(filtered_labels,rel_matrix, gt_labels_tgts)
            if not (len(keep_indices)==0 or len(keep_indices)==1 or  int(len(new_rel_pair))==0):
                features_new.append(features[i])
                proposals_new.append(proposals[i])
                targets_new.append(targets[i])
        
        return features_new, proposals_new, targets_new


def build_roi_relation_head(cfg, in_channels):
    """
    Constructs a new relation head.
    By default, uses ROIRelationHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIRelationHead(cfg, in_channels)
