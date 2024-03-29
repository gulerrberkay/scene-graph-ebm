# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT
import time
from collections import Counter
from random import shuffle
import torch
from torch_scatter import scatter
from itertools import product
from maskrcnn_benchmark.modeling.roi_heads.relation_head.sampling import RelationSampling
from maskrcnn_benchmark.modeling.energy_head.graph import Graph
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_motifs import (
    encode_box_info, to_onehot)
import torch.nn.functional as F
import math
import numpy as np
import logging
logger = logging.getLogger(__name__)

def normalize_states(states):
    states = states - torch.min(states, dim=-1, keepdim=True)[0]
    states = states/torch.max(states, dim=1, keepdim=True)[0]

    return states

def prepare_test_pairs( size_len ,dev):
    # prepare object pairs for relation prediction
    rel_pair_idxs = []
    indices = [k for k in range(0, size_len, 1)]
    tmp = list(product(indices,indices))
    tgts = [list(k) for k in tmp if not k[0]==k[1]]
    
    return torch.tensor(tgts, device=dev, dtype=torch.long)

def find_bg_fg_pairs(filtered_labels, rel_matrix, gt_labels_tgts):

    rel_pair_idxs_tgts = torch.nonzero(rel_matrix).tolist()
    # Find foreground o1-o2 pairs in GT rels.
    gt_obj_classes = []
    for tgt_pairs in rel_pair_idxs_tgts:
        head_idx = tgt_pairs[0]
        tail_idx = tgt_pairs[1]
        new_element = []
        new_element.append(gt_labels_tgts[head_idx])
        new_element.append(gt_labels_tgts[tail_idx])
        gt_obj_classes.append(new_element)
    
    # Find foreground/background relations.
    new_rel_pair = []
    new_rel_not_pair = []
    for j,label1 in enumerate(filtered_labels):
        if label1 == 0:
            continue 
        for k,label2 in enumerate(filtered_labels):
            if label2 == 0:
                continue
            if [label1, label2] in gt_obj_classes:
                new_rel_pair.append([j,k])
            else:
                new_rel_not_pair.append([j,k])
    res = []
    [res.append(x) for x in new_rel_pair if x not in res and x[0]!=x[1]] 
    new_rel_pair = res 
    res = []
    [res.append(x) for x in new_rel_not_pair if x not in res and x[0]!=x[1]]
    new_rel_not_pair = res
    shuffle(new_rel_not_pair)

    return new_rel_pair, new_rel_not_pair

def get_predicted_sg(targets,cfg, detections, num_obj_classes, mode, noise_var):
    '''
    This function converts the detction in scene grpah strucuter 
    Parameters:
    -----------
        detection: A tuple of (relation_logits, object_logits, rel_pair_idxs, proposals)
    '''
    if cfg.MODEL.WEAKLY_ON:
        # samp_processor = RelationSampling(
        #     cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        #     cfg.MODEL.ROI_RELATION_HEAD.REQUIRE_BOX_OVERLAP,
        #     cfg.MODEL.ROI_RELATION_HEAD.NUM_SAMPLE_PER_GT_REL,
        #     cfg.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE, 
        #     cfg.MODEL.ROI_RELATION_HEAD.POSITIVE_FRACTION,
        #     cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX,
        #     cfg.TEST.RELATION.REQUIRE_OVERLAP,
        # )
        # _, _, rel_pair_idxs_GT, _ = samp_processor.detect_relsample(detections[3], targets)
        relation_logits = list(detections[0])
        object_logits = list(detections[1])
        rel_pair_idxs = detections[2]
        rel_pair_all = rel_pair_idxs
        
        for i, (proposal,target) in enumerate(zip(detections[3],targets)):
            pred_labe = proposal.get_field('pred_labels').tolist()
            #pred_labe = object_logits[i].max(1)[1] # Find BG objects too.
            #pred_labe = pred_labe.tolist()
            tgt_label = target.get_field('labels').tolist()

            # Find filtered labels.
            filtered_labels = []
            for p,label in enumerate(pred_labe):
                if label in tgt_label:
                    filtered_labels.append(label)
                else:
                    filtered_labels.append(0)
            
            # Sample fg/bg relations.
            rel_matrix     = target.get_field('relation')
            gt_labels_tgts = target.get_field('labels').tolist()
            new_rel_pair, new_rel_not_pair = find_bg_fg_pairs(filtered_labels,rel_matrix, gt_labels_tgts)

            # Find indices in filtered labels.
            indices = []
            deleted_idxs = []
            for j,k in enumerate(filtered_labels): 
                if int(k) != 0:
                    indices.append(j)
                else:
                    deleted_idxs.append(j)
                    
            check = torch.tensor(new_rel_pair, device=detections[0][0].device).unique().tolist() # proposals that has fg rels.
            n_check    = len(check)
            n_filtered = int(np.count_nonzero(filtered_labels))

            new_rel_not_pair2 = []
            for t2_idx, pair in enumerate(new_rel_not_pair):
                if not ((pair[0] not in check) and (pair[1] not in check)):
                    new_rel_not_pair2.append(pair)

            new_rel_not_pair = new_rel_not_pair2[0:math.ceil((n_check*(n_filtered-1))/2)] 

            new_new_rel_pair_idxs = new_rel_pair + new_rel_not_pair
            #import pdb; pdb.set_trace()
            new_idxs2=[]
            for j, pair in enumerate(rel_pair_all[i].tolist()):
                if pair in new_new_rel_pair_idxs:
                    new_idxs2.append(j)
            
            relation_logits[i] = relation_logits[i][new_idxs2,:]
            rel_pair_idxs[i]   = torch.tensor(new_new_rel_pair_idxs, device=detections[0][0].device, dtype=torch.long)

        offset = 0
        pair_list = []
        batch_list = []
        edge_batch_list = []

        ################################################################################################
        rel_list = torch.cat(relation_logits, dim= 0)
        #rel_list = normalize_states(rel_list)       
        #rel_list = torch.mul(mask,rel_list)

        node_list = torch.cat(object_logits, dim= 0)
        #node_list = normalize_states(node_list)
        #node_list = torch.mul(mask,node_list)

        #node_list = torch.mul(node_list, confident.reshape(-1,1))
        ################################################################################################

        for i in range(len(rel_pair_idxs)):
            pair_list.append(rel_pair_idxs[i] + offset)
            batch_list.append(torch.full((object_logits[i].shape[0], ) , i, dtype=torch.long))
            edge_batch_list.append(torch.full( (relation_logits[i].shape[0], ), i, dtype=torch.long))
            offset += object_logits[i].shape[0]
     
        pair_list = torch.cat(pair_list, dim=0)
        batch_list = torch.cat(batch_list, dim=0).to(node_list.device)
        edge_batch_list = torch.cat(edge_batch_list, dim=0).to(node_list.device)

    else: # If weakly mode is OFF.
        offset = 0
        pair_list = []
        batch_list = []
        edge_batch_list = []

    ################################################################################################
        rel_list = torch.cat(detections[0], dim= 0)
        rel_list = normalize_states(rel_list)
    # rel_list = (rel_list - torch.min(rel_list, dim=-1, keepdim=True)[0])
    # rel_list = rel_list/torch.max(rel_list, dim=1, keepdim=True)[0]
    # if detach:
    #     rel_list.detach()
        #import pdb; pdb.set_trace()
    
    #if 0:
    #    node_list=[]
    #    for i in range(len(detections[0])):
    #        indices = detections[2][i].unique().tolist()
    #        node_list.append(detections[1][i][indices][:])
    #    node_list = torch.cat(node_list, dim= 0)
    #else:
        node_list = torch.cat(detections[1], dim= 0)

        if mode == 'predcls':
        #Add small noise to the input
            node_noise = torch.rand_like(node_list).normal_(0, noise_var)
            node_list.data.add_(node_noise)
        else:
            node_list = normalize_states(node_list)

    # if detach:
    #     node_list.detach()
    ################################################################################################
    
        for i in range(len(detections[0])):
            pair_list.append(detections[2][i] + offset)
            batch_list.append(torch.full((detections[1][i].shape[0], ) , i, dtype=torch.long))
            edge_batch_list.append(torch.full( (detections[0][i].shape[0], ), i, dtype=torch.long))
            offset += detections[1][i].shape[0]
    
    
        pair_list = torch.cat(pair_list, dim=0)
        batch_list = torch.cat(batch_list, dim=0).to(node_list.device)
        edge_batch_list = torch.cat(edge_batch_list, dim=0).to(node_list.device)
        
        #print(node_list)
        #print(torch.max(node_list,dim=1))
    
    return node_list, rel_list, pair_list, batch_list, edge_batch_list

def get_gt_scene_graph(targets, num_obj_classes, num_rel_classes, noise_var):
    '''
    Converts gorund truth annotations into graph structure
    '''
    offset = 0
    pair_list = []
    node_list = []
    rel_list = []
    batch_list = []
    edge_batch_list = []

    for i, target in enumerate(targets):
        rel = target.get_field('relation')
        rel_pair_idxs = torch.nonzero(rel)
        pair_list.append(rel_pair_idxs + offset)
        
        node_list.append(target.get_field('labels'))
        rel_list.append(rel[rel_pair_idxs[:,0], rel_pair_idxs[:,1]])

        batch_list.extend([i]*len(target))
        # import ipdb; ipdb.set_trace()
        edge_batch_list.extend([i]*len(pair_list[-1]))
        offset += len(target)

    node_list = to_onehot(torch.cat(node_list, dim=0), num_obj_classes)
    node_noise = torch.rand_like(node_list).normal_(0, noise_var)
    node_list.data.add_(node_noise)

    rel_list = to_onehot(torch.cat(rel_list, dim=0), num_rel_classes)
    rel_noise = torch.rand_like(rel_list).normal_(0, noise_var)
    rel_list.data.add_(rel_noise)
    batch_list = torch.tensor(batch_list).to(node_list.device)
    pair_list = torch.cat(pair_list, dim=0)
    edge_batch_list = torch.tensor(edge_batch_list).to(node_list.device)
    
    # adj_matrix = torch.zeros(size=(node_list.shape[0], node_list.shape[0])).to(node_list.device)
    # adj_matrix[pair_list[:,0], pair_list[:,1]] = 1

    # rel_list = torch.sparse.FloatTensor(pair_list.t(), rel_list, torch.Size([adj_matrix.shape[0], adj_matrix.shape[0], rel_list.shape[-1]])).to_dense()
    
    return node_list, rel_list,  pair_list, batch_list, edge_batch_list

def get_gt_im_graph(node_states, images, detections, base_model, noise_var):
    #Extract region feature from the target bbox
    # import ipdb; ipdb.set_trace()
    if node_states is None:
        features = base_model.backbone(images.tensors)
        node_states = base_model.roi_heads.relation.box_feature_extractor(features, detections)
    #import pdb; pdb.set_trace()
    node_noise = torch.rand_like(node_states).normal_(0, noise_var)
    node_states.data.add_(node_noise)

    return node_states

def get_pred_im_graph(cfg, node_states, images, detections, base_model, noise_var, detach=True):
    #Extract region feature from the predictions
    if node_states is None:
        features = base_model.backbone(images.tensors)
        node_states = base_model.roi_heads.relation.box_feature_extractor(features, detections[-1])

    #import pdb; pdb.set_trace()
    node_noise = torch.rand_like(node_states).normal_(0, noise_var)
    node_states.data.add_(node_noise)
    if detach:
        node_states.detach()
    
    # If weakly on only use objects with high scores. Just as relation_head part topK selections K = 30.
    #if 1:
    #    offset = 0
    #    out_indices = []
    #    for i in range(len(detections[2])):
    #        indices = detections[2][i].unique().tolist()
    #        out_indices = out_indices + [k+offset for k in indices]
    #        offset = offset + len(indices)
    #    node_states = node_states[out_indices,:]
    #    return node_states
    #else:
    return node_states

def detection2graph(targets, cfg, node_states, images, detections, base_model, num_obj_classes, mode, noise_var):

    '''
    Create image graph and scene graph given the detections
    Parameters:
    ----------
        images: Batch of input images
        detection: A tuple of (relation_logits, object_logits, rel_pair_idxs, proposals)
        base_model: realtion predcition model (Used of extracting features)
        num_obj_classes: Number of object classes in the dataset(Used for ocnvertin to one hot encoding)
    Return:
    ----------
        im_graph: A graph corresponding to the image
        scene_graph: A graph corresponding to the scene graph
    '''
    #Scene graph Creation
    
    sg_node_states, sg_rel_states, adj_matrix, batch_list, edge_batch_list  = get_predicted_sg(targets, cfg, detections, num_obj_classes, mode, noise_var)
    
    if 0:
        bboxes = encode_box_info(detections[-1])
        #bboxes = bboxes.split(30, dim=0)
        #tmp_boxes = []
        #for k in range(len(bboxes)):
        #    tmp = bboxes[k]
        #    tmp_boxes.append(tmp[indices_list[k],:])
        #bboxes = torch.cat(tmp_boxes,dim=0)
    else:
        bboxes = encode_box_info(detections[-1])
    #Iage graph generation
    if cfg.MODEL.IMAGE_GRAPH_ON:
        im_node_states = get_pred_im_graph(cfg, node_states, images, detections, base_model, noise_var)
        im_graph = Graph(im_node_states, adj_matrix, batch_list)
    else:
        im_graph = None
    
    scene_graph = Graph(sg_node_states, adj_matrix, batch_list, sg_rel_states, edge_batch_list)
    #im_graph = Graph(im_node_states, adj_matrix, batch_list)

    return im_graph, scene_graph, bboxes

def gt2graph(cfg,node_states, images, targets, base_model, num_obj_classes, num_rel_classes, noise_var):

    '''
    Create image graph and scene graph given the detections
    Parameters:
    ----------
        images: Batch of input images
        target: Gt Target
        base_model: realtion predcition model (Used of extracting features)
        num_obj_classes: Number of object classes in the dataset(Used for ocnvertin to one hot encoding)
    Return:
    ----------
        im_graph: A graph corresponding to the image
        scene_graph: A graph corresponding to the scene graph
    '''

    sg_node_states, sg_edge_states, adj_matrix, batch_list, edge_batch_list = get_gt_scene_graph(targets, num_obj_classes, num_rel_classes, noise_var)
    
    if cfg.MODEL.IMAGE_GRAPH_ON:
        im_node_states = get_gt_im_graph(node_states, images, targets, base_model, noise_var)
        im_graph = Graph(im_node_states, adj_matrix, batch_list)
    else:
        im_graph = None

    sg_graph = Graph(sg_node_states, adj_matrix, batch_list, sg_edge_states, edge_batch_list)
    #im_graph = Graph(im_node_states, adj_matrix, batch_list)
    
    return im_graph, sg_graph, encode_box_info(targets),
