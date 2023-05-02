# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT
import time

import torch
from torch_scatter import scatter
from itertools import product
from maskrcnn_benchmark.modeling.energy_head.graph import Graph
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_motifs import (
    encode_box_info, to_onehot)

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

def get_predicted_sg(targets,cfg, detections, num_obj_classes, mode, noise_var):
    '''
    This function converts the detction in scene grpah strucuter 
    Parameters:
    -----------
        detection: A tuple of (relation_logits, object_logits, rel_pair_idxs, proposals)
    '''

    if cfg.MODEL.WEAKLY_ON and 0:
        relation_logits = list(detections[0])
        object_logits = list(detections[1])
        rel_pair_idxs = detections[2]
        confident = []
        for i, (proposal,target) in enumerate(zip(detections[3],targets)):
            keep_idxs = []
            delete_idxs = []
            
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
                rel_pair_idxs[i] = torch.tensor(tgts, device=detections[0][0].device, dtype=torch.long)
            else:
                rel_pair_idxs[i] = torch.tensor(res, device=detections[0][0].device, dtype=torch.long)
            '''
            # import pdb; pdb.set_trace()

            #keep_idxs = torch.tensor(res, device=detections[0][0].device, dtype=torch.long).unique().tolist()
            #object_logits[i] = object_logits[i][keep_idxs,:]
            
            #new_idxs2=[]
            #pair_list_loop = rel_pair_idxs[i].tolist()
            
            #for j, pair in enumerate(pair_list_loop):
            #    if pair in res:
            #        new_idxs2.append(j)

            #new_idxs2.sort()
       
            #relation_logits[i] = relation_logits[i][new_idxs2,:]
            

            #import pdb; pdb.set_trace()
            # print('sa xd')
            gt_labels   = proposal.get_field('filtered_labels').tolist()
            pred_labels = proposal.get_field('pred_labels').tolist()
            for j, k in enumerate(gt_labels):
                if int(k) != 0:
                    keep_idxs.append(j)
                else:
                    delete_idxs.append(j)  # [0,2,3]
            
            #import pdb; pdb.set_trace()
            if not keep_idxs:
                keep_idxs = [k for k in range(len(pred_labels))]
            
            object_logits[i] = object_logits[i][keep_idxs,:] 


            new_idxs2=[]
            pair_list_loop = rel_pair_idxs[i].tolist()
            
            for j, pair in enumerate(pair_list_loop):
                if (pair[0] in keep_idxs) and (pair[1] in keep_idxs):
                    new_idxs2.append(j)

            new_idxs2.sort()
       
            relation_logits[i] = relation_logits[i][new_idxs2,:]
            rel_pair_idxs[i]   = prepare_test_pairs(object_logits[i].shape[0], detections[0][0].device)

            confident_score = proposal.get_field("pred_scores")
            confident.append(confident_score[keep_idxs])
            
            #print("rel_list",relation_logits[i].shape,"object_list",object_logits[i].shape,"rel_pair_idxs",rel_pair_idxs[i].shape)
            #import pdb; pdb.set_trace()
        
        confident = torch.cat(confident,0)
        
        offset = 0
        pair_list = []
        batch_list = []
        edge_batch_list = []

        ################################################################################################
        rel_list = torch.cat(relation_logits, dim= 0)
        rel_list = normalize_states(rel_list)       

        node_list = torch.cat(object_logits, dim= 0)
        node_list = normalize_states(node_list)

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


    else:
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
        #import pdb; pdb.set_trace()
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
    
    sg_node_states, sg_rel_states, adj_matrix, batch_list, edge_batch_list = get_predicted_sg(targets, cfg, detections, num_obj_classes, mode, noise_var)
        
    #Iage graph generation
    if cfg.MODEL.IMAGE_GRAPH_ON:
        im_node_states = get_pred_im_graph(cfg, node_states, images, detections, base_model, noise_var)
        im_graph = Graph(im_node_states, adj_matrix, batch_list)
    else:
        im_graph = None
    
    scene_graph = Graph(sg_node_states, adj_matrix, batch_list, sg_rel_states, edge_batch_list)
    #im_graph = Graph(im_node_states, adj_matrix, batch_list)

    return im_graph, scene_graph, encode_box_info(detections[-1])

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
