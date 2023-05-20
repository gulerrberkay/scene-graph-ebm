# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT
import torch
import torch.nn.functional as F
from maskrcnn_benchmark.modeling import registry

@registry.SAMPLER.register("SGLD")
class SGLD(object):
    '''Class for Stochastic Gradient Langevin Dynamics'''

    def __init__(self, cfg):

        self.sgld_lr = float(cfg.SAMPLER.LR)
        self.sgld_var = float(cfg.SAMPLER.VAR)
        self.grad_clip = float(cfg.SAMPLER.GRAD_CLIP)
        self.iters = cfg.SAMPLER.ITERS

    def normalize_states(self,states):

        state_norm = torch.sigmoid(states)
        bg_score,indices = torch.max(state_norm,dim=1)
        bg_score = bg_score.reshape(state_norm.shape[0],-1)
        bg_score = 1-bg_score
        #print(values.shape)
        #print(state_norm[:,1:].shape)

        out = torch.cat((bg_score,state_norm[:,1:] ),dim=1)
        return out
    
    def normalize_nodes(self, states):
        #states = states - torch.min(states, dim=-1, keepdim=True)[0]
        #states = states/torch.max(states, dim=1, keepdim=True)[0]
        state_norm = torch.sigmoid(states)
        
        return state_norm
    

    def sample(self, cfg, model, im_graph, scene_graph, bbox, mode, set_grad=False):

        model.train()
        if set_grad:
            scene_graph.requires_grad(mode)

        if mode == 'predcls':
            noise = torch.rand_like(scene_graph.edge_states)
            
            for _ in range(self.iters):
                 #For autograd
                noise.normal_(0, self.sgld_var)
                scene_graph.edge_states.data.add_(noise.data)

                edge_states_grads = torch.autograd.grad(model(im_graph, scene_graph, bbox).sum(), [scene_graph.edge_states], retain_graph=True)[0]
                edge_states_grads.data.clamp_(-self.grad_clip, self.grad_clip)
                
                scene_graph.edge_states.data.add_(edge_states_grads, alpha=-self.sgld_lr)
                # scene_graph.edge_states = F.softmax(scene_graph.edge_states, dim=1)
                # scene_graph.edge_states = scene_graph.edge_states/torch.sum(scene_graph.edge_states, dim=1,  keepdim=True)
                #Normalize to [0,1]
                scene_graph.edge_states = (scene_graph.edge_states - torch.min(scene_graph.edge_states, dim=-1, keepdim=True)[0])
                scene_graph.edge_states = scene_graph.edge_states/torch.max(scene_graph.edge_states, dim=1, keepdim=True)[0]

            #scene_graph.edge_states.detach_()

        else:
            import pdb; pdb.set_trace()
            if cfg.MODEL.WEAKLY_ON:
                scene_graph.node_states = self.normalize_states(scene_graph.node_states)
                scene_graph.edge_states = self.normalize_states(scene_graph.edge_states)

            noise = torch.rand_like(scene_graph.edge_states)
            noise2 = torch.rand_like(scene_graph.node_states)

            for _ in range(self.iters):
                noise.normal_(0, self.sgld_var)
                noise2.normal_(0, self.sgld_var)

                scene_graph.edge_states.data.add_(noise.data)
                scene_graph.node_states.data.add_(noise2.data)
                
                edge_states_grads, node_states_grads = torch.autograd.grad(model(im_graph, scene_graph, bbox).sum(), [scene_graph.edge_states, scene_graph.node_states], retain_graph=True)
                edge_states_grads.data.clamp_(-self.grad_clip, self.grad_clip)
                node_states_grads.data.clamp_(-self.grad_clip, self.grad_clip)

                scene_graph.edge_states.data.add_(edge_states_grads, alpha=-self.sgld_lr)
                scene_graph.node_states.data.add_(node_states_grads, alpha=-self.sgld_lr)

                scene_graph.edge_states = (scene_graph.edge_states - torch.min(scene_graph.edge_states, dim=-1, keepdim=True)[0])
                scene_graph.edge_states = scene_graph.edge_states/torch.max(scene_graph.edge_states, dim=1, keepdim=True)[0]

                scene_graph.node_states = (scene_graph.node_states - torch.min(scene_graph.node_states, dim=-1, keepdim=True)[0])
                scene_graph.node_states = scene_graph.node_states/torch.max(scene_graph.node_states, dim=1, keepdim=True)[0]
                # scene_graph.node_states = scene_graph.node_states/torch.sum(scene_graph.node_states, dim=1,  keepdim=True)

            #scene_graph.edge_states.detach_()
            #scene_graph.node_states.detach_()


        return scene_graph

def build_sampler(cfg):

    sampler = registry.SAMPLER[cfg.SAMPLER.NAME]
    return sampler(cfg)
