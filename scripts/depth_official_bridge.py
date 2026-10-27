"""D/DV matching inside the official ToMe vision blocks; preserve native attention."""
from dataclasses import asdict
import hashlib
import types

import numpy as np
from PIL import Image
import torch

from depth_region_vlm.geometry_v5 import build_geometry_fast
from depth_region_vlm.merge_v3 import feasibility
from depth_region_vlm.merge_v5 import (apply_plan_fast_fp32, apply_plan_stable,
    prepare_d_plans_stable, select_mapping_stable, tensor_geometry_stable, update_geometry_stable)
from depth_region_vlm.relative_depth import normalize_relative_inverse_depth


def build_single_geometry(raw):
    normalized,_=normalize_relative_inverse_depth(raw)
    resized=np.asarray(Image.fromarray(normalized).resize((384,384),Image.Resampling.BILINEAR),dtype=np.float32)
    g=build_geometry_fast(resized,0,14,14,k=4,kmeans_iterations=50,min_valid_ratio=.8,
        boundary_threshold=.15,small_region_max=4)
    digest=hashlib.sha256()
    for a in [g.depth,g.xy,g.region,g.protected]:digest.update(np.ascontiguousarray(a).tobytes())
    return g,dict(geometry_sha256=digest.hexdigest(),protected_tokens=int(g.protected.sum()),
        feasible391=asdict(feasibility(g,391)),feasible79=asdict(feasibility(g,79)))


class DepthBridge:
    def __init__(self,vision,method):
        if method not in ['Depth-only','Depth-RGB']:raise ValueError(method)
        self.vision=vision;self.method=method;self.enabled=False
        self.originals=[];self.layers=vision.vision_model.encoder.layers
        for i,layer in enumerate(self.layers):
            original=layer.forward;self.originals.append(original)
            def forward(module,hidden_states,r,size,attention_mask,causal_attention_mask,
                        output_attentions=False,_index=i,_original=original):
                if not self.enabled or r==0:
                    result=_original(hidden_states,r,size,attention_mask,causal_attention_mask,output_attentions)
                    if self.enabled:self.tokens.append(int(result[0][0].shape[1]))
                    return result
                return self.forward_layer(module,hidden_states,r,size,attention_mask,causal_attention_mask,
                    output_attentions,_index)
            layer.forward=types.MethodType(forward,layer)

    def configure(self,g,r,audit=False):
        self.tokens=[];self.mapping_hash=hashlib.sha256();self.audit=audit;self.plan_stats=[]
        self.enabled=True;self.r=r
        self.schedule={i:729-i*r for i in range(1,27)}
        f=feasibility(g,self.schedule[26])
        if f.optimistic_min_tokens>self.schedule[26]:return f
        self.geometry=tensor_geometry_stable([g],'cuda:0')
        if r and self.method=='Depth-only':
            self.plans,_,_=prepare_d_plans_stable([g],self.schedule,'cuda:0',max_distance=.16,num_layers=26)
        return f

    def forward_layer(self,base,hidden,r,size,mask,causal,output_attentions,index):
        assert r==self.r
        n=hidden.shape[1];residual=hidden
        native_size=None if size is None else size.to(hidden.dtype)
        attention,attn_weights,metric=base.self_attn(hidden_states=base.layer_norm1(hidden),size=native_size,
            attention_mask=mask,causal_attention_mask=causal,output_attentions=output_attentions)
        hidden=residual+attention
        if self.method=='Depth-only':
            plan=self.plans[index]
        else:
            plan=select_mapping_stable(self.geometry,r,metric,.5,max_distance=.16,
                medoid_tolerance=1e-6,score_resolution=1e-6,validate_capacity=True)
        assert plan.input_tokens==n and plan.output_tokens==self.schedule[index+1]
        if self.audit:
            # Independently reconstruct group labels/protection for each plan.
            state=self.geometry
            assert not bool((plan.selected & state.protected).any())
            destination_components=state.component.gather(1,plan.destination)
            assert bool((destination_components[plan.selected]==state.component[plan.selected]).all())
        merger=apply_plan_fast_fp32 if self.method=='Depth-only' else apply_plan_stable
        hidden,size=merger(hidden,size,plan)
        assert bool(torch.isfinite(hidden).all()) and float(size.sum())==729.
        self.mapping_hash.update(plan.mapping.detach().cpu().numpy().tobytes())
        if self.method=='Depth-RGB' or self.audit:self.geometry=update_geometry_stable(self.geometry,plan)
        self.plan_stats.append(dict(layer=index+1,input_tokens=n,output_tokens=plan.output_tokens))
        residual=hidden;hidden=base.mlp(base.layer_norm2(hidden))+residual
        self.tokens.append(int(hidden.shape[1]))
        outputs=(hidden,)
        if output_attentions:outputs+=(attn_weights,)
        return outputs,size

    def close(self):
        for layer,forward in zip(self.layers,self.originals):layer.forward=forward
