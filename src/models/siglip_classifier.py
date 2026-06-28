import torch
from torch import nn
from transformers import AutoModel

class SigLIPClassifier(nn.Module):
    def __init__(self, model_name, num_labels, freeze_encoder=False, use_hierarchy=False, num_organ=1, num_category=1, num_family=4):
        super().__init__(); self.backbone=AutoModel.from_pretrained(model_name); self.use_hierarchy=use_hierarchy
        hidden=getattr(self.backbone.config,'projection_dim',None) or getattr(self.backbone.config,'hidden_size',None) or 768
        if hasattr(self.backbone.config,'vision_config'): hidden=getattr(self.backbone.config.vision_config,'hidden_size',hidden)
        self.head=nn.Linear(hidden,num_labels); self.organ_head=nn.Linear(hidden,num_organ) if use_hierarchy else None; self.category_head=nn.Linear(hidden,num_category) if use_hierarchy else None; self.family_head=nn.Linear(hidden,num_family) if use_hierarchy else None
        if freeze_encoder:
            for p in self.backbone.parameters(): p.requires_grad=False
    def image_features(self, pixel_values):
        # Support different attribute names depending on how the class was initialized.
        if hasattr(self, "vision"):
            vision_model = self.vision
        elif hasattr(self, "vision_model"):
            vision_model = self.vision_model
        elif hasattr(self, "backbone"):
            vision_model = self.backbone
        elif hasattr(self, "model"):
            vision_model = self.model
        else:
            raise AttributeError(
                "No vision backbone found. Expected one of: "
                "self.vision, self.vision_model, self.backbone, self.model"
            )

        # If this is a full SigLIP model, use its vision_model.
        if hasattr(vision_model, "vision_model"):
            vision_model = vision_model.vision_model

        out = vision_model(pixel_values=pixel_values)

        # SigLIP/SigLIP2 vision models may return BaseModelOutputWithPooling.
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            feat = out.last_hidden_state[:, 0]
        elif isinstance(out, (tuple, list)):
            if len(out) > 1 and out[1] is not None:
                feat = out[1]
            else:
                feat = out[0][:, 0]
        else:
            feat = out

        return feat

    def _lazy_resize(self, feat):
        if feat.shape[-1] != self.head.in_features:
            device=feat.device; self.head=nn.Linear(feat.shape[-1], self.head.out_features).to(device)
            if self.use_hierarchy:
                self.organ_head=nn.Linear(feat.shape[-1], self.organ_head.out_features).to(device); self.category_head=nn.Linear(feat.shape[-1], self.category_head.out_features).to(device); self.family_head=nn.Linear(feat.shape[-1], self.family_head.out_features).to(device)
    def forward(self,pixel_values):
        feat=self.image_features(pixel_values); self._lazy_resize(feat); out={'logits':self.head(feat),'features':feat}
        if self.use_hierarchy: out.update({'organ_logits':self.organ_head(feat),'category_logits':self.category_head(feat),'family_logits':self.family_head(feat)})
        return out
