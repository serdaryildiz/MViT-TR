import torch.nn as nn

from Model.attention import PositionAttention
from Model.backbone import VisionTransformer


class TTR(nn.Module):
    def __init__(self, args: dict):
        super().__init__()
        self.args = args

        self.backbone = VisionTransformer(img_size=args["img_size"],
                                          patch_size=args["patch_size"],
                                          in_channels=3,
                                          embed_dim=args["embed_dim"],
                                          num_heads=args["num_heads"],
                                          mask_ratio=args["mask_ratio"])

        self.positionAttention = PositionAttention(max_length=26,
                                                   in_channels=args["embed_dim"],
                                                   num_channels=args["position_attention_hidden"],
                                                   h=args["img_size"][0] // args["patch_size"][0],
                                                   w=args["img_size"][1] // args["patch_size"][1],
                                                   mode='nearest')
        self.cls = nn.Linear(args["embed_dim"], 43)
        return

    def forward(self, image):
        features = self.backbone(image)
        attn_vecs, attn_scores = self.positionAttention(features)
        logits = self.cls(attn_vecs)
        return logits
