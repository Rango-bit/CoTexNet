import torch
import torch.nn as nn

from model.CLIPSeg_pse import CLIPSegPlus
from transformers import CLIPSegForImageSegmentation


class ImgToText(nn.Module):
    def __init__(self, clipseg_hf_api):
        super(ImgToText, self).__init__()
        self.vision_model = CLIPSegForImageSegmentation.from_pretrained(clipseg_hf_api).clip.vision_model

        self.vision_model.requires_grad_(False)
        # self.vision_linear = nn.Linear(in_features=768, out_features=512)
        self.mapping_module = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512)  # Ensure that the final feature distribution is stable
        )

    def forward(self, x):
        x = self.vision_model(pixel_values=x,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True)
        x = x[0][:,0,:]  # cls token
        x_embedding = self.mapping_module(x)

        return x_embedding

def assign_value(model):
    for idx, params in model.named_parameters():
        if 'output_convolution' in idx:
            print(idx)
            if 'weight' in idx:
                nn.init.xavier_normal_(params)
            elif 'bias' in idx:
                nn.init.constant_(params, 0)
            else:
                raise ValueError

class Img2text_module(nn.Module):
    def __init__(self, embed_dim=768, middle_dim=768, output_dim=512, n_layer=2, dropout=0.):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        self.fc_fn = nn.Sigmoid()
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.ReLU())
            dim = middle_dim
            layers.append(nn.Sequential(*block))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        x = self.fc_fn(self.fc_out(x))
        return x


class CLIPSeg_fintune(nn.Module):
    def __init__(self,
                 clipseg_hf_api,
                 num_classes=2,
                 freeze_clipseg_encoder: bool = True,
                 freeze_clipseg_decoder: bool = False
                 ):
        super().__init__()

        self.num_classes = num_classes
        self.im2text = ImgToText(clipseg_hf_api)
        self.img_projection = Img2text_module(embed_dim=768)

        self.clipsegp = CLIPSegPlus(num_classes, clipseg_hf_api, self.img_projection,
                                 freeze_clipseg_encoder, freeze_clipseg_decoder)
        assign_value(self.clipsegp)
        self.sigmoid = nn.Sigmoid()

    def forward(self, **kwargs):

        if 'roi_img' in kwargs:
            kwargs['roi_embedding'] = self.im2text(kwargs['roi_img'])
        else:
            kwargs['roi_embedding'] = None
        del kwargs['roi_img']
        
        output = self.clipsegp(**kwargs)

        if self.num_classes == 2:
            logits = self.sigmoid(output.logits)
        else:
            logits = self.sigmoid(output.logits)


        return logits, output.img_embedding, output.conditional_embeddings
