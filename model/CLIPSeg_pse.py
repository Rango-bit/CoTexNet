from typing import Optional, Union, Tuple, Any
from dataclasses import dataclass
import torch
import math
import copy

from torch import nn
from transformers.models.clipseg.modeling_clipseg import (CLIPSegOutput,
                                                          CLIPSegPreTrainedModel,
                                                          CLIPSegConfig,
                                                          CLIPSegDecoderLayer,
                                                          CLIPSegDecoderOutput)
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils.generic import ModelOutput

from model.CLIPSeg_code import CLIPSegModel


class init_weight(object):
    def __init__(self, device, epoch: int, current_epoch: int = 0, max_epoch: int=50, step: int = 1, init_value: float = 0.):
        self.epoch = epoch
        self.device = device
        self.current_epoch = current_epoch
        self.step = step
        self.init_value = init_value
        self.weight = torch.FloatTensor([init_value]).to(device)
        self.max_epoch = max_epoch

    def update(self):
        self.current_epoch += 1
        if self.current_epoch % self.step == 0:
            self.weight = self.init_value + (1.0 - self.init_value) / 2.0 * (self.current_epoch / self.max_epoch)
        if self.current_epoch >= self.max_epoch:
            self.current_epoch = 0

    def reset(self):
        self.current_epoch = 0
        self.weight = torch.FloatTensor([self.init_value]).to(self.device)

    @property
    def value(self):
        return self.weight



@dataclass
class CLIPSegOut(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    conditional_embeddings: torch.FloatTensor = None
    pooled_output: torch.FloatTensor = None
    vision_model_output: BaseModelOutputWithPooling = None
    decoder_output: CLIPSegDecoderOutput = None
    img_embedding: torch.FloatTensor = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["vision_model_output", "decoder_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


@dataclass
class CLIPSegDecoderOutput(ModelOutput):
    """
    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, height, width)`):
            Classification scores for each pixel.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class CLIPseg_pse(CLIPSegPreTrainedModel):
    config_class = CLIPSegConfig

    def __init__(self, config: CLIPSegConfig, num_classes, img_projection):
        super().__init__(config)

        self.config = config
        self.clip = CLIPSegModel(config)
        self.extract_layers = config.extract_layers

        self.decoder = CLIPSegDecoder_plus(num_classes, config)
        # Initialize weights and apply final processing
        self.post_init()

        self.img_projection = img_projection
        self.weight_alpha = nn.Parameter(torch.FloatTensor([0.2]), requires_grad=True)

        self.sigmoid = nn.Sigmoid()
        self.var = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.mean = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.var.data.fill_(1.0)
        self.mean.data.fill_(-0.002)
        self.drop = nn.Dropout(p=0.2)

    def get_conditional_embeddings(
        self,
        batch_size: int = None,
        input_ids: Optional[torch.Tensor] = None,
        roi_embedding: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ):
        # compute conditional embeddings from texts
        if len(input_ids) != batch_size:
            raise ValueError("Make sure to pass as many prompt texts as there are query images")


        # with torch.no_grad(): # 会禁止梯度计算
        conditional_embeddings = self.clip.get_text_features(
            input_ids, attention_mask=attention_mask, position_ids=position_ids, roi_embedding=roi_embedding
        )

        return conditional_embeddings


    def forward(
        self,
        use_cls_token: bool = False,
        roi_embedding: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, CLIPSegOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the query images through the frozen CLIP vision encoder
        with torch.no_grad():
            vision_outputs = self.clip.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=True,  # we need the intermediate hidden states
                return_dict=return_dict,
            )

            cls_token = vision_outputs[0][:, 0]

            pooled_output = self.clip.visual_projection(vision_outputs[1])

            hidden_states = vision_outputs.hidden_states if return_dict else vision_outputs[2]
            # we add +1 here as the hidden states also include the initial embeddings
            activations = [hidden_states[i + 1] for i in self.extract_layers]   # 图像特征

            # update vision_outputs
            if return_dict:
                vision_outputs = BaseModelOutputWithPooling(
                    last_hidden_state=vision_outputs.last_hidden_state,
                    pooler_output=vision_outputs.pooler_output,
                    hidden_states=vision_outputs.hidden_states if output_hidden_states else None,
                    attentions=vision_outputs.attentions,
                )
            else:
                vision_outputs = (
                    vision_outputs[:2] + vision_outputs[3:] if not output_hidden_states else vision_outputs
                )

        # img_embedding = self.img_projection(pooled_output)
        img_embedding = self.img_projection(cls_token)
        img_embedding = self.sigmoid(img_embedding)


        # In the first stage of training, text embedding obtained using text enocder is used as the text input
        conditional_embeddings = self.get_conditional_embeddings(
            batch_size=pixel_values.shape[0],
            input_ids=input_ids,
            roi_embedding=roi_embedding,
            attention_mask=attention_mask,
            position_ids=position_ids
        )

        conditional_embeddings = self.sigmoid(conditional_embeddings)

        # This parameter is used for the subsequent calculation of the fitting loss
        condit_embed_temp = conditional_embeddings.clone().detach()

        if not use_cls_token: # training phase
            # Noise-added
            conditional_embeddings = (1-self.weight_alpha) * conditional_embeddings + self.weight_alpha * img_embedding

        else: # Testing and validation phase

            # Phase two training as well as test phase
            # The predicted labels obtained from the two-layer linear model trained using regression are used as text inputs
            conditional_embeddings = img_embedding

        # Inverse-z score
        conditional_embeddings = conditional_embeddings * self.var + self.mean

        # step 3: forward both the pooled output and the activations through the lightweight decoder to predict masks
        decoder_outputs = self.decoder(
            activations,
            conditional_embeddings,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = decoder_outputs.logits if return_dict else decoder_outputs[0]

        loss = None
        if labels is not None:
            # move labels to the correct device to enable PP
            labels = labels.to(logits.device)
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)

        if not return_dict:
            output = (logits, conditional_embeddings, pooled_output, vision_outputs, decoder_outputs, img_embedding)
            return ((loss,) + output) if loss is not None else output

        return CLIPSegOut(
            loss=loss,
            logits=logits,
            conditional_embeddings=condit_embed_temp,
            pooled_output=pooled_output,
            vision_model_output=vision_outputs,
            decoder_output=decoder_outputs,
            img_embedding=img_embedding
        )

class CLIPSegDecoder_plus(CLIPSegPreTrainedModel):
    def __init__(self, num_classes, config: CLIPSegConfig):
        super().__init__(config)

        self.conditional_layer = config.conditional_layer

        self.film_mul = nn.Linear(config.projection_dim, config.reduce_dim)
        self.film_add = nn.Linear(config.projection_dim, config.reduce_dim)

        if num_classes == 2:
            if config.use_complex_transposed_convolution:
                transposed_kernels = (config.vision_config.patch_size // 4, config.vision_config.patch_size // 4)

                self.transposed_convolution = nn.Sequential(
                    nn.Conv2d(config.reduce_dim, config.reduce_dim, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(
                        config.reduce_dim,
                        config.reduce_dim // 2,
                        kernel_size=transposed_kernels[0],
                        stride=transposed_kernels[0],
                    ),
                    nn.ReLU(),
                    nn.ConvTranspose2d(
                        config.reduce_dim // 2, 1, kernel_size=transposed_kernels[1], stride=transposed_kernels[1]
                    ),
                )
            else:
                self.transposed_convolution = nn.ConvTranspose2d(
                    config.reduce_dim, 1, config.vision_config.patch_size, stride=config.vision_config.patch_size
                )
        elif num_classes > 2:
            if config.use_complex_transposed_convolution:
                transposed_kernels = (config.vision_config.patch_size // 4, config.vision_config.patch_size // 4)
                self.transposed_convolution = nn.Sequential(
                    nn.Conv2d(config.reduce_dim, config.reduce_dim, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(
                        config.reduce_dim,
                        config.reduce_dim // 2,
                        kernel_size=transposed_kernels[0],
                        stride=transposed_kernels[0],
                    ),
                    nn.ReLU(),
                )
                self.output_convolution = nn.ConvTranspose2d(
                    config.reduce_dim // 2, num_classes, kernel_size=transposed_kernels[1], stride=transposed_kernels[1]
                )
            else:
                self.transposed_convolution = nn.Identity()
                self.output_convolution = nn.ConvTranspose2d(
                    config.reduce_dim, num_classes, config.vision_config.patch_size,
                    stride=config.vision_config.patch_size
                )
        else:
            raise ValueError("Incorrect value for 'num_classes'!")
        self.num_classes = num_classes

        depth = len(config.extract_layers)
        self.reduces = nn.ModuleList(
            [nn.Linear(config.vision_config.hidden_size, config.reduce_dim) for _ in range(depth)]
        )

        decoder_config = copy.deepcopy(config.vision_config)
        decoder_config.hidden_size = config.reduce_dim
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        decoder_config.hidden_act = "relu"
        self.layers = nn.ModuleList([CLIPSegDecoderLayer(decoder_config) for _ in range(len(config.extract_layers))])


    def forward(
        self,
        hidden_states: Tuple[torch.Tensor],
        conditional_embeddings: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None


        activations = hidden_states[::-1] # bs 485 768 逆序保存

        output = None

        for i, (activation, layer, reduce) in enumerate(zip(
                activations, self.layers, self.reduces)):
            if output is not None:
                output = reduce(activation) + output
            else:
                output = reduce(activation)

            if i == self.conditional_layer:
                output = self.film_mul(conditional_embeddings) * output.permute(1, 0, 2) + self.film_add(
                    conditional_embeddings
                )
                output = output.permute(1, 0, 2)

            layer_outputs = layer(
                output, attention_mask=None, causal_attention_mask=None, output_attentions=output_attentions
            )

            output = layer_outputs[0]

            if output_hidden_states:
                all_hidden_states += (output,)

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        output = output[:, 1:, :].permute(0, 2, 1)  # remove cls token and reshape to [batch_size, reduce_dim, seq_len]

        size = int(math.sqrt(output.shape[2]))

        batch_size = conditional_embeddings.shape[0]
        output = output.view(batch_size, output.shape[1], size, size)

        logits = self.transposed_convolution(output)

        if self.num_classes > 2:
            logits = self.output_convolution(logits)

        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_attentions] if v is not None)

        return CLIPSegDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_attentions
        )


class CLIPSegPlus(nn.Module):
    def __init__(self, num_classes, clipseg_hf_api, img_projection,
                 freeze_encoder=True, freeze_decoder=False):
        super().__init__()

        self.clipseg = CLIPseg_pse.from_pretrained(clipseg_hf_api, num_classes=num_classes, img_projection=img_projection)
        self.clipseg.clip.requires_grad_(not freeze_encoder)
        self.clipseg.decoder.requires_grad_(not freeze_decoder)


    def forward(self, **kwargs):
        outputs = self.clipseg(**kwargs)

        return outputs
