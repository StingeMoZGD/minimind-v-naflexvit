import os
import torch
import warnings
from .model_minimind import *
from typing import Optional, Tuple, List, Union
from torch import nn
from transformers import SiglipImageProcessor, SiglipVisionModel
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
# from transformers import NaFlexVitModel, NaFlexVitImageProcessor
warnings.filterwarnings('ignore')

# 修改 VLMConfig，可以增加动态分辨率相关配置（可选）
class VLM_naflexvitConfig(MiniMindConfig):
    model_type = "minimind-v"

    def __init__(self, image_special_token='<|image_pad|>', image_ids=[12], **kwargs):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        self.image_hidden_size = kwargs.get("image_hidden_size", 768)   # NaFlexVit-base 也是 768
        self.image_token_len = kwargs.get("image_token_len", 128)       # NaFlexVit 默认 patch 输出数量可变，需根据实际动态调整
        self.projector_impl = "cross-attn"                              # cross-attn or mlp
        super().__init__(**kwargs)

class PerceiverBlock(nn.Module):
    def __init__(self, dim, num_heads=8, ff_mult=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.dropout = dropout
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        # Q, K, V 投影层（不使用内置 MultiheadAttention，以便灵活调用 SDPA）
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),          # 添加 FFN 内的 dropout
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout)
        )
        self.attn_dropout = nn.Dropout(dropout)
        
    def forward(self, q, kv):
        B, N_q, D = q.shape
        B, N_kv, D = kv.shape
        # Pre‑norm
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        # 投影并切分为多头
        Q = self.q_proj(q_norm).view(B, N_q, self.num_heads, -1).transpose(1, 2)  # [B, H, N_q, head_dim]
        K = self.k_proj(kv_norm).view(B, N_kv, self.num_heads, -1).transpose(1, 2)
        V = self.v_proj(kv_norm).view(B, N_kv, self.num_heads, -1).transpose(1, 2)
        # 使用 Flash Attention 加速
        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale
        )  # [B, H, N_q, head_dim]
        # 合并多头并输出
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N_q, D)
        attn_out = self.out_proj(attn_out)
        attn_out = self.attn_dropout(attn_out)
        
        q = q + attn_out
        q = q + self.ffn(q)
        return q

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        dim,
        num_queries=128,
        depth=2,
        num_heads=8
    ):
        super().__init__()
        self.queries = nn.Parameter(
            torch.randn(num_queries, dim) * 0.02
        )
        self.layers = nn.ModuleList([
            PerceiverBlock(
                dim=dim,
                num_heads=num_heads
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        B = x.shape[0]
        q = self.queries.unsqueeze(0).expand(
            B,
            -1,
            -1
        )
        for layer in self.layers:
            q = layer(q, x)

        return q

class MMVisionProjector(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        target_tokens=128,
        projector_impl = "cross_attn",
    ):
        super().__init__()
        self.target_tokens = target_tokens
        if projector_impl == "cross_attn":
            self.resampler = PerceiverResampler(
                dim=in_dim,
                num_queries=target_tokens,
                depth=2,
                num_heads=8
            )
            print("projector: cross_attn")
        else:
            self.resampler = None
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        if self.resampler != None:
            x = self.resampler(x)
        else:
            if x.shape[1] != self.target_tokens:
                x = x.transpose(1,2)
                x = F.adaptive_avg_pool1d(
                    x,
                    self.target_tokens
                )
                x = x.transpose(1,2)
        x = self.mlp(x)
        print(f"projector: {x.shape}")
        return x

    def forward(self, x):
        if x.shape[1] != self.target_tokens:
            x = x.transpose(1,2)
            x = F.adaptive_avg_pool1d(
                x,
                self.target_tokens
            )
            x = x.transpose(1,2)
        return self.mlp(x)

# 修改 MiniMindVLM 的 __init__ 默认路径
class MiniMindVLM_naflexvit(MiniMindForCausalLM):
    config_class = VLM_naflexvitConfig

    def __init__(self, config: VLM_naflexvitConfig = None, vision_model_path="google/nativeflexvit-base"):
        self.config = config or VLM_naflexvitConfig()
        super().__init__(self.config)
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)
        # 投影层的输入维度需要与视觉编码器的输出维度一致
        vision_hidden_size = self.config.image_hidden_size  # 动态获取
        self.vision_proj = MMVisionProjector(vision_hidden_size, self.config.hidden_size, 
                                             target_tokens=self.config.image_token_len)
    @staticmethod
    def get_vision_model(model_path: str):
        import timm
        from torchvision import transforms
        # 注意：model_path 可以是本地路径或 huggingface 模型名
        try:
            model = timm.create_model(
                    "naflexvit_base_patch16_parfac_gap",
                    pretrained=False,
                    num_classes=0,
                    dynamic_img_size=True
                )
            checkpoint_path = '/data1/yym/MINIMIND/minimind-v-master/model/NaFlexvit/pytorch_model.bin'
            # 或 .safetensors 文件，可以先用 torch.load 或 safetensors 读取
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            print(f"vision_encoder:naflexvit_base_patch16_parfac_gap")
            processor = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485,0.456,0.406),
                        std=(0.229,0.224,0.225)
                    )
                ])
        except (RuntimeError, ValueError, OSError):
            # 如果加载失败，回退到本地的 Siglip（保留原逻辑）
            # 这里为了简洁，直接抛出异常或返回 None
            raise ValueError(f"Failed to load NaFlexVit model from {model_path}")
        # 冻结视觉编码器
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        inputs = processor(image)
        return inputs

    @staticmethod
    def image2tensor_web(image, processor):
        if image.mode in ['RGBA', 'LA']: 
            image = image.convert('RGB')
        inputs = processor(image)
        # 如果 processor 是 torchvision.transforms.Compose，它会返回一个张量
        # 将张量包装成字典，key 根据模型需要决定（通常是 "pixel_values"）
        return {"pixel_values": inputs.unsqueeze(0)}

    @staticmethod
    def image2tensor_traing(image, processor):
        from torchvision import transforms
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 固定尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        inputs = transform(image)
        return inputs

    @staticmethod
    def get_image_embeddings(image_inputs, vision_model):
        # NaFlexVit 的 forward 直接接受 processor 返回的字典
        if hasattr(image_inputs, 'keys'):
            # 去除多余的 batch 维度（如果 processor 返回了 5D 张量）
            image_inputs = {k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v for k, v in image_inputs.items()}
        with torch.no_grad():
            if isinstance(image_inputs, dict):
                # print(type(pixel_values))
                # print(pixel_values.shape)
                x = image_inputs["pixel_values"]
            else:
                x = image_inputs
            # print(x.shape)
            if x.ndim == 5 and x.shape[1] == 1:
                x = x.squeeze(1)
            if x.dtype != next(vision_model.parameters()).dtype:
                x = x.to(next(vision_model.parameters()).dtype)
            # print("inputs", x.dtype)
            # print("outputs", next(vision_model.parameters()).dtype)
            outputs = vision_model.forward_features(x)
            # print("outputs", outputs.shape)
        # NaFlexVit 输出 last_hidden_state，形状为 (batch, num_patches, hidden_size)
        return outputs[:, 4:, :]

    @torch.compiler.disable
    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        if vision_tensors is None or not self.config.image_ids:
            return h
        marker, vf = self.config.image_ids[0], vision_tensors
        if vf.dim() == 3:
            vf = vf.unsqueeze(1)
        out = []
        for b in range(h.size(0)):
            hb, seq, k, i = h[b], tokens[b].tolist(), 0, 0
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if k < vf.size(1):
                        hb = torch.cat((hb[:start], vf[b][k][:i - start], hb[i:]), dim=0)[:seqlen]
                        k += 1
                else:
                    i += 1
            out.append(hb)
        return torch.stack(out)

    # forward 中关于 pixel_values 的分支逻辑无需改动，因为 get_image_embeddings 返回统一格式
    # 但需要关注 image_token_len 的配置：NaFlexVit 输出 patch 数量随输入分辨率变化，
    # 你的 count_vision_proj 方法依赖 image_token_len 和实际 vision_tensors 的第二维。
    # 建议在 forward 中动态设置 image_token_len：
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                labels: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        if pixel_values is not None and start_pos == 0:
            # 获取视觉特征
            # print(type(pixel_values))
            # print(pixel_values.shape)
            image_embeds = self.get_image_embeddings(pixel_values, self.vision_encoder)  # (bs, num_patches, hidden)
            # 投影到 LLM 维度
            vision_tensors = self.vision_proj(image_embeds)  # (bs, num_patches, llm_hidden)
            # 动态更新 image_token_len 为实际 patches 数量
            self.config.image_token_len = vision_tensors.shape[1]
            # 后续替换 token 的位置（count_vision_proj）会自动使用 vision_tensors 的第二维
            hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states, 
                                                   vision_tensors=vision_tensors, 
                                                   seqlen=input_ids.shape[1])
            
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.model.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.model.freqs_cos, self.model.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        aux_loss = sum([l.mlp.aux_loss for l in self.model.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        aux_loss = aux_loss + sum(p.sum() for p in self.vision_proj.parameters()) * 0  # dummy gradient for DDP
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        output = MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=presents, hidden_states=hidden_states)
        return output

    def generate(self, *args, num_return_sequences=1, **kwargs):
        if num_return_sequences > 1 and 'pixel_values' in kwargs:
            pv = kwargs['pixel_values']
            if hasattr(pv, 'keys'):
                kwargs['pixel_values'] = {k: v.repeat(num_return_sequences, *([1] * (v.ndim - 1))) for k, v in pv.items()}
            else:
                kwargs['pixel_values'] = pv.repeat(num_return_sequences, *([1] * (pv.ndim - 1)))
        return super().generate(*args, num_return_sequences=num_return_sequences, **kwargs)