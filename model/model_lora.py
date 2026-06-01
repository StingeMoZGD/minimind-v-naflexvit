import torch
from torch import optim, nn


class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)
    def forward(self, x):
        return self.B(self.A(x)) * self.scaling

TARGETS = ['qkv', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'out_proj']
def apply_lora(model, rank=32, lora_target=0):
    """
    对模型的不同部分应用 LoRA。
    
    Args:
        model: 模型实例
        rank: LoRA 的秩
        lora_target: 
            0: 对 model.layers, vision_encoder, vision_proj 全部应用 LoRA
            1: 只对 vision_encoder 应用 LoRA
            (可扩展其他值，例如 2: 只对 vision_proj 等)
    """
    # 根据 lora_target 确定要处理的模块名前缀列表
    if lora_target == 0:
        allowed_prefixes = ['model.layers', 'vision_encoder', 'vision_proj']
    elif lora_target == 1:
        allowed_prefixes = ['model.layers', 'vision_proj']
    else:
        # 默认只对 vision_encoder 和 vision_proj（可自定义）
        allowed_prefixes = ['vision_encoder', 'vision_proj']
    # 记录已经处理过的模块，避免重复添加
    processed = set()
    for name, module in model.named_modules():
        # 跳过已经是 LoRA 层的内部 Linear（如 lora.A, lora.B）
        if 'lora' in name:
            continue
        if id(module) in processed:
            continue
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.startswith(prefix) for prefix in allowed_prefixes):
            continue
        if not any(t in name for t in TARGETS):
            continue
        lora = LoRA(module.in_features, module.out_features, rank).to(module.weight.device)
        module.lora = lora
        processed.add(id(module))
        original_forward = module.forward
        def forward_with_lora(x, layer=original_forward, lora_layer=lora):
            return layer(x) + lora_layer(x)
        module.forward = forward_with_lora
        print(f"LoRA attached to {name} (rank={rank})")
    return model


def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)


def merge_lora(model, lora_path, save_path):
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)
