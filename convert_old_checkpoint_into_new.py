import sys
import torch

def convert_timm_swim_weight_to_hf_swin_weight(timm_weight):
    embeddings_weight_mapping = {
        'encoder.patch_embed.proj.weight':'encoder.embeddings.patch_embeddings.projection.weight', 
        'encoder.patch_embed.proj.bias'  :'encoder.embeddings.patch_embeddings.projection.bias',
        'encoder.patch_embed.norm.weight':'encoder.embeddings.norm.weight', 
        'encoder.patch_embed.norm.bias'  :'encoder.embeddings.norm.bias', 
    }
    state_dict = {}
    for key, val in timm_weight.items():
        if not  key.startswith('encoder.'):
            state_dict[key]=   val     
            continue
        if key.startswith('encoder.patch_embed'):
            new_key = embeddings_weight_mapping[key]
            state_dict[new_key]=   val   
            continue 
        if key.startswith('encoder.norm.'):
            new_key = key.replace('norm.','layernorm.')
            state_dict[new_key]=   val   
            continue 
        key = 'encoder.'+key
        if 'qkv' in key:
            q_weight, k_weight, v_weight = val.chunk(3)
            new_key = key.replace('attn.qkv.','attention.self.query.')
            state_dict[new_key]=q_weight
            new_key = key.replace('attn.qkv.','attention.self.key.')
            state_dict[new_key]=k_weight
            new_key = key.replace('attn.qkv.','attention.self.value.')
            state_dict[new_key]=v_weight
            continue
        if 'attn_mask' in key:continue
        new_key = key.replace('norm1.','layernorm_before.'
                    ).replace('attn.','attention.self.'
                    ).replace('self.proj.','output.dense.'
                    ).replace('mlp.fc1.','intermediate.dense.'
                    ).replace('mlp.fc2.','output.dense.'
                    ).replace('norm2.','layernorm_after.'
                    )
        state_dict[new_key]=   val     
    return state_dict

old_path = sys.argv[1]#'/mnt/data/oss_beijing/sunyu/nougat/PromptNougat/result/nougat/20240309/last.ckpt'
new_path = sys.argv[2]
weight = torch.load(old_path)

new_state_dict = {}
for key,val in weight.items():
    key = key.replace('model.','')
    new_state_dict[key] = val

new_state_dict = convert_timm_swim_weight_to_hf_swin_weight(new_state_dict)
torch.save(new_state_dict, new_path)