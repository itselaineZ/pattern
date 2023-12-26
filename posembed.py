import numpy as np

def get_1d_sincos_pos_embed(embed_dim, pos):
    """
    embed_dim: 对每个位置输出的编码数量
    pos: 需要被编码的位置序列, 即公式中的pos长度为n
    return: [n, embed_dim]
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega = omega * 2. / embed_dim
    omega = 1. / (10000**omega)
    
    pos = pos.reshape(-1) # 拉平成一行
    out = np.einsum('i,j->ij', pos, omega) # 向量外积，[n, embed_dim/2]
    
    embed_sin = np.sin(out)
    embed_cos = np.cos(out)
    embed = np.concatenate([embed_sin, embed_cos], axis = 1) # 在第一维拼接，[n, embed_dim]
    return embed

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token = False):
    """
    embed_dim: 对每个位置输出的编码数量
    grid_size: 一个int表示grid的高和宽
    return: [grid_size * grid_size, embed_dim] 或者 [1 + grid_size*grid_size, embed_dim]
            取决于是否有cls_token
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_h, grid_w) # 一个包含两个array的list, [第0维扩展，第1维扩展]
    grid = np.stack(grid, axis=0) # 堆成一个list, [2, grid_size, grid_size]
    # grid = grid.reshape([2, 1, grid_size, grid_size])
    
    embed_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    embed_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    pos_embed = np.concatenate([embed_h, embed_w], axis=1)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

# get_1d_sincos_pos_embed(4, np.arange(4, dtype=float))
# print(get_2d_sincos_pos_embed(4, 4))

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed