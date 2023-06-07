import math
from einops import rearrange
import torch 
import dgl
import torch.nn.functional as F


def generate_swin_causal_mask(current_size: int, max_size: int = 256, kernel_size : int = 3, device='cuda'):
    r"""Generate a convolutional mask for the sequence, to compute attention only on adjacent elements, 
    like in a convolution kernel. The masked positions are filled with float('-inf'). 
    Unmasked positions are filled with float(0.0).
    """
    h =  w = int(math.sqrt(max_size))
    keep = kernel_size **2 // 2
    seq_idxs = torch.tensor(range(max_size), dtype=torch.float32, device=device) + 1 # Add one to distinguidh 0 index from padding
    seq_idxs = rearrange(seq_idxs, '(h w) -> 1 1 h w', h=h)

    padding = kernel_size // 2
    swin = F.unfold(seq_idxs.float(), kernel_size, padding=padding) # Defines the matrix for convolutional operations
    swin = swin[0, :keep, :].type(torch.int64).clone()

    mask = torch.zeros(max_size, max_size, device=device)
    mask = mask.scatter_add(1, swin.t() , torch.ones(max_size,max_size, device=device)) # After having defined the indices with swin, I add a one in the positions defined by the indices
    # mask = mask[:-1,1:] # remove the last row and first column to compensate the padding
    mask[:,0] = 1 # fill the first column to compensate the padding (this is the attention on the SOS token)
    mask = mask[:current_size,:current_size].bool().clone()
    return torch.where(mask != 0, 0.0, float('-inf'))

def generate_square_subsequent_mask(sz: int):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

def remove_sos(x):
    return x[1:]

def remove_last(x):
    return x[:-1]

def get_next_token(probs, type='multinomial'):
    if type == 'multinomial':
        return torch.multinomial(probs, num_samples=1)
    elif type == 'greedy':
        return probs.max(-1)[1].unsqueeze(1)

def add_sos(tgt, K, batch_size=None):
    if not batch_size:
        batch_size = tgt.shape[1]
    start = torch.zeros(batch_size, dtype=torch.int32).unsqueeze(0).to(tgt.device).fill_(K)
    return torch.cat((start, tgt))

def make_graph_padding_mask(graph, pad_idx=0):
    h = graph.ndata['feat']
    nnodes = dgl.unbatch(graph)[0].num_nodes()
    h = rearrange(h, '(b s) -> b s', s=nnodes)
    return h == pad_idx

def configure_optimizers(trf, lr):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear)
        blacklist_weight_modules = (
            torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.Parameter)

        for mn, m in trf.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif (pn.endswith('weight') and isinstance(m, whitelist_weight_modules)) or \
                        ('decoder' in mn):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif (pn.endswith('weight') and isinstance(m, blacklist_weight_modules)):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in trf.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(
            inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn]
                        for pn in sorted(list(decay))], "weight_decay": 1e-4},
            {"params": [param_dict[pn]
                        for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95))
        return optimizer