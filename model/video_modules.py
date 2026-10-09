import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class DynamicFeatureModule(nn.Module):
    def __init__(self, in_dim, key_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, key_dim, kernel_size=3, padding=1)
        self.shrink = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        self.select = nn.Conv2d(in_dim, key_dim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.proj.weight.data)
        nn.init.zeros_(self.proj.bias.data)

    def forward(self, x):
        shrink_factor = self.shrink(x) ** 2 + 1
        select_mask = torch.sigmoid(self.select(x))
        return self.proj(x), shrink_factor, select_mask


class ValueEncoder(nn.Module):
    def __init__(self, key_dim, value_dim, hidden_dim, single_object=False):
        super().__init__()
        self.single_object = single_object
        self.key_dim = key_dim
        self.value_dim = value_dim

        self.conv_fusion = nn.Conv2d(
            in_channels=key_dim + 1,
            out_channels=value_dim,
            kernel_size=3,
            padding=1
        )
        self.bn = nn.BatchNorm2d(value_dim)
        self.relu = nn.ReLU(inplace=True)

        if hidden_dim > 0:
            self.hidden_reinforce = HiddenReinforcer(value_dim, hidden_dim)
        else:
            self.hidden_reinforce = None

    def forward(self, key, select, h=None, is_deep_update=True):
        batch_size, num_objects = select.shape[:2]
        key_expanded = key.unsqueeze(1).repeat(1, num_objects, 1, 1, 1)  # (batch_size, num_objects, key_dim, H, W)
        key_expanded = key_expanded.view(batch_size * num_objects, self.key_dim, *key.shape[-2:])

        fused_features = torch.cat([key_expanded, select], dim=1)  # (batch_size * num_objects, key_dim + 1/2, H, W)

        g = self.conv_fusion(fused_features)  # (batch_size * num_objects, value_dim, H, W)
        g = self.bn(g)
        g = self.relu(g)

        g = g.view(batch_size, num_objects, *g.shape[1:])

        if is_deep_update and self.hidden_reinforce is not None and h is not None:
            h = self.hidden_reinforce(g, h)

        return g.unsqueeze(3), h


class HiddenReinforcer(nn.Module):
    # Used in the value encoder, a single GRU
    def __init__(self, g_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.transform = GConv2D(g_dim + hidden_dim, hidden_dim * 3, kernel_size=3, padding=1)
        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        g = torch.cat([g, h], 2)

        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:, :, :self.hidden_dim])
        update_gate = torch.sigmoid(values[:, :, self.hidden_dim:self.hidden_dim * 2])
        new_value = torch.tanh(values[:, :, self.hidden_dim * 2:])
        new_h = forget_gate * h * (1 - update_gate) + update_gate * new_value

        return new_h


class GConv2D(nn.Conv2d):
    def forward(self, g):
        batch_size, num_objects = g.shape[:2]
        g = super().forward(g.flatten(start_dim=0, end_dim=1))
        return g.view(batch_size, num_objects, *g.shape[1:])


def read_memory(query_key, query_selection, memory_key, memory_shrinkage, memory_value):
    """
    query_key       : B * CK * H * W
    query_selection : B * CK * H * W
    memory_key      : B * CK * T * H * W
    memory_shrinkage: B * 1  * T * H * W
    memory_value    : B * num_objects * CV * T * H * W
    """
    batch_size, num_objects = memory_value.shape[:2]
    memory_value = memory_value.flatten(start_dim=1, end_dim=2)

    affinity = get_affinity(memory_key, memory_shrinkage, query_key, query_selection)
    memory = readout(affinity, memory_value)
    memory = memory.view(batch_size, num_objects, *memory.shape[1:])

    return memory  # B num_objects CV H W


def get_affinity(mk, ms, qk, qe):
    # shorthand used in training with no top-k
    similarity = get_similarity(mk, ms, qk, qe)
    affinity = do_softmax(similarity)
    return affinity


def readout(affinity, mv):
    B, CV, T, H, W = mv.shape

    mo = mv.view(B, CV, T * H * W)
    mem = torch.bmm(mo, affinity)
    mem = mem.view(B, CV, H, W)

    return mem


def get_similarity(mk, ms, qk, qe):
    # used for training/inference and memory reading/memory potentiation
    # mk: B x CK x [N]    - Memory keys
    # ms: B x  1 x [N]    - Memory shrinkage
    # qk: B x CK x [HW/P] - Query keys
    # qe: B x CK x [HW/P] - Query selection
    # Dimensions in [] are flattened
    CK = mk.shape[2]
    mk = mk.transpose(1, 2).flatten(start_dim=2)
    ms = ms.transpose(1, 2).flatten(start_dim=1).unsqueeze(2) if ms is not None else None
    qk = qk.flatten(start_dim=2)
    qe = qe.flatten(start_dim=2) if qe is not None else None

    if qe is not None:
        # See appendix for derivation
        # Expanded squared-distance form for memory-key similarity.
        mk = mk.transpose(1, 2)
        a_sq = (mk.pow(2) @ qe)
        two_ab = 2 * (mk @ (qk * qe))
        b_sq = (qe * qk.pow(2)).sum(1, keepdim=True)
        similarity = (-a_sq + two_ab - b_sq)
    else:
        # similar to STCN if we don't have the selection term
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        two_ab = 2 * (mk.transpose(1, 2) @ qk)
        similarity = (-a_sq + two_ab)

    if ms is not None:
        similarity = similarity * ms / math.sqrt(CK)  # B*N*HW
    else:
        similarity = similarity / math.sqrt(CK)  # B*N*HW

    return similarity


def do_softmax(similarity, top_k: Optional[int] = None, inplace=False, return_usage=False):
    # normalize similarity with top-k softmax
    # similarity: B x N x [HW/P]
    # use inplace with care
    if top_k is not None:
        values, indices = torch.topk(similarity, k=top_k, dim=1)

        x_exp = values.exp_()
        x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
        if inplace:
            similarity.zero_().scatter_(1, indices, x_exp)  # B*N*HW
            affinity = similarity
        else:
            affinity = torch.zeros_like(similarity).scatter_(1, indices, x_exp)  # B*N*HW
    else:
        maxes = torch.max(similarity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(similarity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum
        indices = None

    if return_usage:
        return affinity, affinity.sum(dim=2)

    return affinity


class MemoryDecoder(nn.Module):
    def __init__(self, val_dim, hidden_dim):
        super().__init__()

        self.fuser = nn.Sequential(
            nn.Conv2d(val_dim * 2 + hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.hidden_update = nn.Conv2d(hidden_dim*2, hidden_dim, 3, padding=1)

    def forward(self, imge, hidden_state, memory_readout, h_out=True):
        fused = torch.cat([imge, hidden_state.squeeze(1), memory_readout.squeeze(1)], dim=1)
        fused = self.fuser(fused)  # [B, D, H, W]

        update_gate = torch.sigmoid(self.hidden_update(torch.cat([fused, hidden_state.squeeze(1)], dim=1)))
        new_hidden = update_gate * hidden_state.squeeze(1) + (1 - update_gate) * fused

        return new_hidden.unsqueeze(1), fused.unsqueeze(1)


class HiddenUpdater(nn.Module):
    # Used in the decoder, multi-scale feature + GRU
    def __init__(self, mid_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.transform = GConv2D(mid_dim + hidden_dim, hidden_dim * 3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):

        g = torch.cat([g[0], h], 2)

        # defined slightly differently than standard GRU,
        # namely the new value is generated before the forget gate.
        # The new value is computed before the forget gate.
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:, :, :self.hidden_dim])
        update_gate = torch.sigmoid(values[:, :, self.hidden_dim:self.hidden_dim * 2])
        new_value = torch.tanh(values[:, :, self.hidden_dim * 2:])
        new_h = forget_gate * h * (1 - update_gate) + update_gate * new_value

        return new_h
