import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassAwareNeighborAttention(nn.Module):
    """
    Refine each frame using neighboring frames and class-distribution similarity.
    """
    def __init__(self, dim, num_classes, kernel_size=3):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.pad = kernel_size // 2
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.num_classes = num_classes

        # Estimate class distributions used to gate neighbor attention.
        self.tmp_classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        """
        x: Frame features [N, T, C].
        Returns: Refined frame features [N, T, C].
        """
        N, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Preliminary class distribution [N, T, num_classes].
        logits = self.tmp_classifier(x)
        probs = logits.softmax(dim=-1)

        out = []
        for t in range(T):
            left, right = max(0, t - self.pad), min(T, t + self.pad + 1)

            q_t = q[:, t:t+1, :]           # [N, 1, C]
            k_t = k[:, left:right, :]      # [N, W, C]
            v_t = v[:, left:right, :]      # [N, W, C]

            attn = (q_t @ k_t.transpose(-2, -1)) / (C ** 0.5)  # [N, 1, W]
            attn = attn.softmax(dim=-1)

            # Gate attention by class-distribution similarity.
            prob_t = probs[:, t:t+1, :]             # [N, 1, num_classes]
            prob_neighbors = probs[:, left:right, :]# [N, W, num_classes]
            # Cosine similarity [N, W].
            class_sim = F.cosine_similarity(prob_t, prob_neighbors, dim=-1)
            class_sim = class_sim.unsqueeze(1)      # [N, 1, W]

            gated_attn = attn * class_sim
            gated_attn = gated_attn / (gated_attn.sum(dim=-1, keepdim=True) + 1e-6)

            out_t = gated_attn @ v_t  # [N, 1, C]
            out.append(out_t)

        out = torch.cat(out, dim=1)  # [N, T, C]
        return self.proj(out)
