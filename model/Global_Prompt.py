import torch
import torch.nn as nn

class PromptBank(nn.Module):
    """
    Combine class embeddings and gated experts into a global prompt.

    Accepts one-hot or soft class weights of shape [B, C] and returns [B, d_prompt].
    """
    def __init__(self, num_classes: int, d_prompt: int, num_experts: int = 4, gate_hidden: int = 128, drop: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.d_prompt = d_prompt
        self.num_experts = num_experts

        # Learnable class embeddings.
        self.class_table = nn.Parameter(torch.randn(num_classes, d_prompt) * 0.02)

        # K experts per class.
        self.experts = nn.Parameter(torch.randn(num_classes, num_experts, d_prompt) * 0.02)

        # Map class weights to expert weights.
        self.gate = nn.Sequential(
            nn.Linear(num_classes, gate_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(gate_hidden, num_experts)
        )

    def forward(self, prompt_onehot: torch.Tensor):
        """
        prompt_onehot: One-hot or soft class weights [B, C].
        Returns: Global prompt [B, d_prompt].
        """
        B, C = prompt_onehot.shape
        class_w = prompt_onehot  # [B, C]

        # Weighted class embedding.
        class_center = class_w @ self.class_table  # [B, d_prompt]

        # Sample-specific expert weights.
        gate_logits = self.gate(class_w)  # [B, K]
        gate = torch.softmax(gate_logits, dim=-1)  # [B, K]

        # Aggregate experts across classes, then across experts.
        experts = self.experts  # [C, K, d]
        fused_experts = torch.einsum('bc,ckd->bkd', class_w, experts)  # [B, K, d_prompt]

        experts_agg = torch.einsum('bk,bkd->bd', gate, fused_experts)

        # Add the class and expert contributions.
        out = class_center + experts_agg  # [B, d_prompt]
        return out


class AdaLN(nn.Module):
    """
    Generate channel-wise scale and shift from the global prompt after layer normalization.
    """
    def __init__(self, d_model: int, d_prompt: int, hidden: int = 2 * 128, drop: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.hidden = hidden or (2 * d_model)
        # Initialize the input projection on the first forward pass.
        self.fc_in = None
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc_out = nn.Linear(self.hidden, 2 * d_model)  # -> [B_total, 2D]

    def __init_fc_in(self, in_dim: int, device, dtype):
        if self.fc_in is None:
            self.fc_in = nn.Linear(in_dim, self.hidden).to(device=device, dtype=dtype)

    @staticmethod
    def _expand_gprompt_for_video(gp: torch.Tensor, B_total: int, is_video: bool, tframe: int) -> torch.Tensor:
        """
        Expand clip-level prompts to frames or align prompts to the target batch size.
        """
        B_gp = gp.shape[0]
        if is_video and tframe > 1:
            B_clip = max(B_total // tframe, 1)
            if B_gp == B_clip:
                return gp.repeat_interleave(tframe, dim=0)
            if B_gp == B_total:
                return gp
        if B_gp == 1:
            return gp.expand(B_total, -1)
        if B_gp != B_total:
            rep = (B_total + B_gp - 1) // B_gp
            gp = gp.repeat(rep, 1)[:B_total]
        return gp

    def forward(self, x: torch.Tensor, gprompt: torch.Tensor, is_video: bool = False, tframe: int = 1) -> torch.Tensor:
        """
        x:       [B_total, L, D]
        gprompt: [B_clip, P], [B_total, P], or [1, P]
        """
        if gprompt is None:
            return x

        B_total, L, D = x.shape
        assert D == self.d_model, f"D mismatch: got {D}, expected {self.d_model}"

        # Align prompts with frame-level features.
        gp = self._expand_gprompt_for_video(gprompt, B_total, is_video, tframe)  # [B_total, P]

        self.__init_fc_in(gp.shape[-1], device=x.device, dtype=x.dtype)

        h = self.fc_in(gp)              # [B_total, hidden]
        h = self.act(h)
        h = self.drop(h)
        affine = self.fc_out(h)         # [B_total, 2D]
        scale, shift = affine.chunk(2, dim=-1)  # Each [B_total, D].
        scale = 1.0 + scale

        x = x * scale.unsqueeze(1) + shift.unsqueeze(1)  # Broadcast across tokens.
        return x


class ResidualPrompt(nn.Module):
    """
    Generate a residual prompt broadcast across tokens.

    Clip-level prompts are expanded to frames for video input.
    """
    def __init__(self, d_model: int, hidden: int = None, drop: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.hidden = hidden or (2 * d_model)
        self.fc_in = None  # Initialized using the prompt dimension on first use.
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc_out = nn.Linear(self.hidden, d_model)  # -> [B_total, D]

    def __init_fc_in(self, in_dim: int, device, dtype):
        if self.fc_in is None:
            self.fc_in = nn.Linear(in_dim, self.hidden).to(device=device, dtype=dtype)

    @staticmethod
    def _expand_gprompt_for_video(gp: torch.Tensor, B_total: int, is_video: bool, tframe: int) -> torch.Tensor:
        """
        Expand clip-level prompts to frames or align prompts to the target batch size.
        """
        B_gp = gp.shape[0]
        if is_video and tframe > 1:
            B_clip = max(B_total // tframe, 1)
            if B_gp == B_clip:
                return gp.repeat_interleave(tframe, dim=0)
            if B_gp == B_total:
                return gp
        if B_gp == 1:
            return gp.expand(B_total, -1)
        if B_gp != B_total:
            rep = (B_total + B_gp - 1) // B_gp
            gp = gp.repeat(rep, 1)[:B_total]
        return gp

    def forward(self, x: torch.Tensor, gprompt: torch.Tensor, is_video: bool = False, tframe: int = 1) -> torch.Tensor:
        """
        x:       [B_total, L, D]
        gprompt: [B_clip, P], [B_total, P], or [1, P]
        return:  residual [B_total, L, D]
        """
        if gprompt is None:
            return torch.zeros_like(x)

        B_total, L, D = x.shape
        assert D == self.d_model, f"D mismatch: got {D}, expected {self.d_model}"

        gp = self._expand_gprompt_for_video(gprompt, B_total, is_video, tframe)  # [B_total, P]
        self.__init_fc_in(gp.shape[-1], device=x.device, dtype=x.dtype)

        h = self.fc_in(gp)     # [B_total, hidden]
        h = self.act(h)
        h = self.drop(h)
        r = self.fc_out(h)     # [B_total, D]
        r = r.unsqueeze(1).expand(B_total, L, D)
        return r

