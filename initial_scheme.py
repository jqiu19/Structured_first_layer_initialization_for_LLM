
import torch
from safetensors.torch import save_file
import os
import torch.nn as nn
import torch.nn.functional as F

def initial_embedding(model, args):
    Dtype = model.model.embed_tokens.weight.dtype
    Device = model.model.embed_tokens.weight.device

    if args.initialization == "Xavier":
        model.model.embed_tokens.weight = torch.nn.init.xavier_uniform_(model.model.embed_tokens.weight)
        # 1) 第一层 Attention 的 q_proj / k_proj / v_proj 用 Xavier
        # attn0 = get_first_layer_attention(model)
        # [torch.nn.init.xavier_uniform_(proj.weight) for proj in (attn0.q_proj, attn0.k_proj, attn0.v_proj)]

        # 2) 第一层 FFN 的 gate_proj / up_proj 用 Xavier
        # mlp0 = get_first_layer_mlp(model)
        # torch.nn.init.xavier_uniform_(mlp0.gate_proj.weight)
        # torch.nn.init.xavier_uniform_(mlp0.up_proj.weight)

        # 3) 所有层的 FFN 的 gate_proj / up_proj 用 Xavier
        xavier_init_all_ffn_gate_up(model)


    elif args.initialization == "Kaiming":
        model.model.embed_tokens.weight = torch.nn.init.kaiming_uniform_(model.model.embed_tokens.weight)


        # attn0 = get_first_layer_attention(model)
        # [torch.nn.init.kaiming_uniform_(proj.weight) for proj in (attn0.q_proj, attn0.k_proj, attn0.v_proj)]
        kaiming_init_all_ffn_gate_up(model)

    elif args.initialization == "erank_kernel":
        construc_weight = kernel_fun(args, model.model.embed_tokens.weight)
        construc_weight = construc_weight.to(dtype=Dtype, device=Device)
        model.model.embed_tokens.weight.data = construc_weight

    elif args.initialization == "erank_basis":
        construc_weight = basis_fun(args, model.model.embed_tokens.weight)
        construc_weight = construc_weight.to(dtype=Dtype, device=Device)
        model.model.embed_tokens.weight.data = construc_weight

    elif args.initialization == "erank_implicit":
        construc_weight = implicit_fun(args, model.model.embed_tokens.weight)
        construc_weight = construc_weight.to(dtype=Dtype, device=Device)
        model.model.embed_tokens.weight.data = construc_weight

        attn0 = get_first_layer_attention(model)
        init_qkv_with_custom_gaussian_and_bias(attn0, model, args)
        add_qkv_activation(attn0, args)
    elif args.initialization == "Xavier_plus_SFLI":
        model.model.embed_tokens.weight = torch.nn.init.xavier_uniform_(model.model.embed_tokens.weight)

        attn0 = get_first_layer_attention(model)
        init_qkv_with_custom_gaussian_and_bias(attn0, model, args)
        add_qkv_activation(attn0, args)
    elif args.initialization == "Xavier_FFN_SFLI":
        model.model.embed_tokens.weight = torch.nn.init.xavier_uniform_(model.model.embed_tokens.weight)

        # mlp0 = get_first_layer_mlp(model)
        # init_ffn_with_custom_gaussian_and_bias_ug(mlp0, model, args)

        mlps = get_all_layer_mlp(model)
        for mlp in mlps:
            init_ffn_with_custom_gaussian_and_bias_ug(mlp, model, args)

    elif args.initialization == "Kaiming_plus_SFLI":
        model.model.embed_tokens.weight = torch.nn.init.kaiming_uniform_(model.model.embed_tokens.weight)

        attn0 = get_first_layer_attention(model)
        init_qkv_with_custom_gaussian_and_bias(attn0, model, args)
        add_qkv_activation(attn0, args)
    elif args.initialization == "Kaiming_FFN_SFLI":
        model.model.embed_tokens.weight = torch.nn.init.kaiming_uniform_(model.model.embed_tokens.weight)

        # mlp0 = get_first_layer_mlp(model)
        # init_ffn_with_custom_gaussian_and_bias_ug(mlp0, model, args)
        mlps = get_all_layer_mlp(model)
        for mlp in mlps:
            init_ffn_with_custom_gaussian_and_bias_ug(mlp, model, args)

    elif args.initialization == "Gaussian_default":
        pass
    elif args.initialization == "Gaussian_plus_SFLI":
        # attn0 = get_first_layer_attention(model)
        # init_qkv_with_custom_gaussian_and_bias(attn0, model, args)
        # add_qkv_activation(attn0, args)

        # attens = get_all_layer_attention(model)
        # for attn in attens:
        #     init_qkv_with_custom_gaussian_and_bias_q(attn, model, args)
        #     add_qkv_activation_q(attn, args)

        # attens = get_all_layer_attention(model)
        # for attn in attens:
        #     init_qkv_with_custom_gaussian_and_bias_k(attn, model, args)
        #     add_qkv_activation_k(attn, args)
        #
        attens = get_all_layer_attention(model)
        for attn in attens:
            init_qkv_with_custom_gaussian_and_bias_v(attn, model, args)
            add_qkv_activation_v(attn, args)
        #
        # attens = get_all_layer_attention(model)
        # for attn in attens:
        #     init_qkv_with_custom_gaussian_and_bias_qk(attn, model, args)
        #     add_qkv_activation_qk(attn, args)
        #
        # attens = get_all_layer_attention(model)
        # for attn in attens:
        #     init_qkv_with_custom_gaussian_and_bias_qv(attn, model, args)
        #     add_qkv_activation_qv(attn, args)

        # attens = get_all_layer_attention(model)
        # for attn in attens:
        #     init_qkv_with_custom_gaussian_and_bias_kv(attn, model, args)
        #     add_qkv_activation_kv(attn, args)

        # attens = get_all_layer_attention(model)
        # for attn in attens:
        #     init_qkv_with_custom_gaussian_and_bias_qkv(attn, model, args)
        #     add_qkv_activation_qkv(attn, args)

    elif args.initialization == "gaussian_FFN_SFLI":
        mlps = get_all_layer_mlp(model)
        for mlp in mlps:
            init_ffn_with_custom_gaussian_and_bias_ug(mlp, model, args)


    elif args.initialization == "gaussian_FFN_SFLI_g":
        mlps = get_all_layer_mlp(model)
        for mlp in mlps:
            init_ffn_with_custom_gaussian_and_bias_g(mlp, model, args)
    elif args.initialization == "gaussian_FFN_SFLI_u":
        mlps = get_all_layer_mlp(model)
        for mlp in mlps:
            init_ffn_with_custom_gaussian_and_bias_u(mlp, model, args)

    elif args.initialization == "gaussian_FFN_SFLI_d":
        mlps = get_all_layer_mlp(model)
        for mlp in mlps:
            init_ffn_with_custom_gaussian_and_bias_d(mlp, model, args)

    elif args.initialization == "gaussian_FFN_SFLI_gd":
        mlps = get_all_layer_mlp(model)
        for mlp in mlps:
            init_ffn_with_custom_gaussian_and_bias_gd(mlp, model, args)

    elif args.initialization == "gaussian_FFN_SFLI_ud":
        mlps = get_all_layer_mlp(model)
        for mlp in mlps:
            init_ffn_with_custom_gaussian_and_bias_ud(mlp, model, args)

    elif args.initialization == "gaussian_FFN_SFLI_ugd":
        mlps = get_all_layer_mlp(model)
        for mlp in mlps:
            init_ffn_with_custom_gaussian_and_bias_ugd(mlp, model, args)

    elif args.initialization == "gaussian_best_combination":
        attens = get_all_layer_attention(model)
        # attens = [get_first_layer_attention(model)]
        # attens = get_25_layer_attention(model)
        # attens = get_33_layer_attention(model)
        # attens = get_50_layer_attention(model)
        # attens = get_66_layer_attention(model)
        for attn in attens:
            init_qkv_with_custom_gaussian_and_bias_v(attn, model, args)
            add_qkv_activation_v(attn, args)
        mlps = get_all_layer_mlp(model)
        # mlps = [get_first_layer_mlp(model)]
        # mlps = get_25_layer_mlp(model)
        # mlps = get_33_layer_mlp(model)
        # mlps = get_50_layer_mlp(model)
        # mlps = get_66_layer_mlp(model)
        for mlp in mlps:
            init_ffn_with_custom_gaussian_and_bias_g(mlp, model, args)


    return model

def kernel_fun(args,weight, is_random=False):
    M, N = weight.shape
    x_int=1
    dx_int=1
    x = torch.linspace(-x_int, x_int, M).reshape(-1, 1)
    dx = torch.linspace(-dx_int, dx_int, N).reshape(1, -1)

    if args.init_name == 'exp':
        scale = 1e3
        lamb = 1e5
        fun=lambda x: scale * (torch.exp(-lamb * x ** 2))
    elif args.init_name  == 'sinc':
        scale = args.sinc_scale #1e2
        lamb = args.sinc_lamb #1e5
        fun= lambda x: scale * torch.sinc(lamb * x)
    elif args.init_name  == 'tanh':
        scale = 1e0
        lamb = 1e6
        fun= lambda x: scale * torch.tanh(lamb * x)
    else:
        raise ValueError(f"no '{args.init_name }' kernal function")

    new_weight = fun(x-dx)
    return new_weight

def basis_fun(args,weight, is_random=False):
    M, N = weight.shape
    if args.init_name  == 'fourier':
        scale = 1e-1
        x=torch.linspace(0, 2*torch.pi, M+1)[:M]
        k=torch.arange(N)
        new_weight = scale*torch.real(torch.exp(1j*k.reshape(1, -1)*x.reshape(-1, 1)))
    elif args.init_name  == 'chebyshev':
        scale = 1e-1
        x = torch.cos(torch.pi * torch.arange(M) / (M - 1))
        k = torch.arange(N)
        new_weight = scale*torch.cos(k.reshape(1, -1) * torch.arccos(x.reshape(-1, 1)))
    else:
        raise ValueError(f"no '{args.init_name }' basis function")
    return new_weight

def implicit_fun(args ,weight,exp_max=21290,exp_min=20,eig=None):
    M, N = weight.shape
    if args.init_name  == 'qr':
        scale = 2e-1
        if eig is None:
            alpha=2.01
            eig=torch.tan(torch.linspace(-torch.pi/alpha,torch.pi/alpha,N))
            eig_min,eig_max=torch.min(eig),torch.max(eig)
            eig=exp_min + (eig - eig_min) * (exp_max - exp_min) / (eig_max - eig_min)
        random_matrix = torch.randn(M,N)
        q, _ = torch.linalg.qr(random_matrix)
        q = q[:, :N]

        new_weight = scale*q@torch.diag(torch.sqrt(eig))
        print(f'mean:{new_weight.mean()}, variance:{new_weight.var(unbiased=False)}')

        var = new_weight.var(unbiased=False)
        factor = 0.01 / torch.sqrt(var)
        result = factor * new_weight
        print(f'mean:{result.mean()}, variance:{result.var(unbiased=False)}')
        new_weight = result
    else:
        raise ValueError(f"no '{args.init_name }' implicit method")
    return new_weight

def _replace_linear_with_bias(linear: nn.Linear) -> nn.Linear:
    new = nn.Linear(linear.in_features, linear.out_features, bias=True, device=linear.weight.device, dtype=linear.weight.dtype)
    with torch.no_grad():
        new.weight.copy_(linear.weight)
        new.bias.zero_()
    return new

def get_first_layer_attention(model):
    layer0 = model.model.layers[0]  # HF LLaMA 通常是这个
    if hasattr(layer0, "self_attn"):
        return layer0.self_attn

    # # 兜底：在 layer0 内找第一个带 q/k/v 的模块
    # for _, m in layer0.named_modules():
    #     if hasattr(m, "q_proj") and hasattr(m, "k_proj") and hasattr(m, "v_proj"):
    #         return m

    raise RuntimeError("Cannot find attention module with q_proj/k_proj/v_proj in model.model.layers[0]")

def get_all_layer_attention(model):
    attns = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn"):
            attns.append(layer.self_attn)
        else:
            raise RuntimeError(f"Cannot find self_attn in model.model.layers[{i}]")
    return attns

def get_25_layer_attention(model):
    attns = []
    for i, layer in enumerate(model.model.layers[:3]):
        if hasattr(layer, "self_attn"):
            attns.append(layer.self_attn)
        else:
            raise RuntimeError(f"Cannot find self_attn in model.model.layers[{i}]")
    return attns

def get_33_layer_attention(model):
    attns = []
    for i, layer in enumerate(model.model.layers[:4]):  # only layers 0,1,2,3
        if hasattr(layer, "self_attn"):
            attns.append(layer.self_attn)
        else:
            raise RuntimeError(f"Cannot find self_attn in model.model.layers[{i}]")
    return attns

def get_50_layer_attention(model):
    attns = []
    for i, layer in enumerate(model.model.layers[:6]):
        if hasattr(layer, "self_attn"):
            attns.append(layer.self_attn)
        else:
            raise RuntimeError(f"Cannot find self_attn in model.model.layers[{i}]")
    return attns

def get_66_layer_attention(model):
    attns = []
    for i, layer in enumerate(model.model.layers[:8]):
        if hasattr(layer, "self_attn"):
            attns.append(layer.self_attn)
        else:
            raise RuntimeError(f"Cannot find self_attn in model.model.layers[{i}]")
    return attns



def _init_single_qkv_proj(attn_module, proj_name, model, args):
    """
    对 attn_module.{q_proj,k_proj,v_proj} 中的某一个做 SFLI 初始化：
    1) 确保 bias=True
    2) weight ~ N(0, std^2) * hd_coeffs
    3) bias = arithmetic progression + noise, then * hd_coeffs
    """
    if proj_name in ["q_proj"]:
        base_std = getattr(args, "q_qkv_std", None)
        bias_start = getattr(args, "q_bias_start", 0.0)
        bias_step = getattr(args, "q_bias_step", 0.0)
        bias_noise_std = getattr(args, "q_bias_noise_std", 0.0)
        hd_coeffs = getattr(args, "q_hd_coeffs", 1.0)
    elif proj_name in ["k_proj"]:
        base_std = getattr(args, "k_qkv_std", None)
        bias_start = getattr(args, "k_bias_start", 0.0)
        bias_step = getattr(args, "k_bias_step", 0.0)
        bias_noise_std = getattr(args, "k_bias_noise_std", 0.0)
        hd_coeffs = getattr(args, "k_hd_coeffs", 1.0)
    elif proj_name == "v_proj":
        base_std = getattr(args, "v_qkv_std", None)
        bias_start = getattr(args, "v_bias_start", 0.0)
        bias_step = getattr(args, "v_bias_step", 0.0)
        bias_noise_std = getattr(args, "v_bias_noise_std", 0.0)
        hd_coeffs = getattr(args, "v_hd_coeffs", 1.0)
    else:
        raise ValueError(f"Unknown proj_name: {proj_name}")

    lin = getattr(attn_module, proj_name)

    if lin.bias is None:
        lin = _replace_linear_with_bias(lin)
        setattr(attn_module, proj_name, lin)

    if base_std is not None:
        std = float(base_std)
    else:
        fan_in = lin.in_features
        init_range = getattr(model.config, "initializer_range", 0.02)
        std = float(init_range / (fan_in ** 0.5))

    with torch.no_grad():
        lin.weight.normal_(mean=0.0, std=std)
        #lin.weight.div_(lin.weight.norm(dim=1, keepdim=True).clamp_min(1e-12))
        lin.weight.mul_(hd_coeffs)

        out_dim = lin.out_features
        arith = bias_start + bias_step * torch.arange(
            out_dim, device=lin.weight.device, dtype=lin.weight.dtype
        )
        noise = torch.randn(
            out_dim, device=lin.weight.device, dtype=lin.weight.dtype
        ) * bias_noise_std
        lin.bias.copy_(arith + noise)
        lin.bias.mul_(hd_coeffs)

        print(
            f"[QKV SFLI] {proj_name}: "
            f"W mean={lin.weight.mean().item():.3e}, "
            f"var={lin.weight.var(unbiased=False).item():.3e}, "
            f"b mean={lin.bias.mean().item():.3e}, "
            f"var={lin.bias.var(unbiased=False).item():.3e}")



def _init_qkv_subset(attn_module, model, args, proj_names):
    for proj_name in proj_names:
        _init_single_qkv_proj(attn_module, proj_name, model, args)


def init_qkv_with_custom_gaussian_and_bias_q(attn_module, model, args):
    _init_qkv_subset(attn_module, model, args, ["q_proj"])


def init_qkv_with_custom_gaussian_and_bias_k(attn_module, model, args):
    _init_qkv_subset(attn_module, model, args, ["k_proj"])


def init_qkv_with_custom_gaussian_and_bias_v(attn_module, model, args):
    _init_qkv_subset(attn_module, model, args, ["v_proj"])


def init_qkv_with_custom_gaussian_and_bias_qk(attn_module, model, args):
    _init_qkv_subset(attn_module, model, args, ["q_proj", "k_proj"])


def init_qkv_with_custom_gaussian_and_bias_qv(attn_module, model, args):
    _init_qkv_subset(attn_module, model, args, ["q_proj", "v_proj"])


def init_qkv_with_custom_gaussian_and_bias_kv(attn_module, model, args):
    _init_qkv_subset(attn_module, model, args, ["k_proj", "v_proj"])


def init_qkv_with_custom_gaussian_and_bias_qkv(attn_module, model, args):
    _init_qkv_subset(attn_module, model, args, ["q_proj", "k_proj", "v_proj"])


class SincAct(nn.Module):
    def forward(self, x):
        return torch.sinc(x)

class CosAct(nn.Module):
    def forward(self, x):
        return torch.cos(x)

class SinAct(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class HatAct(nn.Module):
    """
    hat(x) = max(0, 1 - |x|)
    也叫 triangular / tent function
    """
    def forward(self, x):
        return torch.clamp(1.0 - x.abs(), min=0.0)

class SiLUAct(nn.Module):
    def forward(self, x):
        return torch.nn.functional.silu(x)

class ChebyshevAct(nn.Module):
    """
    f(x) = sum_{k=0}^N a_k T_k(z),  z = clip(x / L, -1, 1)
    """
    def __init__(
        self,
        N: int = 8,
        L: float = 1.0,
        clamp: bool = True,
        init: str = "identity",
        sigma: float = 0.02,   # <-- 新增：gaussian_init 用的 std
    ):
        super().__init__()
        assert N >= 0
        self.N = int(N)
        self.L = float(L)
        self.clamp = bool(clamp)
        self.sigma = float(sigma)

        # Learnable coefficients a_0..a_N
        self.a = nn.Parameter(torch.zeros(self.N + 1))

        if init == "identity":
            with torch.no_grad():
                self.a.zero_()
                if self.N >= 1:
                    self.a[1].fill_(1.0)
        elif init == "zero":
            with torch.no_grad():
                self.a.zero_()
        elif init == "gaussian_init":
            with torch.no_grad():
                # a_k ~ N(0, sigma^2)
                self.a.normal_(mean=0.0, std=self.sigma)
        else:
            raise ValueError(f"Unknown init: {init}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x / self.L
        if self.clamp:
            z = torch.clamp(z, -1.0, 1.0)

        a = self.a.to(dtype=x.dtype, device=x.device)

        y = a[0] * torch.ones_like(z)
        if self.N == 0:
            return y

        Tkm2 = torch.ones_like(z)  # T0
        Tkm1 = z                   # T1
        y = y + a[1] * Tkm1

        for k in range(2, self.N + 1):
            Tk = 2.0 * z * Tkm1 - Tkm2
            y = y + a[k] * Tk
            Tkm2, Tkm1 = Tkm1, Tk

        return y



class SincPolyAct(nn.Module):
    """
    phi_multi(x) = sum_{i=1..M} sum_{j=-N..N} c_{i,j} * S(j, h_i)(x)
    where S(j,h)(x) = sin((pi/h)(x-jh)) / ((pi/h)(x-jh)) = sinc(x/h - j)

    - M, N, h0: manually chosen
    - c_{i,j}: learnable parameters, shape (M, 2N+1)
    - h_i: inverse decay schedule
    """
    def __init__(
        self,
        M: int = 4,
        N: int = 8,
        h0: float = 1.0,
        inverse_decay: str = "h0_over_i",  # "h0_over_i" or "one_over_i_h0"
        init: str = "identity_like",       # "identity_like" or "zero"
        dtype=None,
        device=None,
    ):
        super().__init__()
        assert M >= 1 and N >= 0
        self.M = int(M)
        self.N = int(N)
        self.h0 = float(h0)
        self.inverse_decay = inverse_decay

        # learnable coefficients c_{i,j}
        self.c = nn.Parameter(torch.zeros(self.M, 2 * self.N + 1, dtype=dtype, device=device))

        # pre-store integer j grid as a buffer (moves with .to(device))
        j = torch.arange(-self.N, self.N + 1, dtype=dtype if dtype is not None else torch.float32, device=device)
        self.register_buffer("j_grid", j, persistent=False)

        self.reset_parameters(init)

    def _make_h(self, x: torch.Tensor) -> torch.Tensor:
        # h_i schedule (inverse decay)
        i = torch.arange(1, self.M + 1, device=x.device, dtype=x.dtype)
        if self.inverse_decay == "h0_over_i":
            # h_i = h0 / i   (most common "inverse decay")
            h = (self.h0 / i)
        elif self.inverse_decay == "one_over_i_h0":
            # interpret "h_i = 1 / (i * h0)"
            h = 1.0 / (i * self.h0)
        else:
            raise ValueError(f"Unknown inverse_decay: {self.inverse_decay}")
        return h  # shape (M,)

    @torch.no_grad()
    def reset_parameters(self, init: str = "identity_like"):
        if init == "zero":
            self.c.zero_()
            return

        if init == "identity_like":
            # 目标：一开始尽量不伤训练（让激活像“近似恒等/近似线性”）
            # sinc 在 0 附近 ~ 1 - O(z^2)，所以用中心 j=0 的基函数做一个“平滑门”
            # 这里给 i=1 的 (j=0) 一个 1，其余 0
            self.c.zero_()
            center = self.N  # j=0 对应的位置
            self.c[0, center] = 1.0
            return

        raise ValueError(f"Unknown init: {init}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: any shape [...]
        return: same shape [...]
        """
        # compute basis: S(j, h_i)(x) = sinc(x/h_i - j)
        # shapes:
        #   x[..., None, None] -> [..., 1, 1]
        #   h[None, :, None]   -> [1, M, 1]
        #   j[None, None, :]   -> [1, 1, 2N+1]
        h = self._make_h(x)  # (M,)
        j = self.j_grid.to(device=x.device, dtype=x.dtype)  # (2N+1,)

        z = x[..., None, None] / h[None, :, None] - j[None, None, :]  # [..., M, 2N+1]
        basis = torch.sinc(z)  # [..., M, 2N+1]

        # weighted sum with c_{i,j}
        # c: (M, 2N+1) -> broadcast to [..., M, 2N+1]
        y = (basis * self.c.to(dtype=x.dtype, device=x.device)).sum(dim=(-2, -1))  # [...]
        return y

def _elu1(x: torch.Tensor) -> torch.Tensor:
    # ELU(x, alpha=1): range (-1, +inf)
    return F.elu(x, alpha=1.0)

def jacobi_P(n: int, a: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute Jacobi polynomial P_n^{(a,b)}(x) for x in [-1,1].
    a,b can be broadcastable to x (e.g., scalar or [C] with x shape [..., C]).
    Recurrence is standard (see e.g. Wikipedia / Szegő).
    """
    if n == 0:
        return torch.ones_like(x)
    if n == 1:
        return 0.5 * ((2.0 + a + b) * x + (a - b))

    Pnm2 = torch.ones_like(x)  # P_0
    Pnm1 = 0.5 * ((2.0 + a + b) * x + (a - b))  # P_1

    for k in range(2, n + 1):
        kf = float(k)
        two_k_ab = 2.0 * kf + a + b
        # coefficients
        A = (two_k_ab - 1.0) * ((two_k_ab) * (two_k_ab - 2.0) * x + (a*a - b*b))
        B = 2.0 * (kf + a - 1.0) * (kf + b - 1.0) * (two_k_ab)
        C = 2.0 * kf * (kf + a + b) * (two_k_ab - 2.0)

        Pn = (A * Pnm1 - B * Pnm2) / C
        Pnm2, Pnm1 = Pnm1, Pn

    return Pnm1

class FractionalJacobiAct(nn.Module):
    """
    A pointwise activation based on *fractional-order* Jacobi functions, in the spirit of fKAN:
        z = sigmoid(x) in (0,1)
        t = 2*z^gamma - 1   (maps to [-1,1])
        y = P_n^{(alpha,beta)}(t)

    Parameters alpha,beta are constrained to (-1, +inf) via ELU(.,1),
    gamma constrained to (0,1) via sigmoid.
    """
    def __init__(
        self,
        degree: int = 8,
        per_channel: bool = False,
        channels: int = 768,
        alpha_init: float = 0.0,
        beta_init: float = 0.0,
        gamma_init: float = 0.5,
    ):
        super().__init__()
        self.degree = int(degree)
        self.per_channel = bool(per_channel)

        shape = (channels,) if per_channel else (1,)
        self.alpha_raw = nn.Parameter(torch.full(shape, float(alpha_init)))
        self.beta_raw  = nn.Parameter(torch.full(shape, float(beta_init)))
        self.gamma_raw = nn.Parameter(torch.full(shape, float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [..., C] typically
        # constrain parameters
        alpha = _elu1(self.alpha_raw)  # (-1, inf)
        beta  = _elu1(self.beta_raw)   # (-1, inf)
        gamma = torch.sigmoid(self.gamma_raw)  # (0,1)

        # broadcast to x
        while alpha.dim() < x.dim():
            alpha = alpha.unsqueeze(0)
            beta  = beta.unsqueeze(0)
            gamma = gamma.unsqueeze(0)

        z = torch.sigmoid(x)                 # (0,1)
        t = 2.0 * torch.pow(z, gamma) - 1.0  # [-1,1]
        return jacobi_P(self.degree, alpha, beta, t)

def _make_act(name: str) -> nn.Module:
    if name == "none":
        return nn.Identity()
    if name == "tanh":
        return nn.Tanh()
    if name == "sinc":
        return SincAct()
    if name == "cos":
        return CosAct()
    if name == "sin":
        return SinAct()
    if name == "hat":
        return HatAct()
    if name == "silu":
        return SiLUAct()
    if name == "cheb":
        # 你可以在 argparse 里加 --cheb_N --cheb_L
        N = 16
        L = 768
        clamp = True
        init = "gaussian_init"
        sigma = 0.1
        return ChebyshevAct(N=N, L=L, clamp=clamp, init=init, sigma=sigma)
    if name == "sinc_poly":
        M = 3
        N = 16
        sinc_h0 = 2
        return SincPolyAct(M=M, N=N, h0=sinc_h0)
    if name == "frac_jacobi":
        # 建议：先用对称的 Legendre-like 初始化 (alpha=beta=0)，gamma=0.5
        # 如果你想“每个通道一套参数”，per_channel=True（更灵活但更不稳定/更贵）
        return FractionalJacobiAct(
            degree= 8,
            per_channel= False,
            channels= 768,  # 你也可以直接写死 768
            alpha_init= 0.0,
            beta_init= 0.0,
            gamma_init= 1, # 取[-3,3]，太大太小会导致sigmoid饱和影响梯度
        )

    raise ValueError(f"Unknown qkv_act: {name}")

def _add_activation_to_single_qkv_proj(attn_module, proj_name, args):
    act_name = getattr(args, "qkv_act", "none")
    if act_name == "none":
        return

    proj = getattr(attn_module, proj_name)
    if isinstance(proj, nn.Sequential):
        return

    setattr(
        attn_module,
        proj_name,
        nn.Sequential(
            proj,
            _make_act(act_name),
        ),
    )


def _add_activation_to_qkv_subset(attn_module, args, proj_names):
    for proj_name in proj_names:
        _add_activation_to_single_qkv_proj(attn_module, proj_name, args)


def add_qkv_activation_q(attn_module, args):
    _add_activation_to_qkv_subset(attn_module, args, ["q_proj"])


def add_qkv_activation_k(attn_module, args):
    _add_activation_to_qkv_subset(attn_module, args, ["k_proj"])


def add_qkv_activation_v(attn_module, args):
    _add_activation_to_qkv_subset(attn_module, args, ["v_proj"])


def add_qkv_activation_qk(attn_module, args):
    _add_activation_to_qkv_subset(attn_module, args, ["q_proj", "k_proj"])


def add_qkv_activation_qv(attn_module, args):
    _add_activation_to_qkv_subset(attn_module, args, ["q_proj", "v_proj"])


def add_qkv_activation_kv(attn_module, args):
    _add_activation_to_qkv_subset(attn_module, args, ["k_proj", "v_proj"])


def add_qkv_activation_qkv(attn_module, args):
    _add_activation_to_qkv_subset(attn_module, args, ["q_proj", "k_proj", "v_proj"])


def get_first_layer_mlp(model):
    layer0 = model.model.layers[0]
    if hasattr(layer0, "mlp"):
        return layer0.mlp
    raise RuntimeError("Cannot find mlp in model.model.layers[0]")

def get_all_layer_mlp(model):
    mlps = [];
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "mlp"):
            mlps.append(layer.mlp)
        else:
            raise RuntimeError("Cannot find mlp in model.model.layers")
    return mlps

def get_25_layer_mlp(model):
    mlps = []
    for i, layer in enumerate(model.model.layers[:3]):
        if hasattr(layer, "mlp"):
            mlps.append(layer.mlp)
        else:
            raise RuntimeError(f"Cannot find mlp in model.model.layers[{i}]")
    return mlps

def get_33_layer_mlp(model):
    mlps = []
    for i, layer in enumerate(model.model.layers[:4]):  # only layers 0,1,2,3
        if hasattr(layer, "mlp"):
            mlps.append(layer.mlp)
        else:
            raise RuntimeError(f"Cannot find mlp in model.model.layers[{i}]")
    return mlps

def get_50_layer_mlp(model):
    mlps = []
    for i, layer in enumerate(model.model.layers[:6]):
        if hasattr(layer, "mlp"):
            mlps.append(layer.mlp)
        else:
            raise RuntimeError(f"Cannot find mlp in model.model.layers[{i}]")
    return mlps

def get_66_layer_mlp(model):
    mlps = []
    for i, layer in enumerate(model.model.layers[:8]):
        if hasattr(layer, "mlp"):
            mlps.append(layer.mlp)
        else:
            raise RuntimeError(f"Cannot find mlp in model.model.layers[{i}]")
    return mlps

def init_ffn_with_custom_gaussian_and_bias_ug(mlp0, model, args):
    """
    对 layer0.mlp 的 gate_proj 和 up_proj 做 SFLI 初始化：
      - W ~ N(0, std^2) * ffn_hd_coeffs
      - b = 等差(arith) + 噪声(noise)，再 * ffn_hd_coeffs
    """

    # ---- args（优先 ffn_*，否则 fallback 到 v_* / v_hd_coeffs）----
    ffn_base_std = getattr(args, "ffn_base_std", None)
    ffn_bias_start = getattr(args, "ffn_bias_start", getattr(args, "v_bias_start", 0.0))
    ffn_bias_step = getattr(args, "ffn_bias_step", getattr(args, "v_bias_step", 0.0))
    ffn_bias_noise_std = getattr(args, "ffn_bias_noise_std", getattr(args, "v_bias_noise_std", 0.0))
    ffn_hd_coeffs = getattr(args, "ffn_hd_coeffs", getattr(args, "v_hd_coeffs", 1.0))

    for name in ["gate_proj", "up_proj"]:
        if not hasattr(mlp0, name):
            raise RuntimeError(f"LlamaMLP has no {name}; cannot run gaussian_FFN_SFLI")

        lin = getattr(mlp0, name)  # Linear(768 -> 2048, bias=False) by default

        # ---- 确保 bias=True ----
        if lin.bias is None:
            lin = _replace_linear_with_bias(lin)
            setattr(mlp0, name, lin)

        # ---- std：若没给 ffn_base_std，走 initializer_range/sqrt(fan_in) ----
        if ffn_base_std is not None:
            std = float(ffn_base_std)
        else:
            fan_in = lin.in_features
            init_range = getattr(model.config, "initializer_range", 0.02)
            std = float(init_range / (fan_in ** 0.5))

        with torch.no_grad():
            lin.weight.normal_(mean=0.0, std=std)
            #lin.weight.div_(lin.weight.norm(dim=1, keepdim=True).clamp_min(1e-12))
            lin.weight.mul_(ffn_hd_coeffs)

            out_dim = lin.out_features  # 2048
            arith = ffn_bias_start + ffn_bias_step * torch.arange(
                out_dim, device=lin.weight.device, dtype=lin.weight.dtype
            )
            noise = torch.randn(out_dim, device=lin.weight.device, dtype=lin.weight.dtype) * ffn_bias_noise_std
            lin.bias.copy_(arith + noise)
            lin.bias.mul_(ffn_hd_coeffs)

        try:
            print(
                f"[FFN SFLI] {name}: W mean={lin.weight.mean().item():.3e}, var={lin.weight.var(unbiased=False).item():.3e}, "
                f"b mean={lin.bias.mean().item():.3e}, var={lin.bias.var(unbiased=False).item():.3e}"
            )
        except Exception:
            pass

def init_ffn_with_custom_gaussian_and_bias_gd(mlp0, model, args):
    """
    对 layer0.mlp 的 gate_proj 和 up_proj 做 SFLI 初始化：
      - W ~ N(0, std^2) * ffn_hd_coeffs
      - b = 等差(arith) + 噪声(noise)，再 * ffn_hd_coeffs
    """

    # ---- args（优先 ffn_*，否则 fallback 到 v_* / v_hd_coeffs）----
    ffn_base_std = getattr(args, "ffn_base_std", None)
    ffn_bias_start = getattr(args, "ffn_bias_start", getattr(args, "v_bias_start", 0.0))
    ffn_bias_step = getattr(args, "ffn_bias_step", getattr(args, "v_bias_step", 0.0))
    ffn_bias_noise_std = getattr(args, "ffn_bias_noise_std", getattr(args, "v_bias_noise_std", 0.0))
    ffn_hd_coeffs = getattr(args, "ffn_hd_coeffs", getattr(args, "v_hd_coeffs", 1.0))

    for name in ["gate_proj", "down_proj"]:
        if not hasattr(mlp0, name):
            raise RuntimeError(f"LlamaMLP has no {name}; cannot run gaussian_FFN_SFLI")

        lin = getattr(mlp0, name)  # Linear(768 -> 2048, bias=False) by default

        # ---- 确保 bias=True ----
        if lin.bias is None:
            lin = _replace_linear_with_bias(lin)
            setattr(mlp0, name, lin)

        # ---- std：若没给 ffn_base_std，走 initializer_range/sqrt(fan_in) ----
        if ffn_base_std is not None:
            std = float(ffn_base_std)
        else:
            fan_in = lin.in_features
            init_range = getattr(model.config, "initializer_range", 0.02)
            std = float(init_range / (fan_in ** 0.5))

        with torch.no_grad():
            lin.weight.normal_(mean=0.0, std=std)
            #lin.weight.div_(lin.weight.norm(dim=1, keepdim=True).clamp_min(1e-12))
            lin.weight.mul_(ffn_hd_coeffs)

            out_dim = lin.out_features  # 2048
            arith = ffn_bias_start + ffn_bias_step * torch.arange(
                out_dim, device=lin.weight.device, dtype=lin.weight.dtype
            )
            noise = torch.randn(out_dim, device=lin.weight.device, dtype=lin.weight.dtype) * ffn_bias_noise_std
            lin.bias.copy_(arith + noise)
            lin.bias.mul_(ffn_hd_coeffs)

        try:
            print(
                f"[FFN SFLI] {name}: W mean={lin.weight.mean().item():.3e}, var={lin.weight.var(unbiased=False).item():.3e}, "
                f"b mean={lin.bias.mean().item():.3e}, var={lin.bias.var(unbiased=False).item():.3e}"
            )
        except Exception:
            pass

def init_ffn_with_custom_gaussian_and_bias_ud(mlp0, model, args):
    """
    对 layer0.mlp 的 gate_proj 和 up_proj 做 SFLI 初始化：
      - W ~ N(0, std^2) * ffn_hd_coeffs
      - b = 等差(arith) + 噪声(noise)，再 * ffn_hd_coeffs
    """

    # ---- args（优先 ffn_*，否则 fallback 到 v_* / v_hd_coeffs）----
    ffn_base_std = getattr(args, "ffn_base_std", None)
    ffn_bias_start = getattr(args, "ffn_bias_start", getattr(args, "v_bias_start", 0.0))
    ffn_bias_step = getattr(args, "ffn_bias_step", getattr(args, "v_bias_step", 0.0))
    ffn_bias_noise_std = getattr(args, "ffn_bias_noise_std", getattr(args, "v_bias_noise_std", 0.0))
    ffn_hd_coeffs = getattr(args, "ffn_hd_coeffs", getattr(args, "v_hd_coeffs", 1.0))

    for name in ["down_proj", "up_proj"]:
        if not hasattr(mlp0, name):
            raise RuntimeError(f"LlamaMLP has no {name}; cannot run gaussian_FFN_SFLI")

        lin = getattr(mlp0, name)  # Linear(768 -> 2048, bias=False) by default

        # ---- 确保 bias=True ----
        if lin.bias is None:
            lin = _replace_linear_with_bias(lin)
            setattr(mlp0, name, lin)

        # ---- std：若没给 ffn_base_std，走 initializer_range/sqrt(fan_in) ----
        if ffn_base_std is not None:
            std = float(ffn_base_std)
        else:
            fan_in = lin.in_features
            init_range = getattr(model.config, "initializer_range", 0.02)
            std = float(init_range / (fan_in ** 0.5))

        with torch.no_grad():
            lin.weight.normal_(mean=0.0, std=std)
            #lin.weight.div_(lin.weight.norm(dim=1, keepdim=True).clamp_min(1e-12))
            lin.weight.mul_(ffn_hd_coeffs)

            out_dim = lin.out_features  # 2048
            arith = ffn_bias_start + ffn_bias_step * torch.arange(
                out_dim, device=lin.weight.device, dtype=lin.weight.dtype
            )
            noise = torch.randn(out_dim, device=lin.weight.device, dtype=lin.weight.dtype) * ffn_bias_noise_std
            lin.bias.copy_(arith + noise)
            lin.bias.mul_(ffn_hd_coeffs)

        try:
            print(
                f"[FFN SFLI] {name}: W mean={lin.weight.mean().item():.3e}, var={lin.weight.var(unbiased=False).item():.3e}, "
                f"b mean={lin.bias.mean().item():.3e}, var={lin.bias.var(unbiased=False).item():.3e}"
            )
        except Exception:
            pass

def init_ffn_with_custom_gaussian_and_bias_ugd(mlp0, model, args):
    """
    对 layer0.mlp 的 gate_proj 和 up_proj 做 SFLI 初始化：
      - W ~ N(0, std^2) * ffn_hd_coeffs
      - b = 等差(arith) + 噪声(noise)，再 * ffn_hd_coeffs
    """

    # ---- args（优先 ffn_*，否则 fallback 到 v_* / v_hd_coeffs）----
    ffn_base_std = getattr(args, "ffn_base_std", None)
    ffn_bias_start = getattr(args, "ffn_bias_start", getattr(args, "v_bias_start", 0.0))
    ffn_bias_step = getattr(args, "ffn_bias_step", getattr(args, "v_bias_step", 0.0))
    ffn_bias_noise_std = getattr(args, "ffn_bias_noise_std", getattr(args, "v_bias_noise_std", 0.0))
    ffn_hd_coeffs = getattr(args, "ffn_hd_coeffs", getattr(args, "v_hd_coeffs", 1.0))

    for name in ["gate_proj", "up_proj", "down_proj"]:
        if not hasattr(mlp0, name):
            raise RuntimeError(f"LlamaMLP has no {name}; cannot run gaussian_FFN_SFLI")

        lin = getattr(mlp0, name)  # Linear(768 -> 2048, bias=False) by default

        # ---- 确保 bias=True ----
        if lin.bias is None:
            lin = _replace_linear_with_bias(lin)
            setattr(mlp0, name, lin)

        # ---- std：若没给 ffn_base_std，走 initializer_range/sqrt(fan_in) ----
        if ffn_base_std is not None:
            std = float(ffn_base_std)
        else:
            fan_in = lin.in_features
            init_range = getattr(model.config, "initializer_range", 0.02)
            std = float(init_range / (fan_in ** 0.5))

        with torch.no_grad():
            lin.weight.normal_(mean=0.0, std=std)
            #lin.weight.div_(lin.weight.norm(dim=1, keepdim=True).clamp_min(1e-12))
            lin.weight.mul_(ffn_hd_coeffs)

            out_dim = lin.out_features  # 2048
            arith = ffn_bias_start + ffn_bias_step * torch.arange(
                out_dim, device=lin.weight.device, dtype=lin.weight.dtype
            )
            noise = torch.randn(out_dim, device=lin.weight.device, dtype=lin.weight.dtype) * ffn_bias_noise_std
            lin.bias.copy_(arith + noise)
            lin.bias.mul_(ffn_hd_coeffs)

        try:
            print(
                f"[FFN SFLI] {name}: W mean={lin.weight.mean().item():.3e}, var={lin.weight.var(unbiased=False).item():.3e}, "
                f"b mean={lin.bias.mean().item():.3e}, var={lin.bias.var(unbiased=False).item():.3e}"
            )
        except Exception:
            pass

def init_ffn_with_custom_gaussian_and_bias_u(mlp0, model, args):
    """
    对 layer0.mlp 的 gate_proj 和 up_proj 做 SFLI 初始化：
      - W ~ N(0, std^2) * ffn_hd_coeffs
      - b = 等差(arith) + 噪声(noise)，再 * ffn_hd_coeffs
    """

    # ---- args（优先 ffn_*，否则 fallback 到 v_* / v_hd_coeffs）----
    ffn_base_std = getattr(args, "ffn_base_std", None)
    ffn_bias_start = getattr(args, "ffn_bias_start", getattr(args, "v_bias_start", 0.0))
    ffn_bias_step = getattr(args, "ffn_bias_step", getattr(args, "v_bias_step", 0.0))
    ffn_bias_noise_std = getattr(args, "ffn_bias_noise_std", getattr(args, "v_bias_noise_std", 0.0))
    ffn_hd_coeffs = getattr(args, "ffn_hd_coeffs", getattr(args, "v_hd_coeffs", 1.0))

    for name in ["up_proj"]:
        if not hasattr(mlp0, name):
            raise RuntimeError(f"LlamaMLP has no {name}; cannot run gaussian_FFN_SFLI")

        lin = getattr(mlp0, name)  # Linear(768 -> 2048, bias=False) by default

        # ---- 确保 bias=True ----
        if lin.bias is None:
            lin = _replace_linear_with_bias(lin)
            setattr(mlp0, name, lin)

        # ---- std：若没给 ffn_base_std，走 initializer_range/sqrt(fan_in) ----
        if ffn_base_std is not None:
            std = float(ffn_base_std)
        else:
            fan_in = lin.in_features
            init_range = getattr(model.config, "initializer_range", 0.02)
            std = float(init_range / (fan_in ** 0.5))

        with torch.no_grad():
            lin.weight.normal_(mean=0.0, std=std)
            #lin.weight.div_(lin.weight.norm(dim=1, keepdim=True).clamp_min(1e-12))
            lin.weight.mul_(ffn_hd_coeffs)

            out_dim = lin.out_features  # 2048
            arith = ffn_bias_start + ffn_bias_step * torch.arange(
                out_dim, device=lin.weight.device, dtype=lin.weight.dtype
            )
            noise = torch.randn(out_dim, device=lin.weight.device, dtype=lin.weight.dtype) * ffn_bias_noise_std
            lin.bias.copy_(arith + noise)
            lin.bias.mul_(ffn_hd_coeffs)

        try:
            print(
                f"[FFN SFLI] {name}: W mean={lin.weight.mean().item():.3e}, var={lin.weight.var(unbiased=False).item():.3e}, "
                f"b mean={lin.bias.mean().item():.3e}, var={lin.bias.var(unbiased=False).item():.3e}"
            )
        except Exception:
            pass

def init_ffn_with_custom_gaussian_and_bias_g(mlp0, model, args):
    """
    对 layer0.mlp 的 gate_proj 和 up_proj 做 SFLI 初始化：
      - W ~ N(0, std^2) * ffn_hd_coeffs
      - b = 等差(arith) + 噪声(noise)，再 * ffn_hd_coeffs
    """

    # ---- args（优先 ffn_*，否则 fallback 到 v_* / v_hd_coeffs）----
    ffn_base_std = getattr(args, "ffn_base_std", None)
    ffn_bias_start = getattr(args, "ffn_bias_start", getattr(args, "v_bias_start", 0.0))
    ffn_bias_step = getattr(args, "ffn_bias_step", getattr(args, "v_bias_step", 0.0))
    ffn_bias_noise_std = getattr(args, "ffn_bias_noise_std", getattr(args, "v_bias_noise_std", 0.0))
    ffn_hd_coeffs = getattr(args, "ffn_hd_coeffs", getattr(args, "v_hd_coeffs", 1.0))

    for name in ["gate_proj"]:
        if not hasattr(mlp0, name):
            raise RuntimeError(f"LlamaMLP has no {name}; cannot run gaussian_FFN_SFLI")

        lin = getattr(mlp0, name)  # Linear(768 -> 2048, bias=False) by default

        # ---- 确保 bias=True ----
        if lin.bias is None:
            lin = _replace_linear_with_bias(lin)
            setattr(mlp0, name, lin)

        # ---- std：若没给 ffn_base_std，走 initializer_range/sqrt(fan_in) ----
        if ffn_base_std is not None:
            std = float(ffn_base_std)
        else:
            fan_in = lin.in_features
            init_range = getattr(model.config, "initializer_range", 0.02)
            std = float(init_range / (fan_in ** 0.5))

        with torch.no_grad():
            lin.weight.normal_(mean=0.0, std=std)
            #lin.weight.div_(lin.weight.norm(dim=1, keepdim=True).clamp_min(1e-12))
            lin.weight.mul_(ffn_hd_coeffs)

            out_dim = lin.out_features  # 2048
            arith = ffn_bias_start + ffn_bias_step * torch.arange(
                out_dim, device=lin.weight.device, dtype=lin.weight.dtype
            )
            noise = torch.randn(out_dim, device=lin.weight.device, dtype=lin.weight.dtype) * ffn_bias_noise_std
            lin.bias.copy_(arith + noise)
            lin.bias.mul_(ffn_hd_coeffs)

        try:
            print(
                f"[FFN SFLI] {name}: W mean={lin.weight.mean().item():.3e}, var={lin.weight.var(unbiased=False).item():.3e}, "
                f"b mean={lin.bias.mean().item():.3e}, var={lin.bias.var(unbiased=False).item():.3e}"
            )
        except Exception:
            pass

def init_ffn_with_custom_gaussian_and_bias_d(mlp0, model, args):
    """
    对 layer0.mlp 的 gate_proj 和 up_proj 做 SFLI 初始化：
      - W ~ N(0, std^2) * ffn_hd_coeffs
      - b = 等差(arith) + 噪声(noise)，再 * ffn_hd_coeffs
    """

    # ---- args（优先 ffn_*，否则 fallback 到 v_* / v_hd_coeffs）----
    ffn_base_std = getattr(args, "ffn_base_std", None)
    ffn_bias_start = getattr(args, "ffn_bias_start", getattr(args, "v_bias_start", 0.0))
    ffn_bias_step = getattr(args, "ffn_bias_step", getattr(args, "v_bias_step", 0.0))
    ffn_bias_noise_std = getattr(args, "ffn_bias_noise_std", getattr(args, "v_bias_noise_std", 0.0))
    ffn_hd_coeffs = getattr(args, "ffn_hd_coeffs", getattr(args, "v_hd_coeffs", 1.0))

    for name in ["down_proj"]:
        if not hasattr(mlp0, name):
            raise RuntimeError(f"LlamaMLP has no {name}; cannot run gaussian_FFN_SFLI")

        lin = getattr(mlp0, name)  # Linear(768 -> 2048, bias=False) by default

        # ---- 确保 bias=True ----
        if lin.bias is None:
            lin = _replace_linear_with_bias(lin)
            setattr(mlp0, name, lin)

        # ---- std：若没给 ffn_base_std，走 initializer_range/sqrt(fan_in) ----
        if ffn_base_std is not None:
            std = float(ffn_base_std)
        else:
            fan_in = lin.in_features
            init_range = getattr(model.config, "initializer_range", 0.02)
            std = float(init_range / (fan_in ** 0.5))

        with torch.no_grad():
            lin.weight.normal_(mean=0.0, std=std)
            #lin.weight.div_(lin.weight.norm(dim=1, keepdim=True).clamp_min(1e-12))
            lin.weight.mul_(ffn_hd_coeffs)

            out_dim = lin.out_features  # 2048
            arith = ffn_bias_start + ffn_bias_step * torch.arange(
                out_dim, device=lin.weight.device, dtype=lin.weight.dtype
            )
            noise = torch.randn(out_dim, device=lin.weight.device, dtype=lin.weight.dtype) * ffn_bias_noise_std
            lin.bias.copy_(arith + noise)
            lin.bias.mul_(ffn_hd_coeffs)

        try:
            print(
                f"[FFN SFLI] {name}: W mean={lin.weight.mean().item():.3e}, var={lin.weight.var(unbiased=False).item():.3e}, "
                f"b mean={lin.bias.mean().item():.3e}, var={lin.bias.var(unbiased=False).item():.3e}"
            )
        except Exception:
            pass

@torch.no_grad()
def xavier_init_all_ffn_gate_up(model):
    """
    Apply Xavier uniform init to gate_proj and up_proj weights for ALL layers' MLPs.
    (Does NOT touch embedding, attention, down_proj, etc.)
    """
    for i, mlp in enumerate(get_all_layer_mlp(model)):
        if not (hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj")):
            raise RuntimeError(f"Layer {i} mlp missing gate_proj/up_proj")

        torch.nn.init.xavier_uniform_(mlp.gate_proj.weight)
        torch.nn.init.xavier_uniform_(mlp.up_proj.weight)

@torch.no_grad()
def kaiming_init_all_ffn_gate_up(model):
    """
    Apply kaiming init to gate_proj and up_proj weights for ALL layers' MLPs.
    (Does NOT touch embedding, attention, down_proj, etc.)
    """
    for i, mlp in enumerate(get_all_layer_mlp(model)):
        if not (hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj")):
            raise RuntimeError(f"Layer {i} mlp missing gate_proj/up_proj")

        torch.nn.init.kaiming_uniform_(mlp.gate_proj.weight)
        torch.nn.init.kaiming_uniform_(mlp.up_proj.weight)

# def _add_activation_to_single_ud_proj(mlp_module, proj_name, args):
#     act_name = getattr(args, "ffn_act", "none")
#     if act_name == "none":
#         return
#
#     proj = getattr(mlp_module, proj_name)
#     if isinstance(proj, nn.Sequential):
#         return
#
#     setattr(
#         mlp_module,
#         proj_name,
#         nn.Sequential(
#             proj,
#             _make_act(act_name),
#         ),
#     )
#
#
# def _add_activation_to_ud_subset(mlp_module, args, proj_names):
#     for proj_name in proj_names:
#         _add_activation_to_single_ud_proj(mlp_module, proj_name, args)
#
#
# def add_ud_activation_u(mlp_module, args):
#     _add_activation_to_ud_subset(mlp_module, args, ["up_proj"])
#
#
# def add_ud_activation_d(mlp_module, args):
#     _add_activation_to_ud_subset(mlp_module, args, ["down_proj"])
#
#
# def add_ud_activation_ud(mlp_module, args):
#     _add_activation_to_ud_subset(mlp_module, args, ["up_proj", "down_proj"])


### lin.weight.mul_(hd_coeffs)和lin.weight.mul_(ffn_hd_coeffs)替换成(1- l/L)  + 1e-3 l为层数，L为总层数
# import torch
# from safetensors.torch import save_file
# import os
# import torch.nn as nn
# import torch.nn.functional as F
#
# def initial_embedding(model, args):
#     Dtype = model.model.embed_tokens.weight.dtype
#     Device = model.model.embed_tokens.weight.device
#
#     if args.initialization == "Xavier":
#         model.model.embed_tokens.weight = torch.nn.init.xavier_uniform_(model.model.embed_tokens.weight)
#         # 1) 第一层 Attention 的 q_proj / k_proj / v_proj 用 Xavier
#         # attn0 = get_first_layer_attention(model)
#         # [torch.nn.init.xavier_uniform_(proj.weight) for proj in (attn0.q_proj, attn0.k_proj, attn0.v_proj)]
#
#         # 2) 第一层 FFN 的 gate_proj / up_proj 用 Xavier
#         # mlp0 = get_first_layer_mlp(model)
#         # torch.nn.init.xavier_uniform_(mlp0.gate_proj.weight)
#         # torch.nn.init.xavier_uniform_(mlp0.up_proj.weight)
#
#         # 3) 所有层的 FFN 的 gate_proj / up_proj 用 Xavier
#         xavier_init_all_ffn_gate_up(model)
#
#
#     elif args.initialization == "Kaiming":
#         model.model.embed_tokens.weight = torch.nn.init.kaiming_uniform_(model.model.embed_tokens.weight)
#
#
#         # attn0 = get_first_layer_attention(model)
#         # [torch.nn.init.kaiming_uniform_(proj.weight) for proj in (attn0.q_proj, attn0.k_proj, attn0.v_proj)]
#         kaiming_init_all_ffn_gate_up(model)
#
#     elif args.initialization == "erank_kernel":
#         construc_weight = kernel_fun(args, model.model.embed_tokens.weight)
#         construc_weight = construc_weight.to(dtype=Dtype, device=Device)
#         model.model.embed_tokens.weight.data = construc_weight
#
#     elif args.initialization == "erank_basis":
#         construc_weight = basis_fun(args, model.model.embed_tokens.weight)
#         construc_weight = construc_weight.to(dtype=Dtype, device=Device)
#         model.model.embed_tokens.weight.data = construc_weight
#
#     elif args.initialization == "erank_implicit":
#         construc_weight = implicit_fun(args, model.model.embed_tokens.weight)
#         construc_weight = construc_weight.to(dtype=Dtype, device=Device)
#         model.model.embed_tokens.weight.data = construc_weight
#
#         attn0 = get_first_layer_attention(model)
#         init_qkv_with_custom_gaussian_and_bias(attn0, model, args)
#         add_qkv_activation(attn0, args)
#     elif args.initialization == "Xavier_plus_SFLI":
#         model.model.embed_tokens.weight = torch.nn.init.xavier_uniform_(model.model.embed_tokens.weight)
#
#         attn0 = get_first_layer_attention(model)
#         init_qkv_with_custom_gaussian_and_bias(attn0, model, args)
#         add_qkv_activation(attn0, args)
#     elif args.initialization == "Xavier_FFN_SFLI":
#         model.model.embed_tokens.weight = torch.nn.init.xavier_uniform_(model.model.embed_tokens.weight)
#
#         # mlp0 = get_first_layer_mlp(model)
#         # init_ffn_with_custom_gaussian_and_bias_ug(mlp0, model, args)
#
#         mlps = get_all_layer_mlp(model)
#         L = len(mlps)
#         for l, mlp in enumerate(mlps):
#             init_ffn_with_custom_gaussian_and_bias_ug(mlp, model, args, l=l, L=L)
#
#     elif args.initialization == "Kaiming_plus_SFLI":
#         model.model.embed_tokens.weight = torch.nn.init.kaiming_uniform_(model.model.embed_tokens.weight)
#
#         attn0 = get_first_layer_attention(model)
#         init_qkv_with_custom_gaussian_and_bias(attn0, model, args)
#         add_qkv_activation(attn0, args)
#     elif args.initialization == "Kaiming_FFN_SFLI":
#         model.model.embed_tokens.weight = torch.nn.init.kaiming_uniform_(model.model.embed_tokens.weight)
#
#         # mlp0 = get_first_layer_mlp(model)
#         # init_ffn_with_custom_gaussian_and_bias_ug(mlp0, model, args)
#         mlps = get_all_layer_mlp(model)
#         L = len(mlps)
#         for l, mlp in enumerate(mlps):
#             init_ffn_with_custom_gaussian_and_bias_ug(mlp, model, args, l=l, L=L)
#
#     elif args.initialization == "Gaussian_default":
#         pass
#     elif args.initialization == "Gaussian_plus_SFLI":
#         # attn0 = get_first_layer_attention(model)
#         # init_qkv_with_custom_gaussian_and_bias(attn0, model, args)
#         # add_qkv_activation(attn0, args)
#
#         # attens = get_all_layer_attention(model)
#         # for attn in attens:
#         #     init_qkv_with_custom_gaussian_and_bias_q(attn, model, args)
#         #     add_qkv_activation_q(attn, args)
#
#         # attens = get_all_layer_attention(model)
#         # for attn in attens:
#         #     init_qkv_with_custom_gaussian_and_bias_k(attn, model, args)
#         #     add_qkv_activation_k(attn, args)
#         #
#         # attens = get_all_layer_attention(model)
#         # for attn in attens:
#         #     init_qkv_with_custom_gaussian_and_bias_v(attn, model, args)
#         #     add_qkv_activation_v(attn, args)
#         #
#         # attens = get_all_layer_attention(model)
#         # for attn in attens:
#         #     init_qkv_with_custom_gaussian_and_bias_qk(attn, model, args)
#         #     add_qkv_activation_qk(attn, args)
#         #
#         # attens = get_all_layer_attention(model)
#         # for attn in attens:
#         #     init_qkv_with_custom_gaussian_and_bias_qv(attn, model, args)
#         #     add_qkv_activation_qv(attn, args)
#
#         # attens = get_all_layer_attention(model)
#         # for attn in attens:
#         #     init_qkv_with_custom_gaussian_and_bias_kv(attn, model, args)
#         #     add_qkv_activation_kv(attn, args)
#
#         attens = get_all_layer_attention(model)
#         L = len(attens)
#         for l, attn in enumerate(attens):
#             init_qkv_with_custom_gaussian_and_bias_qkv(attn, model, args, l=l, L=L)
#             add_qkv_activation_qkv(attn, args)
#
#     elif args.initialization == "gaussian_FFN_SFLI":
#         mlps = get_all_layer_mlp(model)
#         L = len(mlps)
#         for l, mlp in enumerate(mlps):
#             init_ffn_with_custom_gaussian_and_bias_ug(mlp, model, args, l=l, L=L)
#
#     elif args.initialization == "gaussian_FFN_SFLI_g":
#         mlps = get_all_layer_mlp(model)
#         L = len(mlps)
#         for l, mlp in enumerate(mlps):
#             init_ffn_with_custom_gaussian_and_bias_g(mlp, model, args, l=l, L=L)
#     elif args.initialization == "gaussian_FFN_SFLI_u":
#         mlps = get_all_layer_mlp(model)
#         L = len(mlps)
#         for l, mlp in enumerate(mlps):
#             init_ffn_with_custom_gaussian_and_bias_u(mlp, model, args, l=l, L=L)
#
#     elif args.initialization == "gaussian_FFN_SFLI_d":
#         mlps = get_all_layer_mlp(model)
#         L = len(mlps)
#         for l, mlp in enumerate(mlps):
#             init_ffn_with_custom_gaussian_and_bias_d(mlp, model, args, l=l, L=L)
#
#     elif args.initialization == "gaussian_FFN_SFLI_gd":
#         mlps = get_all_layer_mlp(model)
#         L = len(mlps)
#         for l, mlp in enumerate(mlps):
#             init_ffn_with_custom_gaussian_and_bias_gd(mlp, model, args, l=l, L=L)
#
#     elif args.initialization == "gaussian_FFN_SFLI_ud":
#         mlps = get_all_layer_mlp(model)
#         L = len(mlps)
#         for l, mlp in enumerate(mlps):
#             init_ffn_with_custom_gaussian_and_bias_ud(mlp, model, args, l=l, L=L)
#
#     elif args.initialization == "gaussian_FFN_SFLI_ugd":
#         mlps = get_all_layer_mlp(model)
#         L = len(mlps)
#         for l, mlp in enumerate(mlps):
#             init_ffn_with_custom_gaussian_and_bias_ugd(mlp, model, args, l=l, L=L)
#
#     elif args.initialization == "gaussian_best_combination":
#         attens = get_all_layer_attention(model)
#         # attens = [get_first_layer_attention(model)]
#         # attens = get_25_layer_attention(model)
#         # attens = get_33_layer_attention(model)
#         # attens = get_50_layer_attention(model)
#         # attens = get_66_layer_attention(model)
#         L = len(attens)
#         for l, attn in enumerate(attens):
#             init_qkv_with_custom_gaussian_and_bias_v(attn, model, args, l=l, L=L)
#             add_qkv_activation_v(attn, args)
#         mlps = get_all_layer_mlp(model)
#         # mlps = [get_first_layer_mlp(model)]
#         # mlps = get_25_layer_mlp(model)
#         # mlps = get_33_layer_mlp(model)
#         # mlps = get_50_layer_mlp(model)
#         # mlps = get_66_layer_mlp(model)
#         L = len(mlps)
#         for l, mlp in enumerate(mlps):
#             init_ffn_with_custom_gaussian_and_bias_g(mlp, model, args, l=l, L=L)
#
#     return model
#
# def kernel_fun(args,weight, is_random=False):
#     M, N = weight.shape
#     x_int=1
#     dx_int=1
#     x = torch.linspace(-x_int, x_int, M).reshape(-1, 1)
#     dx = torch.linspace(-dx_int, dx_int, N).reshape(1, -1)
#
#     if args.init_name == 'exp':
#         scale = 1e3
#         lamb = 1e5
#         fun=lambda x: scale * (torch.exp(-lamb * x ** 2))
#     elif args.init_name  == 'sinc':
#         scale = args.sinc_scale #1e2
#         lamb = args.sinc_lamb #1e5
#         fun= lambda x: scale * torch.sinc(lamb * x)
#     elif args.init_name  == 'tanh':
#         scale = 1e0
#         lamb = 1e6
#         fun= lambda x: scale * torch.tanh(lamb * x)
#     else:
#         raise ValueError(f"no '{args.init_name }' kernal function")
#
#     new_weight = fun(x-dx)
#     return new_weight
#
# def basis_fun(args,weight, is_random=False):
#     M, N = weight.shape
#     if args.init_name  == 'fourier':
#         scale = 1e-1
#         x=torch.linspace(0, 2*torch.pi, M+1)[:M]
#         k=torch.arange(N)
#         new_weight = scale*torch.real(torch.exp(1j*k.reshape(1, -1)*x.reshape(-1, 1)))
#     elif args.init_name  == 'chebyshev':
#         scale = 1e-1
#         x = torch.cos(torch.pi * torch.arange(M) / (M - 1))
#         k = torch.arange(N)
#         new_weight = scale*torch.cos(k.reshape(1, -1) * torch.arccos(x.reshape(-1, 1)))
#     else:
#         raise ValueError(f"no '{args.init_name }' basis function")
#     return new_weight
#
# def implicit_fun(args ,weight,exp_max=21290,exp_min=20,eig=None):
#     M, N = weight.shape
#     if args.init_name  == 'qr':
#         scale = 2e-1
#         if eig is None:
#             alpha=2.01
#             eig=torch.tan(torch.linspace(-torch.pi/alpha,torch.pi/alpha,N))
#             eig_min,eig_max=torch.min(eig),torch.max(eig)
#             eig=exp_min + (eig - eig_min) * (exp_max - exp_min) / (eig_max - eig_min)
#         random_matrix = torch.randn(M,N)
#         q, _ = torch.linalg.qr(random_matrix)
#         q = q[:, :N]
#
#         new_weight = scale*q@torch.diag(torch.sqrt(eig))
#         print(f'mean:{new_weight.mean()}, variance:{new_weight.var(unbiased=False)}')
#
#         var = new_weight.var(unbiased=False)
#         factor = 0.01 / torch.sqrt(var)
#         result = factor * new_weight
#         print(f'mean:{result.mean()}, variance:{result.var(unbiased=False)}')
#         new_weight = result
#     else:
#         raise ValueError(f"no '{args.init_name }' implicit method")
#     return new_weight
#
# def _replace_linear_with_bias(linear: nn.Linear) -> nn.Linear:
#     new = nn.Linear(linear.in_features, linear.out_features, bias=True, device=linear.weight.device, dtype=linear.weight.dtype)
#     with torch.no_grad():
#         new.weight.copy_(linear.weight)
#         new.bias.zero_()
#     return new
#
# def get_first_layer_attention(model):
#     layer0 = model.model.layers[0]  # HF LLaMA 通常是这个
#     if hasattr(layer0, "self_attn"):
#         return layer0.self_attn
#
#     # # 兜底：在 layer0 内找第一个带 q/k/v 的模块
#     # for _, m in layer0.named_modules():
#     #     if hasattr(m, "q_proj") and hasattr(m, "k_proj") and hasattr(m, "v_proj"):
#     #         return m
#
#     raise RuntimeError("Cannot find attention module with q_proj/k_proj/v_proj in model.model.layers[0]")
#
# def get_all_layer_attention(model):
#     attns = []
#     for i, layer in enumerate(model.model.layers):
#         if hasattr(layer, "self_attn"):
#             attns.append(layer.self_attn)
#         else:
#             raise RuntimeError(f"Cannot find self_attn in model.model.layers[{i}]")
#     return attns
#
# def get_25_layer_attention(model):
#     attns = []
#     for i, layer in enumerate(model.model.layers[:3]):
#         if hasattr(layer, "self_attn"):
#             attns.append(layer.self_attn)
#         else:
#             raise RuntimeError(f"Cannot find self_attn in model.model.layers[{i}]")
#     return attns
#
# def get_33_layer_attention(model):
#     attns = []
#     for i, layer in enumerate(model.model.layers[:4]):  # only layers 0,1,2,3
#         if hasattr(layer, "self_attn"):
#             attns.append(layer.self_attn)
#         else:
#             raise RuntimeError(f"Cannot find self_attn in model.model.layers[{i}]")
#     return attns
#
# def get_50_layer_attention(model):
#     attns = []
#     for i, layer in enumerate(model.model.layers[:6]):
#         if hasattr(layer, "self_attn"):
#             attns.append(layer.self_attn)
#         else:
#             raise RuntimeError(f"Cannot find self_attn in model.model.layers[{i}]")
#     return attns
#
# def get_66_layer_attention(model):
#     attns = []
#     for i, layer in enumerate(model.model.layers[:8]):
#         if hasattr(layer, "self_attn"):
#             attns.append(layer.self_attn)
#         else:
#             raise RuntimeError(f"Cannot find self_attn in model.model.layers[{i}]")
#     return attns
#
# def _init_single_qkv_proj(attn_module, proj_name, model, args, l=0, L=1):
#     """
#     对 attn_module.{q_proj,k_proj,v_proj} 中的某一个做 SFLI 初始化：
#     1) 确保 bias=True
#     2) weight ~ N(0, std^2) * hd_coeffs
#     3) bias = arithmetic progression + noise, then * hd_coeffs
#     """
#     if proj_name in ["q_proj"]:
#         base_std = getattr(args, "q_qkv_std", None)
#         bias_start = getattr(args, "q_bias_start", 0.0)
#         bias_step = getattr(args, "q_bias_step", 0.0)
#         bias_noise_std = getattr(args, "q_bias_noise_std", 0.0)
#         hd_coeffs = getattr(args, "q_hd_coeffs", 1.0)
#     elif proj_name in ["k_proj"]:
#         base_std = getattr(args, "k_qkv_std", None)
#         bias_start = getattr(args, "k_bias_start", 0.0)
#         bias_step = getattr(args, "k_bias_step", 0.0)
#         bias_noise_std = getattr(args, "k_bias_noise_std", 0.0)
#         hd_coeffs = getattr(args, "k_hd_coeffs", 1.0)
#     elif proj_name == "v_proj":
#         base_std = getattr(args, "v_qkv_std", None)
#         bias_start = getattr(args, "v_bias_start", 0.0)
#         bias_step = getattr(args, "v_bias_step", 0.0)
#         bias_noise_std = getattr(args, "v_bias_noise_std", 0.0)
#         hd_coeffs = getattr(args, "v_hd_coeffs", 1.0)
#     else:
#         raise ValueError(f"Unknown proj_name: {proj_name}")
#
#     lin = getattr(attn_module, proj_name)
#
#     if lin.bias is None:
#         lin = _replace_linear_with_bias(lin)
#         setattr(attn_module, proj_name, lin)
#
#     if base_std is not None:
#         std = float(base_std)
#     else:
#         fan_in = lin.in_features
#         init_range = getattr(model.config, "initializer_range", 0.02)
#         std = float(init_range / (fan_in ** 0.5))
#
#     with torch.no_grad():
#         lin.weight.normal_(mean=0.0, std=std)
#         lin.weight.div_(lin.weight.norm(dim=1, keepdim=True).clamp_min(1e-12))
#         lin.weight.mul_((1 - l / L) + 1e-3)
#
#         out_dim = lin.out_features
#         arith = bias_start + bias_step * torch.arange(
#             out_dim, device=lin.weight.device, dtype=lin.weight.dtype
#         )
#         noise = torch.randn(
#             out_dim, device=lin.weight.device, dtype=lin.weight.dtype
#         ) * bias_noise_std
#         lin.bias.copy_(arith + noise)
#         lin.bias.mul_(hd_coeffs)
#
#         # print(
#         #     f"[QKV SFLI] {proj_name}: "
#         #     f"W mean={lin.weight.mean().item():.3e}, "
#         #     f"var={lin.weight.var(unbiased=False).item():.3e}, "
#         #     f"b mean={lin.bias.mean().item():.3e}, "
#         #     f"var={lin.bias.var(unbiased=False).item():.3e}")
#
# def _init_qkv_subset(attn_module, model, args, proj_names, l=0, L=1):
#     for proj_name in proj_names:
#         _init_single_qkv_proj(attn_module, proj_name, model, args, l=l, L=L)
#
# def init_qkv_with_custom_gaussian_and_bias_q(attn_module, model, args, l=0, L=1):
#     _init_qkv_subset(attn_module, model, args, ["q_proj"], l=l, L=L)
#
# def init_qkv_with_custom_gaussian_and_bias_k(attn_module, model, args, l=0, L=1):
#     _init_qkv_subset(attn_module, model, args, ["k_proj"], l=l, L=L)
#
# def init_qkv_with_custom_gaussian_and_bias_v(attn_module, model, args, l=0, L=1):
#     _init_qkv_subset(attn_module, model, args, ["v_proj"], l=l, L=L)
#
# def init_qkv_with_custom_gaussian_and_bias_qk(attn_module, model, args, l=0, L=1):
#     _init_qkv_subset(attn_module, model, args, ["q_proj", "k_proj"], l=l, L=L)
#
# def init_qkv_with_custom_gaussian_and_bias_qv(attn_module, model, args, l=0, L=1):
#     _init_qkv_subset(attn_module, model, args, ["q_proj", "v_proj"], l=l, L=L)
#
# def init_qkv_with_custom_gaussian_and_bias_kv(attn_module, model, args, l=0, L=1):
#     _init_qkv_subset(attn_module, model, args, ["k_proj", "v_proj"], l=l, L=L)
#
# def init_qkv_with_custom_gaussian_and_bias_qkv(attn_module, model, args, l=0, L=1):
#     _init_qkv_subset(attn_module, model, args, ["q_proj", "k_proj", "v_proj"], l=l, L=L)
#
# class SincAct(nn.Module):
#     def forward(self, x):
#         return torch.sinc(x)
#
# class CosAct(nn.Module):
#     def forward(self, x):
#         return torch.cos(x)
#
# class SinAct(nn.Module):
#     def forward(self, x):
#         return torch.sin(x)
#
# class HatAct(nn.Module):
#     """
#     hat(x) = max(0, 1 - |x|)
#     也叫 triangular / tent function
#     """
#     def forward(self, x):
#         return torch.clamp(1.0 - x.abs(), min=0.0)
#
# class SiLUAct(nn.Module):
#     def forward(self, x):
#         return torch.nn.functional.silu(x)
#
# class ChebyshevAct(nn.Module):
#     """
#     f(x) = sum_{k=0}^N a_k T_k(z),  z = clip(x / L, -1, 1)
#     """
#     def __init__(
#         self,
#         N: int = 8,
#         L: float = 1.0,
#         clamp: bool = True,
#         init: str = "identity",
#         sigma: float = 0.02,   # <-- 新增：gaussian_init 用的 std
#     ):
#         super().__init__()
#         assert N >= 0
#         self.N = int(N)
#         self.L = float(L)
#         self.clamp = bool(clamp)
#         self.sigma = float(sigma)
#
#         # Learnable coefficients a_0..a_N
#         self.a = nn.Parameter(torch.zeros(self.N + 1))
#
#         if init == "identity":
#             with torch.no_grad():
#                 self.a.zero_()
#                 if self.N >= 1:
#                     self.a[1].fill_(1.0)
#         elif init == "zero":
#             with torch.no_grad():
#                 self.a.zero_()
#         elif init == "gaussian_init":
#             with torch.no_grad():
#                 # a_k ~ N(0, sigma^2)
#                 self.a.normal_(mean=0.0, std=self.sigma)
#         else:
#             raise ValueError(f"Unknown init: {init}")
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         z = x / self.L
#         if self.clamp:
#             z = torch.clamp(z, -1.0, 1.0)
#
#         a = self.a.to(dtype=x.dtype, device=x.device)
#
#         y = a[0] * torch.ones_like(z)
#         if self.N == 0:
#             return y
#
#         Tkm2 = torch.ones_like(z)  # T0
#         Tkm1 = z                   # T1
#         y = y + a[1] * Tkm1
#
#         for k in range(2, self.N + 1):
#             Tk = 2.0 * z * Tkm1 - Tkm2
#             y = y + a[k] * Tk
#             Tkm2, Tkm1 = Tkm1, Tk
#
#         return y
#
# class SincPolyAct(nn.Module):
#     """
#     phi_multi(x) = sum_{i=1..M} sum_{j=-N..N} c_{i,j} * S(j, h_i)(x)
#     where S(j,h)(x) = sin((pi/h)(x-jh)) / ((pi/h)(x-jh)) = sinc(x/h - j)
#
#     - M, N, h0: manually chosen
#     - c_{i,j}: learnable parameters, shape (M, 2N+1)
#     - h_i: inverse decay schedule
#     """
#     def __init__(
#         self,
#         M: int = 4,
#         N: int = 8,
#         h0: float = 1.0,
#         inverse_decay: str = "h0_over_i",  # "h0_over_i" or "one_over_i_h0"
#         init: str = "identity_like",       # "identity_like" or "zero"
#         dtype=None,
#         device=None,
#     ):
#         super().__init__()
#         assert M >= 1 and N >= 0
#         self.M = int(M)
#         self.N = int(N)
#         self.h0 = float(h0)
#         self.inverse_decay = inverse_decay
#
#         # learnable coefficients c_{i,j}
#         self.c = nn.Parameter(torch.zeros(self.M, 2 * self.N + 1, dtype=dtype, device=device))
#
#         # pre-store integer j grid as a buffer (moves with .to(device))
#         j = torch.arange(-self.N, self.N + 1, dtype=dtype if dtype is not None else torch.float32, device=device)
#         self.register_buffer("j_grid", j, persistent=False)
#
#         self.reset_parameters(init)
#
#     def _make_h(self, x: torch.Tensor) -> torch.Tensor:
#         # h_i schedule (inverse decay)
#         i = torch.arange(1, self.M + 1, device=x.device, dtype=x.dtype)
#         if self.inverse_decay == "h0_over_i":
#             # h_i = h0 / i   (most common "inverse decay")
#             h = (self.h0 / i)
#         elif self.inverse_decay == "one_over_i_h0":
#             # interpret "h_i = 1 / (i * h0)"
#             h = 1.0 / (i * self.h0)
#         else:
#             raise ValueError(f"Unknown inverse_decay: {self.inverse_decay}")
#         return h  # shape (M,)
#
#     @torch.no_grad()
#     def reset_parameters(self, init: str = "identity_like"):
#         if init == "zero":
#             self.c.zero_()
#             return
#
#         if init == "identity_like":
#             # 目标：一开始尽量不伤训练（让激活像“近似恒等/近似线性”）
#             # sinc 在 0 附近 ~ 1 - O(z^2)，所以用中心 j=0 的基函数做一个“平滑门”
#             # 这里给 i=1 的 (j=0) 一个 1，其余 0
#             self.c.zero_()
#             center = self.N  # j=0 对应的位置
#             self.c[0, center] = 1.0
#             return
#
#         raise ValueError(f"Unknown init: {init}")
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: any shape [...]
#         return: same shape [...]
#         """
#         # compute basis: S(j, h_i)(x) = sinc(x/h_i - j)
#         # shapes:
#         #   x[..., None, None] -> [..., 1, 1]
#         #   h[None, :, None]   -> [1, M, 1]
#         #   j[None, None, :]   -> [1, 1, 2N+1]
#         h = self._make_h(x)  # (M,)
#         j = self.j_grid.to(device=x.device, dtype=x.dtype)  # (2N+1,)
#
#         z = x[..., None, None] / h[None, :, None] - j[None, None, :]  # [..., M, 2N+1]
#         basis = torch.sinc(z)  # [..., M, 2N+1]
#
#         # weighted sum with c_{i,j}
#         # c: (M, 2N+1) -> broadcast to [..., M, 2N+1]
#         y = (basis * self.c.to(dtype=x.dtype, device=x.device)).sum(dim=(-2, -1))  # [...]
#         return y
#
# def _elu1(x: torch.Tensor) -> torch.Tensor:
#     # ELU(x, alpha=1): range (-1, +inf)
#     return F.elu(x, alpha=1.0)
#
# def jacobi_P(n: int, a: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
#     """
#     Compute Jacobi polynomial P_n^{(a,b)}(x) for x in [-1,1].
#     a,b can be broadcastable to x (e.g., scalar or [C] with x shape [..., C]).
#     Recurrence is standard (see e.g. Wikipedia / Szegő).
#     """
#     if n == 0:
#         return torch.ones_like(x)
#     if n == 1:
#         return 0.5 * ((2.0 + a + b) * x + (a - b))
#
#     Pnm2 = torch.ones_like(x)  # P_0
#     Pnm1 = 0.5 * ((2.0 + a + b) * x + (a - b))  # P_1
#
#     for k in range(2, n + 1):
#         kf = float(k)
#         two_k_ab = 2.0 * kf + a + b
#         # coefficients
#         A = (two_k_ab - 1.0) * ((two_k_ab) * (two_k_ab - 2.0) * x + (a*a - b*b))
#         B = 2.0 * (kf + a - 1.0) * (kf + b - 1.0) * (two_k_ab)
#         C = 2.0 * kf * (kf + a + b) * (two_k_ab - 2.0)
#
#         Pn = (A * Pnm1 - B * Pnm2) / C
#         Pnm2, Pnm1 = Pnm1, Pn
#
#     return Pnm1
#
# class FractionalJacobiAct(nn.Module):
#     """
#     A pointwise activation based on *fractional-order* Jacobi functions, in the spirit of fKAN:
#         z = sigmoid(x) in (0,1)
#         t = 2*z^gamma - 1   (maps to [-1,1])
#         y = P_n^{(alpha,beta)}(t)
#
#     Parameters alpha,beta are constrained to (-1, +inf) via ELU(.,1),
#     gamma constrained to (0,1) via sigmoid.
#     """
#     def __init__(
#         self,
#         degree: int = 8,
#         per_channel: bool = False,
#         channels: int = 768,
#         alpha_init: float = 0.0,
#         beta_init: float = 0.0,
#         gamma_init: float = 0.5,
#     ):
#         super().__init__()
#         self.degree = int(degree)
#         self.per_channel = bool(per_channel)
#
#         shape = (channels,) if per_channel else (1,)
#         self.alpha_raw = nn.Parameter(torch.full(shape, float(alpha_init)))
#         self.beta_raw  = nn.Parameter(torch.full(shape, float(beta_init)))
#         self.gamma_raw = nn.Parameter(torch.full(shape, float(gamma_init)))
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x shape: [..., C] typically
#         # constrain parameters
#         alpha = _elu1(self.alpha_raw)  # (-1, inf)
#         beta  = _elu1(self.beta_raw)   # (-1, inf)
#         gamma = torch.sigmoid(self.gamma_raw)  # (0,1)
#
#         # broadcast to x
#         while alpha.dim() < x.dim():
#             alpha = alpha.unsqueeze(0)
#             beta  = beta.unsqueeze(0)
#             gamma = gamma.unsqueeze(0)
#
#         z = torch.sigmoid(x)                 # (0,1)
#         t = 2.0 * torch.pow(z, gamma) - 1.0  # [-1,1]
#         return jacobi_P(self.degree, alpha, beta, t)
#
# def _make_act(name: str) -> nn.Module:
#     if name == "none":
#         return nn.Identity()
#     if name == "tanh":
#         return nn.Tanh()
#     if name == "sinc":
#         return SincAct()
#     if name == "cos":
#         return CosAct()
#     if name == "sin":
#         return SinAct()
#     if name == "hat":
#         return HatAct()
#     if name == "silu":
#         return SiLUAct()
#     if name == "cheb":
#         # 你可以在 argparse 里加 --cheb_N --cheb_L
#         N = 16
#         L = 768
#         clamp = True
#         init = "gaussian_init"
#         sigma = 0.1
#         return ChebyshevAct(N=N, L=L, clamp=clamp, init=init, sigma=sigma)
#     if name == "sinc_poly":
#         M = 3
#         N = 16
#         sinc_h0 = 2
#         return SincPolyAct(M=M, N=N, h0=sinc_h0)
#     if name == "frac_jacobi":
#         # 建议：先用对称的 Legendre-like 初始化 (alpha=beta=0)，gamma=0.5
#         # 如果你想“每个通道一套参数”，per_channel=True（更灵活但更不稳定/更贵）
#         return FractionalJacobiAct(
#             degree= 8,
#             per_channel= False,
#             channels= 768,  # 你也可以直接写死 768
#             alpha_init= 0.0,
#             beta_init= 0.0,
#             gamma_init= 1, # 取[-3,3]，太大太小会导致sigmoid饱和影响梯度
#         )
#
#     raise ValueError(f"Unknown qkv_act: {name}")
#
# def _add_activation_to_single_qkv_proj(attn_module, proj_name, args):
#     act_name = getattr(args, "qkv_act", "none")
#     if act_name == "none":
#         return
#
#     proj = getattr(attn_module, proj_name)
#     if isinstance(proj, nn.Sequential):
#         return
#
#     setattr(
#         attn_module,
#         proj_name,
#         nn.Sequential(
#             proj,
#             _make_act(act_name),
#         ),
#     )
#
# def _add_activation_to_qkv_subset(attn_module, args, proj_names):
#     for proj_name in proj_names:
#         _add_activation_to_single_qkv_proj(attn_module, proj_name, args)
#
# def add_qkv_activation_q(attn_module, args):
#     _add_activation_to_qkv_subset(attn_module, args, ["q_proj"])
#
# def add_qkv_activation_k(attn_module, args):
#     _add_activation_to_qkv_subset(attn_module, args, ["k_proj"])
#
# def add_qkv_activation_v(attn_module, args):
#     _add_activation_to_qkv_subset(attn_module, args, ["v_proj"])
#
# def add_qkv_activation_qk(attn_module, args):
#     _add_activation_to_qkv_subset(attn_module, args, ["q_proj", "k_proj"])
#
# def add_qkv_activation_qv(attn_module, args):
#     _add_activation_to_qkv_subset(attn_module, args, ["q_proj", "v_proj"])
#
# def add_qkv_activation_kv(attn_module, args):
#     _add_activation_to_qkv_subset(attn_module, args, ["k_proj", "v_proj"])
#
# def add_qkv_activation_qkv(attn_module, args):
#     _add_activation_to_qkv_subset(attn_module, args, ["q_proj", "k_proj", "v_proj"])
#
# def get_first_layer_mlp(model):
#     layer0 = model.model.layers[0]
#     if hasattr(layer0, "mlp"):
#         return layer0.mlp
#     raise RuntimeError("Cannot find mlp in model.model.layers[0]")
#
# def get_all_layer_mlp(model):
#     mlps = [];
#     for i, layer in enumerate(model.model.layers):
#         if hasattr(layer, "mlp"):
#             mlps.append(layer.mlp)
#         else:
#             raise RuntimeError("Cannot find mlp in model.model.layers")
#     return mlps
#
# def get_25_layer_mlp(model):
#     mlps = []
#     for i, layer in enumerate(model.model.layers[:3]):
#         if hasattr(layer, "mlp"):
#             mlps.append(layer.mlp)
#         else:
#             raise RuntimeError(f"Cannot find mlp in model.model.layers[{i}]")
#     return mlps
#
# def get_33_layer_mlp(model):
#     mlps = []
#     for i, layer in enumerate(model.model.layers[:4]):  # only layers 0,1,2,3
#         if hasattr(layer, "mlp"):
#             mlps.append(layer.mlp)
#         else:
#             raise RuntimeError(f"Cannot find mlp in model.model.layers[{i}]")
#     return mlps
#
# def get_50_layer_mlp(model):
#     mlps = []
#     for i, layer in enumerate(model.model.layers[:6]):
#         if hasattr(layer, "mlp"):
#             mlps.append(layer.mlp)
#         else:
#             raise RuntimeError(f"Cannot find mlp in model.model.layers[{i}]")
#     return mlps
#
# def get_66_layer_mlp(model):
#     mlps = []
#     for i, layer in enumerate(model.model.layers[:8]):
#         if hasattr(layer, "mlp"):
#             mlps.append(layer.mlp)
#         else:
#             raise RuntimeError(f"Cannot find mlp in model.model.layers[{i}]")
#     return mlps
#
# def init_ffn_with_custom_gaussian_and_bias_ug(mlp0, model, args, l=0, L=1):
#     """
#     对 layer0.mlp 的 gate_proj 和 up_proj 做 SFLI初始化：
#       - W ~ N(0, std^2) * ffn_hd_coeffs
#       - b = 等差(arith) + 噪声(noise)，再 * ffn_hd_coeffs
#     """
#
#     # ---- args（优先 ffn_*，否则 fallback 到 v_* / v_hd_coeffs）----
#     ffn_base_std = getattr(args, "ffn_base_std", None)
#     ffn_bias_start = getattr(args, "ffn_bias_start", getattr(args, "v_bias_start", 0.0))
#     ffn_bias_step = getattr(args, "ffn_bias_step", getattr(args, "v_bias_step", 0.0))
#     ffn_bias_noise_std = getattr(args, "ffn_bias_noise_std", getattr(args, "v_bias_noise_std", 0.0))
#     ffn_hd_coeffs = getattr(args, "ffn_hd_coeffs", getattr(args, "v_hd_coeffs", 1.0))
#
#     for name in ["gate_proj", "up_proj"]:
#         if not hasattr(mlp0, name):
#             raise RuntimeError(f"LlamaMLP has no {name}; cannot run gaussian_FFN_SFLI")
#
#         lin = getattr(mlp0, name)  # Linear(768 -> 2048, bias=False) by default
#
#         # ---- 确保 bias=True ----
#         if lin.bias is None:
#             lin = _replace_linear_with_bias(lin)
#             setattr(mlp0, name, lin)
#
#         # ---- std：若没给 ffn_base_std，走 initializer_range/sqrt(fan_in) ----
#         if ffn_base_std is not None:
#             std = float(ffn_base_std)
#         else:
#             fan_in = lin.in_features
#             init_range = getattr(model.config, "initializer_range", 0.02)
#             std = float(init_range / (fan_in ** 0.5))
#
#         with torch.no_grad():
#             lin.weight.normal_(mean=0.0, std=std)
#             lin.weight.div_(lin.weight.norm(dim=1, keepdim=True).clamp_min(1e-12))
#             lin.weight.mul_((1 - l / L) + 1e-3)
#
#             out_dim = lin.out_features  # 2048
#             arith = ffn_bias_start + ffn_bias_step * torch.arange(
#                 out_dim, device=lin.weight.device, dtype=lin.weight.dtype
#             )
#             noise = torch.randn(out_dim, device=lin.weight.device, dtype=lin.weight.dtype) * ffn_bias_noise_std
#             lin.bias.copy_(arith + noise)
#             lin.bias.mul_(ffn_hd_coeffs)
#
#         try:
#             print(
#                 f"[FFN SFLI] {name}: W mean={lin.weight.mean().item():.3e}, var={lin.weight.var(unbiased=False).item():.3e}, "
#                 f"b mean={lin.bias.mean().item():.3e}, var={lin.bias.var(unbiased=False).item():.3e}"
#             )
#         except Exception:
#             pass
#
# def init_ffn_with_custom_gaussian_and_bias_gd(mlp0, model, args, l=0, L=1):
#     """
#     对 layer0.mlp 的 gate_proj 和 up_proj 做 SFLI初始化：
#       - W ~ N(0, std^2) * ffn_hd_coeffs
#       - b = 等差(arith) + 噪声(noise)，再 * ffn_hd_coeffs
#     """
#
#     # ---- args（优先 ffn_*，否则 fallback 到 v_* / v_hd_coeffs）----
#     ffn_base_std = getattr(args, "ffn_base_std", None)
#     ffn_bias_start = getattr(args, "ffn_bias_start", getattr(args, "v_bias_start", 0.0))
#     ffn_bias_step = getattr(args, "ffn_bias_step", getattr(args, "v_bias_step", 0.0))
#     ffn_bias_noise_std = getattr(args, "ffn_bias_noise_std", getattr(args, "v_bias_noise_std", 0.0))
#     ffn_hd_coeffs = getattr(args, "ffn_hd_coeffs", getattr(args, "v_hd_coeffs", 1.0))
#
#     for name in ["gate_proj", "down_proj"]:
#         if not hasattr(mlp0, name):
#             raise RuntimeError(f"LlamaMLP has no {name}; cannot run gaussian_FFN_SFLI")
#
#         lin = getattr(mlp0, name)  # Linear(768 -> 2048, bias=False) by default
#
#         # ---- 确保 bias=True ----
#         if lin.bias is None:
#             lin = _replace_linear_with_bias(lin)
#             setattr(mlp0, name, lin)
#
#         # ---- std：若没给 ffn_base_std，走 initializer_range/sqrt(fan_in) ----
#         if ffn_base_std is not None:
#             std = float(ffn_base_std)
#         else:
#             fan_in = lin.in_features
#             init_range = getattr(model.config, "initializer_range", 0.02)
#             std = float(init_range / (fan_in ** 0.5))
#
#         with torch.no_grad():
#             lin.weight.normal_(mean=0.0, std=std)
#             lin.weight.div_(lin.weight.norm(dim=1, keepdim=True).clamp_min(1e-12))
#             lin.weight.mul_((1 - l / L) + 1e-3)
#
#             out_dim = lin.out_features  # 2048
#             arith = ffn_bias_start + ffn_bias_step * torch.arange(
#                 out_dim, device=lin.weight.device, dtype=lin.weight.dtype
#             )
#             noise = torch.randn(out_dim, device=lin.weight.device, dtype=lin.weight.dtype) * ffn_bias_noise_std
#             lin.bias.copy_(arith + noise)
#             lin.bias.mul_(ffn_hd_coeffs)
#
#         try:
#             print(
#                 f"[FFN SFLI] {name}: W mean={lin.weight.mean().item():.3e}, var={lin.weight.var(unbiased=False).item():.3e}, "
#                 f"b mean={lin.bias.mean().item():.3e}, var={lin.bias.var(unbiased=False).item():.3e}"
#             )
#         except Exception:
#             pass
#
# def init_ffn_with_custom_gaussian_and_bias_ud(mlp0, model, args, l=0, L=1):
#     """
#     对 layer0.mlp 的 gate_proj 和 up_proj 做 SFLI初始化：
#       - W ~ N(0, std^2) * ffn_hd_coeffs
#       - b = 等差(arith) + 噪声(noise)，再 * ffn_hd_coeffs
#     """
#
#     # ---- args（优先 ffn_*，否则 fallback 到 v_* / v_hd_coeffs）----
#     ffn_base_std = getattr(args, "ffn_base_std", None)
#     ffn_bias_start = getattr(args, "ffn_bias_start", getattr(args, "v_bias_start", 0.0))
#     ffn_bias_step = getattr(args, "ffn_bias_step", getattr(args, "v_bias_step", 0.0))
#     ffn_bias_noise_std = getattr(args, "ffn_bias_noise_std", getattr(args, "v_bias_noise_std", 0.0))
#     ffn_hd_coeffs = getattr(args, "ffn_hd_coeffs", getattr(args, "v_hd_coeffs", 1.0))
#
#     for name in ["down_proj", "up_proj"]:
#         if not hasattr(mlp0, name):
#             raise RuntimeError(f"LlamaMLP has no {name}; cannot run gaussian_FFN_SFLI")
#
#         lin = getattr(mlp0, name)  # Linear(768 -> 2048, bias=False) by default
#
#         # ---- 确保 bias=True ----
#         if lin.bias is None:
#             lin = _replace_linear_with_bias(lin)
#             setattr(mlp0, name, lin)
#
#         # ---- std：若没给 ffn_base_std，走 initializer_range/sqrt(fan_in) ----
#         if ffn_base_std is not None:
#             std = float(ffn_base_std)
#         else:
#             fan_in = lin.in_features
#             init_range = getattr(model.config, "initializer_range", 0.02)
#             std = float(init_range / (fan_in ** 0.5))
#
#         with torch.no_grad():
#             lin.weight.normal_(mean=0.0, std=std)
#             lin.weight.div_(lin.weight.norm(dim=1, keepdim=True).clamp_min(1e-12))
#             lin.weight.mul_((1 - l / L) + 1e-3)
#
#             out_dim = lin.out_features  # 2048
#             arith = ffn_bias_start + ffn_bias_step * torch.arange(
#                 out_dim, device=lin.weight.device, dtype=lin.weight.dtype
#             )
#             noise = torch.randn(out_dim, device=lin.weight.device, dtype=lin.weight.dtype) * ffn_bias_noise_std
#             lin.bias.copy_(arith + noise)
#             lin.bias.mul_(ffn_hd_coeffs)
#
#         try:
#             print(
#                 f"[FFN SFLI] {name}: W mean={lin.weight.mean().item():.3e}, var={lin.weight.var(unbiased=False).item():.3e}, "
#                 f"b mean={lin.bias.mean().item():.3e}, var={lin.bias.var(unbiased=False).item():.3e}"
#             )
#         except Exception:
#             pass
#
# def init_ffn_with_custom_gaussian_and_bias_ugd(mlp0, model, args, l=0, L=1):
#     """
#     对 layer0.mlp 的 gate_proj 和 up_proj 做 SFLI初始化：
#       - W ~ N(0, std^2) * ffn_hd_coeffs
#       - b = 等差(arith) + 噪声(noise)，再 * ffn_hd_coeffs
#     """
#
#     # ---- args（优先 ffn_*，否则 fallback 到 v_* / v_hd_coeffs）----
#     ffn_base_std = getattr(args, "ffn_base_std", None)
#     ffn_bias_start = getattr(args, "ffn_bias_start", getattr(args, "v_bias_start", 0.0))
#     ffn_bias_step = getattr(args, "ffn_bias_step", getattr(args, "v_bias_step", 0.0))
#     ffn_bias_noise_std = getattr(args, "ffn_bias_noise_std", getattr(args, "v_bias_noise_std", 0.0))
#     ffn_hd_coeffs = getattr(args, "ffn_hd_coeffs", getattr(args, "v_hd_coeffs", 1.0))
#
#     for name in ["gate_proj", "up_proj", "down_proj"]:
#         if not hasattr(mlp0, name):
#             raise RuntimeError(f"LlamaMLP has no {name}; cannot run gaussian_FFN_SFLI")
#
#         lin = getattr(mlp0, name)  # Linear(768 -> 2048, bias=False) by default
#
#         # ---- 确保 bias=True ----
#         if lin.bias is None:
#             lin = _replace_linear_with_bias(lin)
#             setattr(mlp0, name, lin)
#
#         # ---- std：若没给 ffn_base_std，走 initializer_range/sqrt(fan_in) ----
#         if ffn_base_std is not None:
#             std = float(ffn_base_std)
#         else:
#             fan_in = lin.in_features
#             init_range = getattr(model.config, "initializer_range", 0.02)
#             std = float(init_range / (fan_in ** 0.5))
#
#         with torch.no_grad():
#             lin.weight.normal_(mean=0.0, std=std)
#             lin.weight.div_(lin.weight.norm(dim=1, keepdim=True).clamp_min(1e-12))
#             lin.weight.mul_((1 - l / L) + 1e-3)
#
#             out_dim = lin.out_features  # 2048
#             arith = ffn_bias_start + ffn_bias_step * torch.arange(
#                 out_dim, device=lin.weight.device, dtype=lin.weight.dtype
#             )
#             noise = torch.randn(out_dim, device=lin.weight.device, dtype=lin.weight.dtype) * ffn_bias_noise_std
#             lin.bias.copy_(arith + noise)
#             lin.bias.mul_(ffn_hd_coeffs)
#
#         try:
#             print(
#                 f"[FFN SFLI] {name}: W mean={lin.weight.mean().item():.3e}, var={lin.weight.var(unbiased=False).item():.3e}, "
#                 f"b mean={lin.bias.mean().item():.3e}, var={lin.bias.var(unbiased=False).item():.3e}"
#             )
#         except Exception:
#             pass
#
# def init_ffn_with_custom_gaussian_and_bias_u(mlp0, model, args, l=0, L=1):
#     """
#     对 layer0.mlp 的 gate_proj 和 up_proj 做 SFLI初始化：
#       - W ~ N(0, std^2) * ffn_hd_coeffs
#       - b = 等差(arith) + 噪声(noise)，再 * ffn_hd_coeffs
#     """
#
#     # ---- args（优先 ffn_*，否则 fallback 到 v_* / v_hd_coeffs）----
#     ffn_base_std = getattr(args, "ffn_base_std", None)
#     ffn_bias_start = getattr(args, "ffn_bias_start", getattr(args, "v_bias_start", 0.0))
#     ffn_bias_step = getattr(args, "ffn_bias_step", getattr(args, "v_bias_step", 0.0))
#     ffn_bias_noise_std = getattr(args, "ffn_bias_noise_std", getattr(args, "v_bias_noise_std", 0.0))
#     ffn_hd_coeffs = getattr(args, "ffn_hd_coeffs", getattr(args, "v_hd_coeffs", 1.0))
#
#     for name in ["up_proj"]:
#         if not hasattr(mlp0, name):
#             raise RuntimeError(f"LlamaMLP has no {name}; cannot run gaussian_FFN_SFLI")
#
#         lin = getattr(mlp0, name)  # Linear(768 -> 2048, bias=False) by default
#
#         # ---- 确保 bias=True ----
#         if lin.bias is None:
#             lin = _replace_linear_with_bias(lin)
#             setattr(mlp0, name, lin)
#
#         # ---- std：若没给 ffn_base_std，走 initializer_range/sqrt(fan_in) ----
#         if ffn_base_std is not None:
#             std = float(ffn_base_std)
#         else:
#             fan_in = lin.in_features
#             init_range = getattr(model.config, "initializer_range", 0.02)
#             std = float(init_range / (fan_in ** 0.5))
#
#         with torch.no_grad():
#             lin.weight.normal_(mean=0.0, std=std)
#             lin.weight.div_(lin.weight.norm(dim=1, keepdim=True).clamp_min(1e-12))
#             lin.weight.mul_((1 - l / L) + 1e-3)
#
#             out_dim = lin.out_features  # 2048
#             arith = ffn_bias_start + ffn_bias_step * torch.arange(
#                 out_dim, device=lin.weight.device, dtype=lin.weight.dtype
#             )
#             noise = torch.randn(out_dim, device=lin.weight.device, dtype=lin.weight.dtype) * ffn_bias_noise_std
#             lin.bias.copy_(arith + noise)
#             lin.bias.mul_(ffn_hd_coeffs)
#
#         try:
#             print(
#                 f"[FFN SFLI] {name}: W mean={lin.weight.mean().item():.3e}, var={lin.weight.var(unbiased=False).item():.3e}, "
#                 f"b mean={lin.bias.mean().item():.3e}, var={lin.bias.var(unbiased=False).item():.3e}"
#             )
#         except Exception:
#             pass
#
# def init_ffn_with_custom_gaussian_and_bias_g(mlp0, model, args, l=0, L=1):
#     """
#     对 layer0.mlp 的 gate_proj 和 up_proj 做 SFLI初始化：
#       - W ~ N(0, std^2) * ffn_hd_coeffs
#       - b = 等差(arith) + 噪声(noise)，再 * ffn_hd_coeffs
#     """
#
#     # ---- args（优先 ffn_*，否则 fallback 到 v_* / v_hd_coeffs）----
#     ffn_base_std = getattr(args, "ffn_base_std", None)
#     ffn_bias_start = getattr(args, "ffn_bias_start", getattr(args, "v_bias_start", 0.0))
#     ffn_bias_step = getattr(args, "ffn_bias_step", getattr(args, "v_bias_step", 0.0))
#     ffn_bias_noise_std = getattr(args, "ffn_bias_noise_std", getattr(args, "v_bias_noise_std", 0.0))
#     ffn_hd_coeffs = getattr(args, "ffn_hd_coeffs", getattr(args, "v_hd_coeffs", 1.0))
#
#     for name in ["gate_proj"]:
#         if not hasattr(mlp0, name):
#             raise RuntimeError(f"LlamaMLP has no {name}; cannot run gaussian_FFN_SFLI")
#
#         lin = getattr(mlp0, name)  # Linear(768 -> 2048, bias=False) by default
#
#         # ---- 确保 bias=True ----
#         if lin.bias is None:
#             lin = _replace_linear_with_bias(lin)
#             setattr(mlp0, name, lin)
#
#         # ---- std：若没给 ffn_base_std，走 initializer_range/sqrt(fan_in) ----
#         if ffn_base_std is not None:
#             std = float(ffn_base_std)
#         else:
#             fan_in = lin.in_features
#             init_range = getattr(model.config, "initializer_range", 0.02)
#             std = float(init_range / (fan_in ** 0.5))
#
#         with torch.no_grad():
#             lin.weight.normal_(mean=0.0, std=std)
#             lin.weight.div_(lin.weight.norm(dim=1, keepdim=True).clamp_min(1e-12))
#             lin.weight.mul_((1 - l / L) + 1e-3)
#
#             out_dim = lin.out_features  # 2048
#             arith = ffn_bias_start + ffn_bias_step * torch.arange(
#                 out_dim, device=lin.weight.device, dtype=lin.weight.dtype
#             )
#             noise = torch.randn(out_dim, device=lin.weight.device, dtype=lin.weight.dtype) * ffn_bias_noise_std
#             lin.bias.copy_(arith + noise)
#             lin.bias.mul_(ffn_hd_coeffs)
#
#         try:
#             print(
#                 f"[FFN SFLI] {name}: W mean={lin.weight.mean().item():.3e}, var={lin.weight.var(unbiased=False).item():.3e}, "
#                 f"b mean={lin.bias.mean().item():.3e}, var={lin.bias.var(unbiased=False).item():.3e}"
#             )
#         except Exception:
#             pass
#
# def init_ffn_with_custom_gaussian_and_bias_d(mlp0, model, args, l=0, L=1):
#     """
#     对 layer0.mlp 的 gate_proj 和 up_proj 做 SFLI初始化：
#       - W ~ N(0, std^2) * ffn_hd_coeffs
#       - b = 等差(arith) + 噪声(noise)，再 * ffn_hd_coeffs
#     """
#
#     # ---- args（优先 ffn_*，否则 fallback 到 v_* / v_hd_coeffs）----
#     ffn_base_std = getattr(args, "ffn_base_std", None)
#     ffn_bias_start = getattr(args, "ffn_bias_start", getattr(args, "v_bias_start", 0.0))
#     ffn_bias_step = getattr(args, "ffn_bias_step", getattr(args, "v_bias_step", 0.0))
#     ffn_bias_noise_std = getattr(args, "ffn_bias_noise_std", getattr(args, "v_bias_noise_std", 0.0))
#     ffn_hd_coeffs = getattr(args, "ffn_hd_coeffs", getattr(args, "v_hd_coeffs", 1.0))
#
#     for name in ["down_proj"]:
#         if not hasattr(mlp0, name):
#             raise RuntimeError(f"LlamaMLP has no {name}; cannot run gaussian_FFN_SFLI")
#
#         lin = getattr(mlp0, name)  # Linear(768 -> 2048, bias=False) by default
#
#         # ---- 确保 bias=True ----
#         if lin.bias is None:
#             lin = _replace_linear_with_bias(lin)
#             setattr(mlp0, name, lin)
#
#         # ---- std：若没给 ffn_base_std，走 initializer_range/sqrt(fan_in) ----
#         if ffn_base_std is not None:
#             std = float(ffn_base_std)
#         else:
#             fan_in = lin.in_features
#             init_range = getattr(model.config, "initializer_range", 0.02)
#             std = float(init_range / (fan_in ** 0.5))
#
#         with torch.no_grad():
#             lin.weight.normal_(mean=0.0, std=std)
#             lin.weight.div_(lin.weight.norm(dim=1, keepdim=True).clamp_min(1e-12))
#             lin.weight.mul_((1 - l / L) + 1e-3)
#
#             out_dim = lin.out_features  # 2048
#             arith = ffn_bias_start + ffn_bias_step * torch.arange(
#                 out_dim, device=lin.weight.device, dtype=lin.weight.dtype
#             )
#             noise = torch.randn(out_dim, device=lin.weight.device, dtype=lin.weight.dtype) * ffn_bias_noise_std
#             lin.bias.copy_(arith + noise)
#             lin.bias.mul_(ffn_hd_coeffs)
#
#         try:
#             print(
#                 f"[FFN SFLI] {name}: W mean={lin.weight.mean().item():.3e}, var={lin.weight.var(unbiased=False).item():.3e}, "
#                 f"b mean={lin.bias.mean().item():.3e}, var={lin.bias.var(unbiased=False).item():.3e}"
#             )
#         except Exception:
#             pass
#
# @torch.no_grad()
# def xavier_init_all_ffn_gate_up(model):
#     """
#     Apply Xavier uniform init to gate_proj and up_proj weights for ALL layers' MLPs.
#     (Does NOT touch embedding, attention, down_proj, etc.)
#     """
#     for i, mlp in enumerate(get_all_layer_mlp(model)):
#         if not (hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj")):
#             raise RuntimeError(f"Layer {i} mlp missing gate_proj/up_proj")
#
#         torch.nn.init.xavier_uniform_(mlp.gate_proj.weight)
#         torch.nn.init.xavier_uniform_(mlp.up_proj.weight)
#
# @torch.no_grad()
# def kaiming_init_all_ffn_gate_up(model):
#     """
#     Apply kaiming init to gate_proj and up_proj weights for ALL layers' MLPs.
#     (Does NOT touch embedding, attention, down_proj, etc.)
#     """
#     for i, mlp in enumerate(get_all_layer_mlp(model)):
#         if not (hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj")):
#             raise RuntimeError(f"Layer {i} mlp missing gate_proj/up_proj")
#
#         torch.nn.init.kaiming_uniform_(mlp.gate_proj.weight)
#         torch.nn.init.kaiming_uniform_(mlp.up_proj.weight)
#
# # def _add_activation_to_single_ud_proj(mlp_module, proj_name, args):
# #     act_name = getattr(args, "ffn_act", "none")
# #     if act_name == "none":
# #         return
# #
# #     proj = getattr(mlp_module, proj_name)
# #     if isinstance(proj, nn.Sequential):
# #         return
# #
# #     setattr(
# #         mlp_module,
# #         proj_name,
# #         nn.Sequential(
# #             proj,
# #             _make_act(act_name),
# #         ),
# #     )
# #
# #
# # def _add_activation_to_ud_subset(mlp_module, args, proj_names):
# #     for proj_name in proj_names:
# #         _add_activation_to_single_ud_proj(mlp_module, proj_name, args)
# #
# #
# # def add_ud_activation_u(mlp_module, args):
# #     _add_activation_to_ud_subset(mlp_module, args, ["up_proj"])
# #
# #
# # def add_ud_activation_d(mlp_module, args):
# #     _add_activation_to_ud_subset(mlp_module, args, ["down_proj"])
# #
# #
# # def add_ud_activation_ud(mlp_module, args):
# #     _add_activation_to_ud_subset(mlp_module, args, ["up_proj", "down_proj"])
#
