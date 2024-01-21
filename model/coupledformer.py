import math
import torch
import torch.nn as nn
from einops import rearrange

from model.relative_rpe import DynamicRelativePositionBias1D, HopRelativePositionBias
from model.Hyperformer import DropPath, import_class, bn_init


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CoupledAttention(nn.Module):
    def __init__(self, dim_in, dim, A, num_heads=6, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.,
                 num_points=25, num_frames=64,
                 relational_bias=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_frames = num_frames
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim_in, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Temporal dynamic relative
        self.temporal_rpb = DynamicRelativePositionBias1D(
            num_heads=num_heads,
            window_size=num_frames,
            mlp_dim=dim,
            num_points=num_points
        )

        # Hop relative positional embedding
        self.hop_rpb = HopRelativePositionBias(
            num_points=num_points,
            A=A,
            num_heads=num_heads,
            num_frames=num_frames,
            mlp_dim=dim
        )

        # relational bias, we choose to initialize it with ajacency matrix
        self.relational_bias = relational_bias
        if relational_bias:
            A = A.sum(0)
            A /= A.sum(axis=-1, keepdims=True)
            # A /= num_frames  # because we flatten it later in the code t v -> t * v
            self.outer = nn.Parameter(
                torch.stack([torch.from_numpy(A).float() for _ in range(num_heads)], dim=0),
                requires_grad=True)

            self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        N, T, C = x.shape  # T = t * v
        assert T == self.num_frames * self.num_points
        qkv = self.qkv(x).reshape(N, T, 3, self.num_heads, self.head_dim)
        qkv = rearrange(qkv, 'n t o h c -> o n h t c')

        q, k, v = qkv  # n h (t v) c

        attn = q @ k.transpose(-2, -1)  # n h (t v) (t v)

        # hop relative positional embedding
        attn_bias_hop = self.hop_rpb()

        # temporal relative positional bias
        attn_bias_temporal = self.temporal_rpb()

        attn = attn + attn_bias_hop + attn_bias_temporal

        attn = attn * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # relational bias
        if self.relational_bias:
            # Initialize a list to hold the attention bias matrices
            attention_biases = []

            # Loop over each slice (i.e., each 25x25 matrix) and create a 64*25x64*25 matrix
            for i in range(self.num_heads):
                # Extract the i-th 25x25 matrix
                single_learnable_matrix = self.outer[i]

                # Create a 64*25x64*25 matrix using block_diag or manual filling
                blocks = [single_learnable_matrix for _ in range(self.num_frames)]
                attention_bias = torch.block_diag(*blocks)

                # Add the created matrix to the list
                attention_biases.append(attention_bias)

            # Stack the individual attention bias matrices to get the final tensor
            attn_relational = torch.stack(attention_biases)

            x = (self.alpha * attn + attn_relational) @ v
        else:
            x = attn @ v

        x = rearrange(x, 'n h t c -> n t (h c)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class unit_coupled(nn.Module):
    def __init__(
            self, dim_in, dim, A, num_heads, qkv_bias=False, qk_scale=None,
            attn_drop=0, drop=0., drop_path=0., norm_layer=nn.LayerNorm,
            num_points=25, num_frames=64, use_mlp=True, ff_mult=4,
    ):
        super().__init__()
        self.norm = norm_layer(dim_in)
        self.norm1 = norm_layer(dim)
        self.attn = CoupledAttention(dim_in, dim, A, num_heads, qkv_bias,
                 qk_scale, attn_drop, proj_drop=drop,
                 num_points=num_points, num_frames=num_frames)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.skip_proj = nn.Linear(
            dim_in, dim, bias=False) if dim_in != dim else nn.Identity()

        self.use_mlp = use_mlp
        if use_mlp:
            self.mlp = Mlp(dim, dim * ff_mult, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, *args):
        if x.dim() == 4:
            x = rearrange(x, 'n c t v -> n (t v) c')

        x = self.skip_proj(x) + self.drop_path(self.attn(self.norm(x)))

        if self.use_mlp:
            x = x + self.drop_path(self.mlp(self.norm1(x)))

        return x


class CoupledFormer(nn.Module):
    def __init__(
            self, num_class=60, num_point=20, num_person=2, graph=None,
            graph_args=dict(), in_channels=3, drop_out=0, num_of_heads=9,
            head_dim=24, num_layers=8, joint_label=[], num_frames=64, **kwargs):
        super().__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25

        self.num_of_heads = num_of_heads
        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.joint_label = joint_label

        hidden_dim = num_of_heads * head_dim

        self.layers = nn.ModuleList([
            unit_coupled(
                3, hidden_dim, A, num_heads=num_of_heads,
                num_points=num_point, num_frames=num_frames)
        ])

        for i in range(1, num_layers):
            self.layers.append(
                unit_coupled(
                    hidden_dim, hidden_dim, A, num_heads=num_of_heads,
                    num_points=num_point, num_frames=num_frames)
            )

        # standard ce loss
        self.fc = nn.Linear(hidden_dim, num_class)

        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x, y):
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> n (m v c) t')
        x = self.data_bn(x)

        x = rearrange(x, 'n (m v c) t -> (n m) (t v) c', m=M, v=V, c=C)

        for layer in self.layers:
            x = layer(x)

        x = rearrange(x, '(n m) t c -> n m t c', n=N, m=M)
        x = x.mean(2).mean(1)
        x = self.fc(self.drop_out(x))

        return x, y