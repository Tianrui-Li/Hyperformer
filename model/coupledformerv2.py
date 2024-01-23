from torch import nn
from einops import rearrange


from model.Hyperformer import TCN_ViT_unit, import_class
from model.coupledformer import unit_coupled


class CoupledFormerV2(nn.Module):
    def __init__(
            self,
            *,
            num_class=60,
            # dims=(24*9, 64, 128, 256),
            # depths=(2, 2, 4, 2),
            # mhsa_types=('l', 'l', 'l', 'g'),
            dims=(24 * 9, 24 * 9, 24 * 9, 256),
            depths=(5, 3, 2, 2),
            mhsa_types=('l', 'l', 'l', 'g'),
            num_point=20,
            num_person=2,
            graph=None,
            graph_args=dict(),
            in_channels=3,
            ff_mult=4,
            # dim_head=64,
            dim_head=24,
            attn_head=64,
            drop=0,
            joint_label=[],
            num_frames=64,
            **kwargs,
    ):
        super().__init__()

        assert len(dims) == len(depths)

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25
        self.num_class = num_class

        frames = [num_frames // (2 ** i) for i in range(len(dims))]
        mhsa_types = tuple(map(lambda t: t.lower(), mhsa_types))

        self.layers = nn.ModuleList([])

        dim_in = in_channels
        for ind, (depth, mhsa_type) in enumerate(zip(depths, mhsa_types)):
            stage_dim = dims[ind]
            heads = stage_dim // dim_head
            Attn_heads = stage_dim // attn_head

            for j in range(1, depth+1):
                if mhsa_type == 'l':
                    Block = TCN_ViT_unit(
                        in_channels=dim_in,
                        out_channels=dims[ind],
                        A=A,
                        # stride=2 if j == depth else 1,
                        stride=2 if j == depth and ind != len(depths) - 2 else 1,
                        num_of_heads=heads,
                        num_point=num_point,
                    )
                elif mhsa_type == 'g':
                    Block = unit_coupled(
                        dim_in=dim_in,
                        dim=dims[ind],
                        A=A,
                        num_heads=Attn_heads,
                        num_frames=frames[ind]*2,
                        ff_mult=ff_mult,
                    )
                else:
                    raise ValueError('unknown mhsa_type')

                self.layers.append(Block)

                dim_in = dims[ind]

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.joint_label = joint_label

        self.fc = nn.Linear(dims[-1], num_class)
        if drop:
            self.drop_out = nn.Dropout(drop)
        else:
            self.drop_out = lambda x: x

    def forward(self, x, y):
        groups = []
        for num in range(max(self.joint_label) + 1):
            groups.append(
                [ind for ind, element in enumerate(self.joint_label) if
                 element == num])

        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> n (m v c) t')
        x = self.data_bn(x)

        x = rearrange(x, 'n (m v c) t -> (n m) c t v', m=M, v=V, c=C)

        for layer in self.layers:
            x = layer(x, self.joint_label, groups)

        if x.dim() == 3:
            x = rearrange(x, '(n m) t c -> n m t c', n=N, m=M)
            x = x.mean(2).mean(1)
        else:
            x = x.view(N, M, 24*9, -1)
            x = x.mean(3).mean(1)

        x = self.fc(self.drop_out(x))

        return x, y