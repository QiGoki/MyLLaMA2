from torch import nn


class MLP(nn.Module):
    def __init__(self, dim:int, hidden_dim:int, multiple_of:int, dropout:float):
        super().__init__()
        # 如果没有指定隐藏层的维度，我们将其设置为输入维度的4倍
        # 然后将其减少到2/3，最后确保它是multiple_of的倍数
        if hidden_dim is None:
            hidden_dim = dim * 4
            hidden_dim = int(2 * hidden_dim / 3)

