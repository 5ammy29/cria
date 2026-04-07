import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class Config:
    d_model: int = 256
    layers: int = 6
    heads: int = 8
    kv_heads: int = 8
    vocab_size: int = 1000
    eps: float = 1e-5
    seq_len: int = 128

class InputEmbedding(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

    def forward(self, x):
        return self.embedding(x)
    
    # embedding = nn.Embedding(10, 6)
    # i |             embd              |
    # - - - - - - - - - - - - - - - - - -
    # 0 | [v01, v02, v03, v04, v05, v06]|
    # 1 | [v11, v12, v13, v14, v15, v16]|
    # 2 | [v21, v22, v23, v24, v25, v26]|
    # 3 | [v31, v32, v33, v34, v35, v36]|
    # 4 | [v41, v42, v43, v44, v45, v46]|
    # 5 | [v51, v52, v53, v54, v55, v56]|
    # 6 | [v61, v62, v63, v64, v65, v66]|
    # 7 | [v71, v72, v73, v74, v75, v76]|
    # 8 | [v81, v82, v83, v84, v85, v86]|
    # 9 | [v91, v92, v93, v94, v95, v96]|
    #
    # X = [
    #   [9, 6, 3, 5],
    #   [1, 0, 3, 6]
    # ] (batch = 2, seq_len = 4)
    #
    # forward(X) (batch = 2, seq_len = 4, d_model = 6)
    # [
    #   [ # seq 1
    #     e[9] = [v91, v92, v93, v94, v95, v96],
    #     e[6] = [v61, v62, v63, v64, v65, v66],
    #     e[3] = [v31, v32, v33, v34, v35, v36],
    #     e[5] = [v51, v52, v53, v54, v55, v56]
    #   ],
    #   [ # seq 2
    #     e[1] = [v11, v12, v13, v14, v15, v16],
    #     e[0] = [v01, v02, v03, v04, v05, v06],
    #     e[3] = [v31, v32, v33, v34, v35, v36],
    #     e[6] = [v61, v62, v63, v64, v65, v66]
    #   ]
    # ]

class RMSNorm(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weights = nn.Parameter(torch.ones(config.d_model))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return self.weights * x

    # e[0] = [v01, v02, v03, v04, v05, v06] -> norm(e[0]) = [v01', v02', v03', v04', v05', v06']
    # e[1] = [v11, v12, v13, v14, v15, v16] -> norm(e[1]) ≈ [v11', v12', v13', v14', v15', v16']
    # .
    # .
    # .
    #
    # weights = [w1, w2, w3, w4, w5, w6]
    #
    # e'[0] =
    # [
    #   w1 * v01',
    #   w2 * v02',
    #   w3 * v03',
    #   w4 * v04',
    #   w5 * v05',
    #   w6 * v06'
    # ]
    # e'[1] =
    # [
    #   w1 * v11',
    #   w2 * v12',
    #   w3 * v13',
    #   w4 * v14',
    #   w5 * v15',
    #   w6 * v16'
    # ]
    # .
    # .
    # .

class RoPE(nn.Module):

    def __init__(self, d_head, seq_len, theta_base=10000.0):
        super().__init__()
        self.d_head = d_head
        self.pairs = self.d_head // 2     
        i_pairs = torch.arange(self.pairs, dtype=torch.float32)
        theta = 1.0 / (theta_base ** (i_pairs / self.pairs))     
        positions = torch.arange(seq_len, dtype=torch.float32)    
        phi = torch.outer(positions, theta)
        # positions = [0, 1, 2]
        # theta = [1.0, 0.0464, 0.00215]
        # phi =
        # [
        #  [0*1.0, 0*0.0464, 0*0.00215],
        #  [1*1.0, 1*0.0464, 1*0.00215],
        #  [2*1.0, 2*0.0464, 2*0.00215]
        # ]
        self.register_buffer("cos_phi", torch.cos(phi), persistent=False)
        self.register_buffer("sin_phi", torch.sin(phi), persistent=False)

    def forward(self, x):
        seq_len = x.size(1)
        cos = self.cos_phi[:seq_len].unsqueeze(0).unsqueeze(2)
        sin = self.sin_phi[:seq_len].unsqueeze(0).unsqueeze(2)
        x_even = x[..., ::2]
        x_odd  = x[..., 1::2]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd  = x_even * sin + x_odd * cos
        x_rotated = torch.stack((rotated_even, rotated_odd), dim=-1)
        return x_rotated.flatten(-2)
    
    # [
    # ["I", "love", "you"],
    # ["Sky", "is", "blue"]
    # ]
    #
    # [
    #   [
    #     [1, 2, 3, 4, 5, 6],
    #     [2, 1, 0, 1, 2, 3],
    #     [3, 3, 3, 3, 3, 3]
    #   ],
    #   [
    #     [1, 0, 1, 0, 1, 0],
    #     [0, 1, 0, 1, 0, 1],
    #     [2, 2, 2, 2, 2, 2]
    #   ]
    # ]
    #
    # pairs = [(x1, x2), (x3, x4), (x5, x6)]
    #
    # for pair_i, theta_i = 1 / (10000^((2 * i)/d_head)) 
    # theta0 = 1
    # theta1 = 0.0464
    # theta2 = 0.00215
    #
    # [
    #   [1, 2, 3, 4, 5, 6] position = 0
    #   phi = position * theta_i
    #   (x, y) -> (x * cos(phi) - y * sin(phi), x * sin(phi) + y * cos(phi))
    #   | pair0 | phi = 0 * 1 = 0       | (1, 2) -> (1, 2) |
    #   | pair1 | phi = 0 * 0.0464 = 0  | (3, 4) -> (3, 4) |
    #   | pair2 | phi = 0 * 0.00215 = 0 | (5, 6) -> (5, 6) |
    #   [1, 2, 3, 4, 5, 6]
    #   - - - - - - - - - - - - - - - - - - - - - - - - - - 
    #   [2, 1, 0, 1, 2, 3] position = 1
    #   | pair0 | phi = 1 * 1 = 0             | (2, 1) -> (0.2391, 2.2233)  |
    #   | pair1 | phi = 1 * 0.0464 = 0.0464   | (0, 1) -> (-0.0464, 0.9989) |
    #   | pair2 | phi = 1 * 0.00215 = 0.00215 | (2, 3) -> (1.9935, 3.0043)  |
    #   [0.2391, 2.2233, -0.0464, 0.9989, 1.9935, 3.0043]
    #   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #   [3, 3, 3, 3, 3, 3] position = 2
    #   | pair0 | phi = 2 * 1 = 2            | (3, 3) -> (-3.9762, 1.4796) |
    #   | pair1 | phi = 2 * 0.0464 = 0.0928  | (3, 3) -> (2.709, 3.2652)   |
    #   | pair2 | phi = 2 * 0.00215 = 0.0043 | (3, 3) -> (2.9871, 3.0129)  |
    #   [-3.9762, 1.4796, 2.709, 3.2652, 2.9871, 3.0129]
    #   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    #
    #   [1, 0, 1, 0, 1, 0] position = 0
    #   | pair0 | phi = 0 * 1 = 0       | (1, 0) -> (1, 0) |
    #   | pair1 | phi = 0 * 0.0464 = 0  | (1, 0) -> (1, 0) |
    #   | pair2 | phi = 0 * 0.00215 = 0 | (1, 0) -> (1, 0) |
    #   [1, 0, 1, 0, 1, 0]
    #   - - - - - - - - - - - - - - - - - - - - - - - - - - 
    #   [0, 1, 0, 1, 0, 1] position = 1
    #   | pair0 | phi = 1 * 1 = 1             | (0, 1) -> (-0.8415, 0.5403)    |
    #   | pair1 | phi = 1 * 0.0464 = 0.0464   | (0, 1) -> (-0.0464, 0.9989)    |
    #   | pair2 | phi = 1 * 0.00215 = 0.00215 | (0, 1) -> (-0.00215, 0.999998) |
    #   [-0.8415, 0.5403, -0.0464, 0.9989, -0.00215, 0.999998]
    #   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    #   [3, 3, 3, 3, 3, 3] position = 2
    #   | pair0 | phi = 2 * 1 = 2            | (3, 3) -> (-3.9762, 1.4796) |
    #   | pair1 | phi = 2 * 0.0464 = 0.0928  | (3, 3) -> (2.7090, 3.2652)  |
    #   | pair2 | phi = 2 * 0.00215 = 0.0043 | (3, 3) -> (2.9871, 3.0129)  |
    #   [-3.9762, 1.4796, 2.7090, 3.2652, 2.9871, 3.0129]
    #   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # ]    
