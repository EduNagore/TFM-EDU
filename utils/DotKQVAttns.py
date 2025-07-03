"""
Use the easy-attention mechanism for time-series prediction in latent space 
We use a stack of encoders only for the prediction, leveraging time-delay embedding 

@author yuningw and roger arnau
"""


import  torch 
from    torch               import nn 

from    utils.NNs.layers    import PositionWiseFeedForward
from    utils.NNs.Embedding import *


class DenseDotAttn(nn.Module):
    
    def __init__(self,
                 d_model,
                 seqLen, 
                 num_head,
                 d_QK,
                 d_V
                 ) -> None:
        """
        (Dense and Scaled) dot product attention mechansim
        used in transformer model for the time-series prediction and reconstruction
        
        Args:
            d_model         :   The embedding dimension for the input tensor
            seqLen          :   The length of the sequence (not used !)
            num_heads       :   The number of head to be used for multi-head attention
            d_QK            :   Dimension of Querys and Keys
            d_V             :   Dimension of Values
        """
        super(DenseDotAttn, self).__init__()
     
        self.d_model     =   d_model
        self.seqLen      =   seqLen
        self.num_heads   =   num_head
        self.d_QK        =   d_QK
        self.d_V         =   d_V
        
        assert d_model / num_head == d_V, "d_V must be d_model / num_head (join mechanism)"

        self.W_Q    = nn.Parameter(torch.randn(size = ( num_head, d_model, d_QK ),
                                                dtype = torch.float), requires_grad = True)
        nn.init.xavier_uniform_(self.W_Q)
        self.W_K    = nn.Parameter(torch.randn(size = ( num_head, d_model, d_QK ),
                                                dtype = torch.float), requires_grad = True)
        nn.init.xavier_uniform_(self.W_K)

        self.W_V        = nn.Parameter(torch.randn(size = ( num_head, d_model, d_V ),
                                                   dtype = torch.float), requires_grad = True)
        nn.init.xavier_uniform_(self.W_V)

        self.soft_max   = nn.functional.softmax
         

    def forward(self,x:torch.Tensor):   
        """
        Forward prop for the easyattention module 
        Following the expression:  x_hat    =   Alpha @ x @ Wv
        Args:  
            self    :   The self objects
            x       :   A tensor of Input data
        Returns:
            x       :   The tensor be encoded by the moudle
        """

        # B = batch size, M = seqLen, N = d_model
        B,M,N   =   x.shape
        H       =   self.num_heads

        # Convert dims
        # x: [B, M, N] -> [B, H, M, N] (copy)
        # Q, K: [B, H, M, d_QK]
        # V: [B, H, M, d_V]
        x       =   x.repeat( H , 1, 1, 1 ).transpose( 0, 1 )
        
        Q       =   x @ self.W_Q
        K       =   x @ self.W_K
        V       =   x @ self.W_V

        # Scaled Dot Product Attention
        attn    =   Q @ K.transpose( 2, 3 ) / self.d_QK**0.5
        attn    =   self.soft_max( attn, dim = 3 )
        x       =   attn @ V

        # Join x dim x: [B, H, M, d_V] -> [B, M, N] (obs: N * d_V = N)
        x       =   x.transpose( 1, 2 ).reshape( B, M, N )
        
        return x
