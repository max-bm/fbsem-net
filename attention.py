"""
Author: Maxwell Buckmire-Monro
maxwell.monro@kcl.ac.uk
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """
    Splits input image into patches (tokens) and embeds them into 'arbitrary' dimension.

    Parameters
    ----------
    img_size: int
        Size of the (square) image.
    
    patch_size: int
        Size of the (square) patches.

    in_channels: int
        Number of input channels. Greyscale = 1.

    embedding_dim: int
        Size of arbitrary embedding dimension.

    Attributes
    ----------
    n_patches: int
        Number of patches.

    embedding_proj: nn.Conv2d
        Convolutional layer that performs splitting and embedding.
    """
    def __init__(self, img_size=144, patch_size=4, in_channels=1, embedding_dim=96):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.embedding_proj = nn.Conv2d(
            in_channels, 
            embedding_dim, 
            kernel_size=patch_size, 
            stride=patch_size)

    def forward(self, img):
        """
        Forward pass.

        Parameters
        ----------
        img: torch.Tensor
            Input image to be split into embedded patches.
            Shape: (batch_size, in_channels, img_size, img_size)

        Returns
        -------
        torch.Tensor
            Embedded patches.
            Shape: (batch_size, n_patches, embedding_dim)
        """
        out = self.embedding_proj(img) # (batch_size, embedding_dim, n_patches ** 0.5, n_patches ** 0.5)
        out = out.flatten(2) # (batch_size, embedding_dim, n_patches)
        out = out.transpose(1, 2) # (batch_size, n_patches, embedding_dim)

        return out


class MSA(nn.Module):
    """
    Multi-headed self-attention mechanism.

    Parameters
    ----------
    embedding_dim: int
        Embedding dimension of 'tokens'.

    n_heads: int
        Number of attention heads.

    qkv_bias: bool
        Toggle for inclusion of bias in query, key and value projection.

    attn_drop: float
        Dropout probability of query, key and value tensors.

    proj_drop: float
        Dropout probability of output.

    Attributes
    ----------
    qk_scale: float
        Normalising constant for qk dot product.

    qkv: nn.Linear
        Linear projection for query, key and value tensors.

    proj: nn.Linear
        Linear mapping for mapping concatenated outputs of attentions heads into new space.

    attn_drop, proj_drop: nn.Dropout
        Dropout layers.
    """
    def __init__(self, embedding_dim, n_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super(MSA, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads # Defined this way so that concatenated has same dim as input.
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embedding_dim, embedding_dim*3, bias=qkv_bias) # Could be written as 3 separate mappings: q, k and v.
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, tkn):
        """
        Forward pass.

        Parameters
        ----------
        tkn: torch.Tensor
            Input embedded token.
            Shape: (batch_size, n_tkns, embedding_dim) (No +1 because no patch token)

        Returns
        -------
        torch.Tensor
            Embedded concatenation of attention head outputs.
            Shape: (batch_size, n_tkns, embedding_dim)
        """
        batch_size, n_tkns, embedding_dim = tkn.shape
        qkv = self.qkv(tkn) # (batch_size, n_tkns, 3 * embedding_dim)
        qkv = qkv.reshape(batch_size, n_tkns, 3, self.n_heads, self.head_dim) # (batch_size, n_tkns, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, batch_size, n_heads, n_tkns, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale # (batch_size, n_heads, n_tkns, n_tkns)
        attn = self.softmax(attn) # (batch_size, n_heads, n_tkns, n_tkns)
        attn = self.attn_drop(attn) # Attention dropout

        out = attn @ v # (batch_size, n_heads, n_tkns, head_dim)
        out = out.transpose(1, 2) # (batch_size, n_tkns, n_heads, head_dim)
        out = out.reshape(batch_size, n_tkns, embedding_dim) # (batch_size, n_tkns, embedding dim)
        out = self.proj(out) # (batch_size, n_tkns, embedding_dim)
        out = self.proj_drop(out) # Projection dropout)

        return out


class MLP(nn.Module):
    """
    Multilayer perceptron for Transformer block.

    Parameters
    ----------
    in_features: int
        Number of input features.
    
    hidden_features: int
        Number of nodes in the hidden layer.

    out_features: int
        Number of output features.

    drop: float
        Dropout probability.
    Attributes
    ----------
    fc1, fc2: nn.Linear
        Fully connected layers.
    act: nn.GELU
        GELU activation function.
    drop: nn.Drouput
        Dropout layer.
    """
    def __init__(self, in_features, hidden_featues=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(MLP, self).__init__()
        hidden_features = hidden_featues or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_featues)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_featues, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Forward pass of MLP.

        Parameteres
        -----------
        x: torch.Tensor
            Input tensor.
            Shape: (batch_size, n_tkns, in_features).

        Returns
        -------
        torch.Tensor
            Output tensor.
            Shape: (batch_size, n_tkns, out_features).
        """
        x = self.fc1(x) # (batch_size, n_tkns, hidden_features)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x) # (batch_size, n_tkns, out_features)
        x = self.drop(x)
        return x


