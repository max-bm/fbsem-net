"""
Author: Maxwell Buckmire-Monro
maxwell.monro@kcl.ac.uk
"""

import torch
from torch._C import Value
import torch.nn as nn
import torch.nn.functional as F

class SimplePatching(nn.Module):
    """
    Splits input image into patches (tokens) with no embedding.

    Parameters
    ----------
    img_size: int
        Size of the (square) image.

    patch_size: int
        Size of the (square) patches.

    in_channels: int
        Number of input channels. Greyscale = 1.

    Attributes
    ----------
    n_patches: int
        Number of patches.
    """
    def __init__(self, img_size=144, patch_size=4):
        super(SimplePatching, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = 1 # Simple patching requires only PET data inputs
        self.n_patches = (img_size // patch_size) ** 2

    def forward(self, img):
        """
        Forward pass.

        Parameters
        ----------
        img: torch.Tensor
            Input image to be split into simple patches.
            Shape: (batch_size, in_channels, img_size, img_size).

        Returns
        -------
        torch.Tensor
            Non-embedded patches.
            Shape: (batch_size, n_patches, patch_size**2).
        """
        batch_size = img.shape[0]
        return img.unfold(3, self.patch_size, self.patch_size).unfold(4,
            self.patch_size, self.patch_size).reshape(batch_size, -1, self.patch_size**2)


class LearnedPatching(nn.Module):
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
    def __init__(self, img_size=144, patch_size=16, in_channels=1, embedding_dim=256):
        super(LearnedPatching, self).__init__()
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
            Shape: (batch_size, in_channels, img_size, img_size).

        Returns
        -------
        torch.Tensor
            Embedded patches.
            Shape: (batch_size, n_patches, embedding_dim).
        """
        out = self.embedding_proj(img) # (batch_size, embedding_dim, n_patches ** 0.5, n_patches ** 0.5)
        out = out.flatten(2) # (batch_size, embedding_dim, n_patches)
        out = out.transpose(1, 2) # (batch_size, n_patches, embedding_dim)
        return out


class SimpleMerging(nn.Module):
    """
    Merges unembedded patches/tokens into single image.

    Parameters
    ----------
    img_size: int
        Size of the (square) image.

    patch_size: int
        Size of the (square) patches.

    in_channels: int
        Number of input channels. Greyscale = 1.
    """
    def __init__(self, img_size=144, patch_size=4, in_channels=1):
        super(SimpleMerging, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.n_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Image tokens to merge into image.
            Shape: (batch_size, n_patches, embedding_dim).

        Returns
        -------
        torch.Tensor
            Reconstructed image.
            Shape: (batch_size, in_channels, img_size, img_size).
        """
        batch_size = x.shape[0]
        x = x.contiguous().view(batch_size, self.in_channels, -1, self.patch_size * self.patch_size)
        x = x.permute(0, 1, 3, 2)
        x = x.contiguous().view(batch_size, self.in_channels * self.patch_size * self.patch_size, -1)
        x = F.fold(x, output_size=(self.img_size, self.img_size), kernel_size=self.patch_size, stride=self.patch_size)
        return x


class LearnedMerging(nn.Module):
    """
    Merges separate image patches/tokens into single image.

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
    unembedding_proj: nn.Linear
        Projection from token space to patch space.

    """
    def __init__(self, img_size=144, patch_size=16, in_channels=1, embedding_dim=256):
        super(LearnedMerging, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.n_patches = (img_size // patch_size) ** 2
        self.unembedding_proj = nn.Linear(embedding_dim, in_channels * patch_size**2)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Embedded image tokens to merge into image.
            Shape: (batch_size, n_patches, embedding_dim).

        Returns
        -------
        torch.Tensor
            Reconstructed image.
            Shape: (batch_size, in_channels, img_size, img_size).
        """
        batch_size = x.shape[0]
        x = self.unembedding_proj(x) # Unembed tokens. (batch_size, n_patches, in_channels * patch_size**2)
        x = x.contiguous().view(batch_size, self.in_channels, -1, self.patch_size * self.patch_size)
        x = x.permute(0, 1, 3, 2)
        x = x.contiguous().view(batch_size, self.in_channels * self.patch_size * self.patch_size, -1)
        x = F.fold(x, output_size=(self.img_size, self.img_size), kernel_size=self.patch_size, stride=self.patch_size)
        return x


class SimpleMSA(nn.Module):
    """
    Multi-headed self-attention mechanism with no learnable parameters.

    Parameters
    ----------
    n_heads: int
        Number of attention heads.

    Attributes
    ----------
    softmax: nn.Softmax
        Softmax layer for normalisation of attention weights.
    """
    def __init__(self, n_heads=1):
        super(SimpleMSA, self).__init__()
        self.n_heads = n_heads
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, tkn, scale):
        """
        Forward pass.

        Parameters
        ----------
        tkn: torch.Tensor
            Input image token.
            Shape: (batch_size, n_tkns, embedding_dim)

        Returns
        -------
        torch.Tensor
            Embedded concatenation of attention head outputs.
            Shape: (batch_size, n_tkns, embedding_dim)
        """
        batch_size, n_tkns, embedding_dim = tkn.shape
        head_dim = embedding_dim // self.n_heads
        tkn = tkn.reshape(batch_size, n_tkns, self.n_heads, head_dim)
        tkn = tkn.permute(0, 2, 1, 3) # (batch_size, n_heads, n_tkns, head_dim)
        q, k, v = tkn, tkn, tkn
        q = q.repeat(1, 1, n_tkns, 1).view(batch_size, 1, n_tkns, n_tkns, head_dim)
        k = k.repeat(1, 1, 1, n_tkns).view(batch_size, 1, n_tkns, n_tkns, head_dim)

        attn = torch.square(q - k).mean(dim=-1) / scale # (batch_size, n_heads, n_tkns, n_tkns)
        attn = self.softmax(-attn) # (batch_size, n_heads, n_tkns, n_tkns)
        # attn now acts as weights for simple weighted sum of similar pixels/patches.
        out = attn @ v # (batch_size, n_heads, n_tkns, head_dim)
        out = out.transpose(1, 2) # (batch_size, n_tkns, n_heads, head_dim)
        out = out.reshape(batch_size, n_tkns, embedding_dim) # (batch_size, n_tkns, embedding dim)
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
    head_dim: float
        Embedding dimension through heads so that concatenated outputs are correct shape.

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
        out = self.proj_drop(out) # Projection dropout
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
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(MLP, self).__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
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


class TransformerEncoder(nn.Module):
    """
    Transformer encorder block.

    Parameters
    ----------
    embedding_dim: int
        Embedding dimension of 'tokens'.

    n_heads: int
        Number of attention heads.

    mlp_ratio: float
        Determines size of hidden dimension of MLP w.r.t embedding dim.

    qkv_bias: bool
        Toggle for inlcusion of bias in query, key and value projections.

    drop, attn_drop: float
        Dropout probabilities.

    Attributes
    ----------
    norm1, norm2: nn,LayerNorm
        Layer normalisation.

    attn: MSA
        Multiheaded self-attention module.

    mlp: MLP
        Multilayer perceptron module.
    """
    def __init__(self, embedding_dim, n_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(embedding_dim, 1e-6)
        self.attn = MSA(embedding_dim, n_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(embedding_dim, 1e-6)
        self.mlp = MLP(embedding_dim, int(embedding_dim * mlp_ratio), embedding_dim)

    def forward(self, x):
        """
        Forward pass of Transformer encoder.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
            Shape: (batch_size, n_tkns, embedding_dim).

        Returns
        -------
        torch.Tensor
            Output tensor.
            Shape: (batch_size, n_tkns, embedding_dim).
        """
        x = x + self.attn(self.norm1(x)) # First residual 'block'
        x = x + self.mlp(self.norm2(x)) # Second residual 'block'
        return x


class SimpleVisionTransformer(nn.Module):
    """
    Simple Vision Transformer network architecture.

    Parameters
    ----------
    img_size: int
        Height and width of square input image.

    patch_size: int
        Height and width of square patch size.

    n_heads: int
        Number of attention heads per block.

    in_channels: int
        Number of input channels. Greyscale = 1.

    scale: bool
        Toggle for learned softmax scaling. Default = False.

    Attributes
    ----------
    patching: SimplePatching
        Layer to compute simple patching of image.

    msa: SimpleMSA
        Layer for simple multi-headed self attention.

    patch_merge: SimpleMerging
        Layer to merge patches into reconstructed image.

    scale: nn.Parameter
        Learned scaling parameter for softmax.
    """
    def __init__(self, img_size=144, patch_size=1, n_heads=1, in_channels=1, scale=False):
        super(SimpleVisionTransformer, self).__init__()
        self.patching = SimplePatching(img_size, patch_size)
        self.msa = SimpleMSA(n_heads=n_heads)
        self.patch_merge = SimpleMerging(img_size, patch_size, in_channels=in_channels)
        self.in_channels = in_channels
        if scale:
            self.scale = nn.Parameter(torch.rand(1), requires_grad=True)
        else:
            self.scale = 1.

    def forward(self, x, _):
        """
        Forward pass of image through Simple Vision Transformer.

        Parameters
        ----------
        x: torch.Tensor
            Input image.
            Shape: (batch_size, in_channels, img_size, img_size).

        Returns
        -------
        torch.Tensor
            'Reconstructed' image.
            Shape: (batch_size, in_channels, img_size, img_size).
        """
        x_max = x.max()
        # Scale the image
        x = x / x_max
        x = self.patching(x) # (batch_size, n_tkns, embedding_dim)
        x = self.msa(x, self.scale)
        x = self.patch_merge(x) # Marge patches back together
        x = x * x_max
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer network architecture.

    Parameters
    ----------
    img_size: int
        Height and width of square input image.

    patch_size: int
        Height and width of square patch size.

    in_channels: int
        Number of input channels. Greyscale = 1.

    embedding_dim: int
        Dimensionality of token/patch embeddings.

    depth: int
        Number of Transformer encoder blocks.

    n_heads: int
        Number of attention heads per block. (Could this vary from block to block?)

    mlp_ratio: float
        Determines size of hidden dimension of MLP w.r.t embedding dim.

    qkv_bias: bool
        Toggle for inlcusion of bias in query, key and value projections.

    drop, attn_drop: float
        Dropout probabilities.

    Attributes
    ----------
    patch_embed: PatchEmbedding
        Layer to compute patch embedding of image.

    pos_embed: nn.Parameter
        Learned positional embedding of patches.

    pos_drop: nn.Dropout
        Drouput layer for positional embedding.

    blocks: nn.ModuleList
        List of blocks.

    norm: nn.LayerNorm
        Layer normalisation.

    patch_unembed: PatchUnembedding
        Layer to merge patches into reconstructed image.
    """
    def __init__(self, img_size=144, patch_size=16, in_channels=1, embedding_dim=256, depth=12,
                 n_heads=12, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super(VisionTransformer, self).__init__()
        self.patch_embed = LearnedPatching(img_size, patch_size, in_channels, embedding_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embedding_dim))
        self.pos_drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList(
            [TransformerEncoder(embedding_dim, n_heads, mlp_ratio, qkv_bias, drop, attn_drop) for i in range(depth)])
        self.norm = nn.LayerNorm(embedding_dim, 1e-6)
        self.patch_merge = LearnedMerging(img_size, patch_size, in_channels, embedding_dim)
        self.in_channels = in_channels

    def forward(self, x, _):
        """
        Forward pass of image through Vision Transformer.

        Parameters
        ----------
        x: torch.Tensor
            Input image.
            Shape: (batch_size, in_channels, img_size, img_size).

        Returns
        -------
        torch.Tensor
            'Reconstructed' image.
            Shape: (batch_size, in_channels, img_size, img_size).
        """
        x = self.patch_embed(x) # (batch_size, n_tkns, embedding_dim)
        x = x + self.pos_embed # (batch_size, n_tkns, embedding_dim)
        x = self.pos_drop(x)
        shortcut = x # Residual connection around Transformer encoder blocks
        for block in self.blocks:
            x = block(x)

        x = x + shortcut
        x = self.norm(x) # LayerNorm not in 16x16 paper
        x = self.patch_merge(x) # Marge patches back together
        return x
