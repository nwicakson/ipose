from __future__ import absolute_import
from lib2to3.refactor import get_fixers_from_package

import torch.nn as nn
import torch
import numpy as np
import scipy.sparse as sp
import copy, math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models.ChebConv import ChebConv, _GraphConv, _ResChebGC
from models.GraFormer import *

### the embedding of diffusion timestep ###
def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

# Existing implementation
class _ResChebGC_diff(nn.Module):
    def __init__(self, adj, input_dim, output_dim, emd_dim, hid_dim, p_dropout):
        super(_ResChebGC_diff, self).__init__()
        self.adj = adj
        self.gconv1 = _GraphConv(input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(hid_dim, output_dim, p_dropout)
        ### time embedding ###
        self.temb_proj = torch.nn.Linear(emd_dim,hid_dim)

    def forward(self, x, temb):
        residual = x
        out = self.gconv1(x, self.adj)
        out = out + self.temb_proj(nonlinearity(temb))[:, None, :]
        out = self.gconv2(out, self.adj)
        return residual + out

# Memory-optimized ImplicitGCNAttentionBlock
class ImplicitGCNAttentionBlock(nn.Module):
    def __init__(self, adj, dim_model, attn, gcn, emd_dim, dropout=0.1, p_dropout=0.1):
        super(ImplicitGCNAttentionBlock, self).__init__()
        self.adj = adj
        self.attn = attn  # Multi-headed attention
        self.gcn = gcn    # Graph convolution
        self.layernorm1 = LayerNorm(dim_model)
        self.layernorm2 = LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)
        
        # Add time embedding projection
        self.temb_proj = torch.nn.Linear(emd_dim, dim_model)
        
        # Solver parameters
        self.tol = 1e-3  # Increased tolerance
        self.max_iter = 1  # Start with 1 iteration
        self.last_solution = None
        self.initial_z = None
        self.num_iterations = 0
        self.enable_warmstart = False
        
    def forward_iteration(self, z, x, mask, temb):
        """Single iteration combining attention and GCN"""
        # Apply attention with layernorm
        z_norm1 = self.layernorm1(z)
        attn_out = self.attn(z_norm1, z_norm1, z_norm1, mask)
        z = z + self.dropout(attn_out)
        
        # Apply GCN with layernorm
        z_norm2 = self.layernorm2(z)
        gcn_out = self.gcn(z_norm2)
        
        # Add time embedding
        if temb is not None:
            time_emb = self.temb_proj(nonlinearity(temb))[:, None, :]
            gcn_out = gcn_out + time_emb
            
        return z + self.dropout(gcn_out)
    
    def forward(self, x, mask, temb, t_fraction=None):
        # Dynamically adjust max iterations based on diffusion step if provided
        if t_fraction is not None:
            if t_fraction > 0.8:  # Early denoising (very noisy)
                self.max_iter = max(1, self.max_iter - 1)
            elif t_fraction < 0.2:  # Late denoising (refining)
                self.max_iter = self.max_iter + 1
                
        # Use input as initial guess
        z = x
            
        # Track convergence
        self.num_iterations = 0
        
        # Fixed point iteration without gradients
        with torch.no_grad():
            for i in range(self.max_iter):
                z_new = self.forward_iteration(z, x, mask, temb)
                residual = torch.norm(z_new - z) / (torch.norm(z) + 1e-6)
                
                if residual < self.tol:
                    z = z_new
                    self.num_iterations = i + 1
                    break
                    
                z = z_new
                self.num_iterations = i + 1
        
        # Final forward pass with gradients
        z = self.forward_iteration(z.detach(), x, mask, temb)
        
        # Store solution for potential reuse
        if self.enable_warmstart:
            self.last_solution = z.detach()
            
        return z

# Modified GCNdiff with memory-efficient implementation
class GCNdiff(nn.Module):
    def __init__(self, adj, config):
        super(GCNdiff, self).__init__()
        
        self.adj = adj
        self.config = config
        ### load gcn configuration ###
        con_gcn = config.model
        self.hid_dim, self.emd_dim, self.coords_dim, num_layers, n_head, dropout, n_pts = \
            con_gcn.hid_dim, con_gcn.emd_dim, con_gcn.coords_dim, \
                con_gcn.num_layer, con_gcn.n_head, con_gcn.dropout, con_gcn.n_pts
                
        self.hid_dim = self.hid_dim
        self.emd_dim = self.hid_dim*4
        
        ### Set the split point between regular and implicit layers ###
        self.implicit_start = num_layers - 1  # Use implicit layers for last layer only
        if hasattr(con_gcn, 'implicit_start_layer'):
            self.implicit_start = con_gcn.implicit_start_layer
            
        self.n_layers = num_layers
        self.num_timesteps = config.diffusion.num_diffusion_timesteps

        _gconv_input = ChebConv(in_c=self.coords_dim[0], out_c=self.hid_dim, K=2)
        _regular_layers = []
        _implicit_layers = []
        _attention_layer_regular = []

        dim_model = self.hid_dim
        c = copy.deepcopy
        
        # Create regular and implicit blocks
        for i in range(num_layers):
            attn = MultiHeadedAttention(n_head, dim_model)
            gcn = GraphNet(in_features=dim_model, out_features=dim_model, n_pts=n_pts)
            
            if i < self.implicit_start:
                # Regular layers for first part
                _regular_layers.append(_ResChebGC_diff(adj=adj, input_dim=self.hid_dim, output_dim=self.hid_dim,
                    emd_dim=self.emd_dim, hid_dim=self.hid_dim, p_dropout=0.1))
                _attention_layer_regular.append(GraAttenLayer(dim_model, c(attn), c(gcn), dropout))
            else:
                # Implicit layers for last part
                _implicit_layers.append(ImplicitGCNAttentionBlock(
                    adj=adj, 
                    dim_model=dim_model, 
                    attn=c(attn), 
                    gcn=c(gcn), 
                    emd_dim=self.emd_dim,
                    dropout=dropout
                ))

        self.gconv_input = _gconv_input
        self.regular_layers = nn.ModuleList(_regular_layers)
        self.implicit_layers = nn.ModuleList(_implicit_layers)
        self.atten_layers_regular = nn.ModuleList(_attention_layer_regular)
        self.gconv_output = ChebConv(in_c=dim_model, out_c=self.coords_dim[1], K=2)
        
        # Diffusion configuration
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.hid_dim, self.emd_dim),
            torch.nn.Linear(self.emd_dim, self.emd_dim),
        ])
        
        # For monitoring convergence
        self.iteration_counts = []
        
        # Initialize iteration counts from config
        initial_iter = getattr(con_gcn, 'implicit_max_iter', 1)
        for layer in self.implicit_layers:
            layer.max_iter = initial_iter
            layer.tol = getattr(con_gcn, 'implicit_tol', 0.01)

    def update_implicit_iterations(self, epoch, total_epochs=80, start_epochs=20, max_iterations=15):
        """
        Update the maximum iterations for implicit layers based on the current epoch.
        
        Args:
            epoch: Current training epoch
            total_epochs: Total number of training epochs
            start_epochs: Number of epochs to keep at minimum iterations
            max_iterations: Maximum number of iterations to reach by the end
        """
        if epoch < start_epochs:
            # First start_epochs: Use only 1 iteration
            max_iter = 1
        else:
            # Gradually increase from 1 to max_iterations over the remaining epochs
            remaining_epochs = total_epochs - start_epochs
            current_position = min(epoch - start_epochs, remaining_epochs)
            # Linear interpolation from 1 to max_iterations
            max_iter = 1 + (max_iterations - 1) * (current_position / remaining_epochs)
            max_iter = int(max_iter)
        
        # Update max_iter for all implicit layers
        for layer in self.implicit_layers:
            layer.max_iter = max_iter
        
        return max_iter

    def forward(self, x, mask, t, cemd):
        # Calculate t_fraction for dynamic iteration control
        t_fraction = t.float() / self.num_timesteps
        
        # Set warm starting based on config
        for layer in self.implicit_layers:
            layer.enable_warmstart = getattr(self.config.testing, 'enable_warmstart', False)
        
        # Timestep embedding
        temb = get_timestep_embedding(t, self.hid_dim)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        
        out = self.gconv_input(x, self.adj)
        
        # First part: regular layers
        for i in range(len(self.regular_layers)):
            out = self.atten_layers_regular[i](out, mask)
            out = self.regular_layers[i](out, temb)
        
        # Second part: implicit layers
        for i in range(len(self.implicit_layers)):
            t_frac_avg = t_fraction.mean().item()  # Average for the batch
            out = self.implicit_layers[i](out, mask, temb, t_frac_avg)
            
            # Store iteration count for monitoring
            self.iteration_counts.append(self.implicit_layers[i].num_iterations)
        
        out = self.gconv_output(out, self.adj)
        return out
    
    def get_iteration_stats(self):
        """Return statistics about implicit iterations for monitoring"""
        if not self.iteration_counts:
            return {"avg": 0, "max": 0, "min": 0}
            
        return {
            "avg": sum(self.iteration_counts) / len(self.iteration_counts),
            "max": max(self.iteration_counts),
            "min": min(self.iteration_counts)
        }
    
    def reset_iteration_stats(self):
        """Reset iteration statistics"""
        self.iteration_counts = []