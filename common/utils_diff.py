from __future__ import absolute_import, division

import os
import torch
import numpy as np

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def compute_alpha(beta, t):
    # Safe concatenation with bounds checking
    t = torch.clamp(t, min=0, max=len(beta))
    if len(beta.shape) == 0:
        beta = beta.unsqueeze(0)  # Handle scalar beta
    beta_padded = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    # Add extra safety check for indexing
    t_safe = torch.clamp(t + 1, max=len(beta_padded) - 1)
    a = (1 - beta_padded).cumprod(dim=0).index_select(0, t_safe).view(-1, 1, 1)
    return a


def generalized_steps(x, src_mask, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        
        # Ensure seq is valid - clip to valid range
        max_timestep = b.shape[0] - 1
        seq = [min(s, max_timestep) for s in seq]
        
        # Create seq_next with safe indexing
        seq_next = [-1] + list(seq[:-1])
        
        x0_preds = []
        xs = [x]
        
        # Reset any previous solutions at the start of sampling
        if hasattr(model, 'module') and hasattr(model.module, 'implicit_layers'):
            for layer in model.module.implicit_layers:
                layer.initial_z = None
        elif hasattr(model, 'implicit_layers'):
            for layer in model.implicit_layers:
                layer.initial_z = None
        
        # Ensure warmstart is properly set
        enable_warmstart = kwargs.get("enable_warmstart", False)
        if hasattr(model, 'module') and hasattr(model.module, 'implicit_layers'):
            for layer in model.module.implicit_layers:
                layer.enable_warmstart = enable_warmstart
        elif hasattr(model, 'implicit_layers'):
            for layer in model.implicit_layers:
                layer.enable_warmstart = enable_warmstart
        
        try:
            for i, j in zip(reversed(seq), reversed(seq_next)):
                # SAFETY CHECK - ensure indices are within bounds
                i_safe = min(i, b.shape[0] - 1)  # Clamp to valid range
                j_safe = max(j, 0)  # Ensure j is not negative
                
                t = (torch.ones(n) * i_safe).cuda()
                next_t = (torch.ones(n) * j_safe).cuda()
                
                # Convert to long safely
                t_long = t.long().clamp(0, b.shape[0] - 1)
                next_t_long = next_t.long().clamp(0, b.shape[0] - 1)
                
                at = compute_alpha(b, t_long)
                at_next = compute_alpha(b, next_t_long)
                
                xt = xs[-1]
                
                # Clear memory periodically
                if i % 5 == 0:
                    torch.cuda.empty_cache()
                    
                et = model(xt, src_mask, t.float(), 0)
                
                # Check for NaNs
                if torch.isnan(et).any():
                    print("NaN detected in noise prediction, replacing with zeros")
                    et = torch.where(torch.isnan(et), torch.zeros_like(et), et)
                
                # Compute x0_t safely
                x0_t = (xt - et * (1 - at).sqrt()) / (at.sqrt() + 1e-8)  # Add epsilon to avoid division by zero
                
                # Check for NaNs again
                if torch.isnan(x0_t).any():
                    print("NaN detected in x0_t calculation, using original xt")
                    x0_t = xt
                
                x0_preds.append(x0_t)
                
                # Compute coefficients safely
                c1 = (
                    kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at + 1e-8)).sqrt()
                )
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                
                # Generate next sample
                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                
                # Check for NaNs in the next sample
                if torch.isnan(xt_next).any():
                    print("NaN detected in xt_next generation, using previous xt")
                    xt_next = xt
                
                xs.append(xt_next)
                
                # Clean up to save memory
                if len(xs) > 2:
                    # Only keep the latest and the final result
                    xs = [xs[0], xs[-1]]
                    
                # Keep only the latest prediction
                if len(x0_preds) > 1:
                    # Replace with the most recent one only
                    x0_preds = [x0_preds[-1]]
        
        except Exception as e:
            print(f"Error in generalized_steps: {e}")
            # Return a safe value in case of error
            if len(xs) < 2:
                # If no steps completed, just return the input
                return [x, x], [x]
        
        return xs, x0_preds