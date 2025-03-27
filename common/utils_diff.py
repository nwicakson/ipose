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
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a


def generalized_steps(x, src_mask, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
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
        
        # Set warm starting for implicit layers based on kwargs
        enable_warmstart = kwargs.get("enable_warmstart", False)
        if hasattr(model, 'module') and hasattr(model.module, 'implicit_layers'):
            for layer in model.module.implicit_layers:
                layer.enable_warmstart = enable_warmstart
        elif hasattr(model, 'implicit_layers'):
            for layer in model.implicit_layers:
                layer.enable_warmstart = enable_warmstart
                
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).cuda()
            next_t = (torch.ones(n) * j).cuda()
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1]
            
            # Clear some memory before the forward pass
            if i % 5 == 0:  # Clear every few steps to save memory
                torch.cuda.empty_cache()
                
            et = model(xt, src_mask, t.float(), 0)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t)
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next)
            
            # Clean up to save memory
            if len(xs) > 2:
                # Only keep the latest and the final result
                xs = [xs[0], xs[-1]]
                
            # Keep only the latest prediction
            if len(x0_preds) > 1:
                # Replace with the most recent one only
                x0_preds = [x0_preds[-1]]

    return xs, x0_preds