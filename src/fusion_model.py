"""
This is the fusion_model.py file.
It implements the Fusion_NSPDE model which combines Neural SPDE with DLR Encoder.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
    
import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.fixed_point_solver import NeuralFixedPoint
from src.root_find_solver import NeuralRootFind
from src.diffeq_solver import DiffeqSolver
from src.utilities import LpLoss
from src.dlr_encoder import LearnableDLREncoder, get_phi4_graph

class SPDEFunc0d(torch.nn.Module):
    """ Modelling local operators F and G in (latent) SPDE (d_t - L)u = F(u)dt + G(u) dxi_t 
    """
    def __init__(self, noise_channels, hidden_channels):
        super().__init__()
        self.noise_channels = noise_channels
        self.hidden_channels = hidden_channels

        # local non-linearity F
        model_F = [nn.Conv1d(hidden_channels, hidden_channels, 1), nn.BatchNorm1d(hidden_channels), nn.Tanh()]
        self.F = nn.Sequential(*model_F)  

        # local non-linearity G
        model_G = [nn.Conv1d(hidden_channels, hidden_channels*noise_channels, 1), nn.BatchNorm1d(hidden_channels*noise_channels), nn.Tanh()]  
        self.G = nn.Sequential(*model_G)

    def forward(self, z):
        return self.F(z), self.G(z).view(z.size(0), self.hidden_channels, self.noise_channels, z.size(2))

class SPDEFunc1d(torch.nn.Module):
    """
    (Explicit Multiplication).
    Input Context: [dXi_raw | Xi_features | U_features]
    Output: F(z, u) + G(z, xi_features) * dXi_raw
    """
    def __init__(self, hidden_channels, dim_u_feat, dim_xi_feat):
        super().__init__()
        
        # --- F (Drift) ---
        input_F = hidden_channels + dim_u_feat
        self.net_F = nn.Sequential(
            nn.Conv2d(input_F, 4*hidden_channels, 1),
            nn.GroupNorm(4, 4*hidden_channels),
            nn.GLU(dim=1),
            nn.Conv2d(2*hidden_channels, hidden_channels, 1),

        )
        
        # ---  G (Diffusion Coefficient) ---

        input_G_channels = hidden_channels + (dim_xi_feat - 1) + dim_u_feat
        
        self.net_G = nn.Sequential(
            nn.Conv2d(input_G_channels, 4*hidden_channels, 1),
            nn.GroupNorm(4, 4*hidden_channels),
            nn.GLU(dim=1),
            nn.Conv2d(2*hidden_channels, hidden_channels, 1),
        )
        
        self.split_point = dim_xi_feat

    def forward(self, z, combined_context):
        """
        z: [Batch, Hidden, X, Y]
        combined_context: [Batch, Total_Channels, X, Y] 
        """
        
        d_xi_raw = combined_context[:, 0:1, ...] # Shape: [B, 1, X, Y]
        
        context_for_G = combined_context[:, 1:, ...] 
        
        u_features_only = combined_context[:, self.split_point:, ...]
        
        drift_term = self.net_F(torch.cat([z, u_features_only], dim=1))
        
        diffusion_coeff = self.net_G(torch.cat([z, context_for_G], dim=1))
        
        stochastic_term = diffusion_coeff * d_xi_raw
        
        return drift_term + stochastic_term


class SPDEFunc2d(torch.nn.Module):
    """ Modelling local operators F and G in (latent) SPDE (d_t - L)u = F(u)dt + G(u) dxi_t 
    """
    def __init__(self, noise_channels, hidden_channels):
        super().__init__()
        self.noise_channels = noise_channels
        self.hidden_channels = hidden_channels

        # local non-linearity F 
        model_F = [nn.Conv3d(hidden_channels, hidden_channels, 1), nn.BatchNorm3d(hidden_channels), nn.Tanh()]
        self.F = nn.Sequential(*model_F)  

        # local non-linearity G
        model_G = [nn.Conv3d(hidden_channels, hidden_channels*noise_channels, 1), nn.BatchNorm3d(hidden_channels*noise_channels), nn.Tanh()]  
        self.G = nn.Sequential(*model_G)

    def forward(self, z):
        return self.F(z), self.G(z).view(z.size(0), self.hidden_channels, self.noise_channels, z.size(2), z.size(3), z.size(4))

class Fusion_NSPDE(torch.nn.Module):  

    def __init__(self, dim, in_channels, noise_channels, hidden_channels, modes1, modes2=None, modes3=None, n_iter=4, solver='fixed_point', T_points=None, X_points=None, Y_points=None, **kwargs):
        super().__init__()
        
        self.dim = dim
        self.solver_type = solver


        self.use_dlr = False
        self.context_channels = 0 
        dim_u_total = 0
        dim_xi_total = 1 

        if T_points is not None and X_points is not None:
            self.use_dlr = True
            

            graph = get_phi4_graph()
            F_U0_ONE_BLOCK = 3 
            F_XI_ONE_BLOCK = 7 
            NUM_BLOCKS = 2
            
            dim_u_total = F_U0_ONE_BLOCK * NUM_BLOCKS    
            dim_xi_context = F_XI_ONE_BLOCK * NUM_BLOCKS  
            
       
            dim_xi_total = 1 + dim_xi_context

            self.context_channels = dim_u_total + dim_xi_context 

            device_dlr = kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            self.dlr_encoder = LearnableDLREncoder(
                graph=graph, 
                T=T_points, X=X_points, Y=Y_points,
                device=device_dlr
            )
            
            self.norm_u = nn.InstanceNorm3d(dim_u_total, affine=True)
            self.norm_xi = nn.InstanceNorm3d(dim_xi_context, affine=True)
            
            print(f"DLR Split Mode ACTIVE. U_feat: {dim_u_total}, Xi_feat: {dim_xi_total}")

        self.lift = nn.Linear(in_channels, hidden_channels)

        if dim==2 and solver=='diffeq':
            self.spde_func = SPDEFunc1d(
                hidden_channels=hidden_channels,
                dim_u_feat=dim_u_total,  
                dim_xi_feat=dim_xi_total  
            )
        else:
            
            pass 

        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, 128), nn.ReLU(), nn.Linear(128, in_channels)
        )

       
        if 'device' in kwargs: del kwargs['device']

        if solver=='fixed_point':
            self.solver = NeuralFixedPoint(self.spde_func, n_iter, modes1, modes2, modes3)
        elif solver=='diffeq':
            self.solver = DiffeqSolver(hidden_channels, self.spde_func, modes1, modes2, **kwargs)
        elif solver=='root_find': 
            self.solver = NeuralRootFind(self.spde_func, n_iter, modes1, modes2, modes3, **kwargs)


    def forward(self, u0, xi, grid=None):
    
        if u0.dim() == 3: u0_in = u0.unsqueeze(1)
        else: u0_in = u0
        if xi.dim() == 4: xi_in = xi.unsqueeze(1)
        else: xi_in = xi

       
        if self.use_dlr:
            xi_dlr_in = xi_in.squeeze(1).permute(0, 3, 1, 2)
            u0_dlr_in = u0_in.squeeze(1)

            # I_c[u0]
            u0_latent = self.dlr_encoder.I_c(u0_dlr_in)

            # Encoder
            u_ctx_raw, xi_ctx_raw = self.dlr_encoder(W=xi_dlr_in, Latent=u0_latent)
            
            # normalize & Permute
            # [B, T, X, Y, C] -> [B, C, X, Y, T]
            u_ctx = self.norm_u(u_ctx_raw.permute(0, 4, 1, 2, 3)).permute(0, 1, 3, 4, 2)
            xi_ctx = self.norm_xi(xi_ctx_raw.permute(0, 4, 1, 2, 3)).permute(0, 1, 3, 4, 2)
            
            # (Strategy Concatenation)
            combined_input = torch.cat([xi_in, xi_ctx, u_ctx], dim=1)
            
        else:
            combined_input = xi_in # Fallback

        # 5. Lifting & Solving
        if grid is not None: grid = grid[0]
        z0 = self.lift(u0_in.permute(0,2,3,1)).permute(0,3,1,2) # [B, H, X, Y]

        zs = self.solver(z0, combined_input, grid) # [B, H, X, Y, T]

        ys = self.readout(zs.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        
        return ys.squeeze(1)

def init_model():
    print("Initialisation of the model.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    T_points = torch.linspace(0, 0.025, 251).to(device) 
    X_points = torch.linspace(0, 1, 32).to(device)
    Y_points = torch.linspace(0, 1, 32).to(device)

    model = Fusion_NSPDE(dim=2, in_channels=1, noise_channels=1, hidden_channels=32,
                       n_iter=1, modes1=8, modes2=8,
                       solver='diffeq',
                       T_points=T_points, X_points=X_points, Y_points=Y_points,
                       device=device).to(device)
    return model

if __name__ == '__main__':

    model = init_model()
    print("Model initialized successfully.")