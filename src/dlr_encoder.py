import torch
from torch import nn
import torch.nn.functional as F
from operator import itemgetter

def get_phi4_graph():
    """
    'Thick' Graph: Focuses on the interaction between u0 and xi.
    Everything revolves around how u0 transforms under the impact of noise and nonlinearity.
    """
    graph = {
        # --- LEVEL 0: RAW MATERIALS ---
        'xi': {},        # Index 0: Base noise
        'Ic(u0)': {},    # Index 1: Linearly propagating u0 (Latent)

        # --- LEVEL 1: BASIC INTERACTIONS ---
        
        # 1. Pure reaction of Noise (Wick base)
        'I(xi)': {'xi': 1}, 
        
        # 2. Nonlinearity of u0 itself (Important for Drift)
        # Simulates u0 self-interacting (u^2, u^3) and then propagating
        'I(u0^2)': {'Ic(u0)': 2},
        'I(u0^3)': {'Ic(u0)': 3},

        # 3. Direct interaction between u0 and xi (Important for Diffusion)
        # u0 is hit by noise xi and then propagates
        'I(u0*xi)': {'Ic(u0)': 1, 'xi': 1},

        # --- LEVEL 2: HIGH-ORDER INTERACTIONS (Deep Interaction) ---

        # 4. Wick Powers (Mandatory for Singularity)
        'I(xi^2)': {'xi': 2},
        'I(xi^3)': {'xi': 3},

        # 5. "Noise Amplification" effect by u0
        # Takes the smoothed noise result (I(xi)) multiplied by the original u0
        'I(I(xi)*u0)': {'I(xi)': 1, 'Ic(u0)': 1},

        # 6. "u0 Deformation" effect by noise
        # Takes the deformed u0 result (I(u0*xi)) multiplied further by xi
        'I(I(u0*xi)*xi)': {'I(u0*xi)': 1, 'xi': 1}
        }   
    return graph

class ParabolicIntegrate_2d(nn.Module):
    def __init__(self, graph, T, X, Y, BC = 'P', eps = 1, device = None, dtype = None):
            self.factory_kwargs = {'device': device, 'dtype': dtype}
            super().__init__()
            keys = list(graph)
            self.graph = [{keys.index(it): graph[key][it] for it in graph[key]} for key in keys] # model graph
            self.isDerivative = [(int(key[1]) if key[1].isdigit() else False) for key in graph.keys()]
            self.Operator = [key[0] for key in graph.keys()] ## I or J
            self.only_xi = [(True if 'u0' not in key else False) for key in graph.keys()] # if feature is only determined by xi
        
            keys_list = list(graph.keys())
            
            
            self.U0FeatureIndex = [
                i for i, key in enumerate(keys_list) 
                if ('u0' in key and 'xi' not in key)
            ]
            
           
            all_indices = set(range(len(keys_list)))
            self.xiFeatureIndex = sorted(list(all_indices - set(self.U0FeatureIndex)))
            
            self.FeatureIndex = list(range(len(keys_list))) 
            
            print(f"Pure U0 Indices: {self.U0FeatureIndex}") 
            print(f"Xi/Mixed Indices: {self.xiFeatureIndex}")
            self.BC = BC #Boundary condition 'D' - Dirichlet, 'N' - Neuman, 'P' - periodic
            self.eps = eps # viscosity
            self.X_points = X # discretization of space (O_X space)
            self.Y_points = Y # discretization of space (O_Y space)
            self.T_points = T # discretization of time (O_T space)
            self.X = len(self.X_points) # number of space X points
            self.Y = len(self.Y_points) # number of space Y points
            self.T = len(self.T_points) # number of time points

            self.dt = self.T_points[1] - self.T_points[0]
            self.dx = self.X_points[1] - self.X_points[0]  # for equaly spaced points
            self.dy = self.Y_points[1] - self.Y_points[0]
            filter = torch.tensor([[[[0.25,0.5,0.25],[0.5,-3.,0.5],[0.25,0.5,0.25]]]], **self.factory_kwargs) ## kernel of 2D Laplace operator
            self.register_buffer("filter", filter)

            filterI = torch.tensor([[[[0.,0.,0.],[0.,1.,0.],[0.,0.,0.]]]], **self.factory_kwargs) + \
                      self.eps * filter * self.dt/self.dx**2 ## kernel of 2D Laplace operator
            self.register_buffer("filterI", filterI)
            DX = self.DiffMat(self.X, self.dx) 
            DY = self.DiffMat(self.Y, self.dy)
            self.register_buffer("DX", DX)
            self.register_buffer("DY", DY)
            Jm = self.JMat(self.X_points, self.Y_points, self.dx, self.dy)
            self.register_buffer("Jm", Jm)

    def JMat(self, X, Y, dx, dy): # [X,Y,X,Y,2]
        K = torch.ones(len(X),len(Y),len(X),len(Y),2, device=X.device, dtype=X.dtype) * dx * dy / (2 * torch.pi)
        return K
        for i in range(len(X)):
            for j in range(len(Y)):
                for k in range(len(X)):
                    for l in range(len(Y)):
                        if (i == k and j == l):
                            K[i,j,k,l,0] = 0.
                            K[i,j,k,l,1] = 0.
                        else:
                            K[i,j,k,l,0] = (Y[j]-Y[l]) / ((X[i]-X[k])**2 + (Y[j] - Y[l])**2)
                            K[i,j,k,l,1] = (X[k]-X[i]) / ((X[i]-X[k])**2 + (Y[j] - Y[l])**2) 
        return K

    def DiffMat(self, N, dx):
        A = torch.diag(-1*torch.ones(N-1), diagonal=1) + torch.diag(torch.ones(N-1), diagonal=-1)
        A[0,-1], A[-1,0] = 1, -1
        A = A.to(**self.factory_kwargs) / (2*dx)

        return A

    def Laplace_2d(self, arr):
        return F.conv2d(F.pad(arr.unsqueeze(1), (1,1,1,1), mode = 'circular'), self.filter).squeeze(1)*self.dt/self.dx**2 # ~ 30s


        return out*self.dt/self.dx**2 # ~ 45s

    def Laplace_I_2d(self, arr):
        return F.conv2d(F.pad(arr.unsqueeze(1), (1,1,1,1), mode = 'circular'), self.filterI).squeeze(1) # ~ 30s

    def I_c(self, U0):
        '''
            U0: [B, X, Y]
            return: [B, T, X, Y]
        '''
        factory_kwargs = {'device': U0.device, 'dtype': U0.dtype}
        Solution = torch.zeros(len(U0), self.T, self.X, self.Y, **factory_kwargs)
        # Initialize
        Solution[:,0,:,:] = U0
        
        # Finite difference method.
        # u_{n+1} = u_n + mu(u_n)*dt + sigma(u_n)*dW_{n} + (dx)^{-2} A*u_{n}*dt 
        # mu = sigma = 0 for solving I_c[u_0]
        for i in range(1, self.T):
            Solution[:,i,:,:] = self.Laplace_I_2d(Solution[:,i-1,:,:])


        return Solution
    # def KernelMat(self, X, Y, dx, dt): # evaluation (I - dt \Delta)^{-1} of shape [XY,XY]

    def forward(self, W = None, Latent = None, XiFeature = None, returnFeature = 'all', diff = False):
        '''
            W: [B, T, X, Y]
            Latent: [B, T, X, Y]
            XiFeature: [B, T, X, Y, F]
            diff: bool

            Return: [B, T, X, Y, F]
        '''
        factory_kwargs = {'device': W.device, 'dtype': W.dtype}
        # differentiate noise/forcing to create dW 
        integrated = []

        # add xi features as integrated[0]
        if XiFeature is not None:
            integrated.append(XiFeature[...,0])
        elif W is not None:
            # differentiate noise/forcing to create dW 
            if diff:
                dW = torch.zeros(W.shape, **factory_kwargs)
                dW[:,1:,:,:] = torch.diff(W, dim = 1)/self.dt
            else:
                dW = W#*self.dt
            integrated.append(dW)
            # if torch.isinf(integrated[-1]).any():
            #     raise ValueError('dW is nan')
        else:
            raise "empty itorchut"

        firiter = 1

        # if itorchut is given, substitude I_c[u_0] by itorchut, recorded in integrated[1]
        if Latent is not None:
            integrated.append(Latent)
            firiter = 2

        B = len(W) if W is not None else len(Latent) # current batchsize

        for k, dic in enumerate(self.graph[firiter:],firiter):
            if (self.only_xi[k] and XiFeature is not None): # have cached XiFeature
                integrated.append(XiFeature[...,k])
                continue
            
            if (not self.only_xi[k] and returnFeature == 'xi'):
                integrated.append(torch.ones(B, self.T, self.X, self.Y,  **factory_kwargs))
                continue

            if (self.isDerivative[k]): # derivative
                if (self.Operator[k] == 'I'):
                    if self.isDerivative[k] == 1:
                        tp = torch.einsum('btxy,xn->btny', integrated[list(dic.keys())[0]], self.DX)
                    elif self.isDerivative[k] == 2:
                        tp = torch.einsum('btxy,yn->btxn', integrated[list(dic.keys())[0]], self.DY)

                elif (self.Operator[k] == 'J'):
                        tp = torch.einsum('btxy,xymn->btmn', integrated[list(dic.keys())[0]], self.Jm[...,self.isDerivative[k]-1])
                integrated.append(tp)
                continue
            
            # compute the integral with u_0
            
            tmp = torch.ones(B, self.T, self.X, self.Y,  **factory_kwargs) # [B, T, X, Y]
            for it, p in dic.items():
                if (p == 1):
                    tmp = tmp * integrated[it] #.clone()
                else:
                    tmp = tmp * torch.pow(integrated[it], p)

            tmp = tmp * self.dt
            tmp[:,0,:,:] = 0
            for i in range(1,self.T):
                tmp[:,i,:,:] = self.Laplace_I_2d(tmp[:,i-1,:,:]) + tmp[:,i,:,:]
            integrated.append(tmp)

        if returnFeature == 'all':
            Feature = torch.stack(integrated, dim = -1)
        elif returnFeature == 'U0':
            if (len(self.U0FeatureIndex) == 1):
                Feature = itemgetter(*self.U0FeatureIndex)(integrated).unsqueeze(-1)
            else:
                Feature = torch.stack(itemgetter(*self.U0FeatureIndex)(integrated), dim = -1)
        elif returnFeature == 'xi':
            Feature = torch.stack(itemgetter(*self.xiFeatureIndex)(integrated), dim = -1)
        else:
            Feature = torch.stack(itemgetter(*self.FeatureIndex)(integrated), dim = -1)
        
        return Feature

    def discrete_diff_2d(self, vec, N, axis, higher = True):
        a = torch.zeros_like(vec)
        if axis == 1:
            if higher: # central approximation of a dervative
                a[...,:-1,:] = (torch.roll(vec[...,:-1,:], -1, dims = -2) - torch.roll(vec[...,:-1,:], 1, dims = -2))/2
            else:
                a[...,:-1,:] = vec[...,:-1,:] - torch.roll(vec[...,:-1,:], 1, dims = -2)
            a[...,-1,:] = a[...,0,:] # enforce periodic boundary condions
        if axis == 2:
            if higher: # central approximation of a dervative
                a[...,:,:-1] = (torch.roll(vec[...,:,:-1], -1, dims = -1) - torch.roll(vec[...,:,:-1], 1, dims = -1))/2
            else:
                a[...,:,:-1] = vec[...,:,:-1] - torch.roll(vec[...,:,:-1], 1, dims = -1)
            a[...,:,-1] = a[...,:,0] # enforce periodic boundary condions

        return a


class LearnableDLREncoder(nn.Module):
    """
    Learnable version of DLR Encoder (Simulating the RSLayer + MLP structure of DLR-Net).
    Performs:
    1. Physical feature calculation (ParabolicIntegrate).
    2. Mixing and compressing features using MLP .
    """
    def __init__(self, graph, T, X, Y, output_channels=4, device=None):
        super().__init__()
        
        # 1. Mathematical Core (Corresponds to self.RSLayer0 in phi41.py)
        self.physics_engine = ParabolicIntegrate_2d(graph, T, X, Y, device=device)
        
        # Calculate the number of input features (F_raw)
        # The number of features returned by the Graph determines the MLP input size
        self.F_in = len(graph) 

        self.u0_idx = self.physics_engine.U0FeatureIndex
        self.xi_idx = self.physics_engine.FeatureIndex # Temporarily referred to as the rest (or use xiFeatureIndex)
        
        # Get the complement of u0_idx to make xi_idx more accurate (avoid duplication if needed)
        all_indices = set(range(self.F_in))
        self.xi_idx_clean = list(all_indices - set(self.u0_idx))
        
        # Re-sort to ensure order
        self.u0_idx.sort()
        self.xi_idx_clean.sort()
        
        # 2. MLP Feature Mixing 
        # In phi41.py: Linear -> GELU -> Linear
        # Here we add LayerNorm and Tanh to ensure stability for the 2D Singular problem
        self.mlp = nn.Sequential(
            nn.Linear(self.F_in, 32),      # Expand/Mix information
            nn.LayerNorm(32),              # Normalization (Crucial to avoid explosion)
            nn.GELU(),                     # Activation similar to original DLR
            nn.Linear(32, 1), # Compress to desired Context channels, previously using output_channels
            nn.Tanh()                      # Amplitude constraint (Safety lock)
        )
        

    def I_c(self, U0):
        # Wrapper to calculate u0 propagation (used for initial Latent creation)
        return self.physics_engine.I_c(U0)

    def forward(self, W, Latent):
    
        """
        Input: W (Noise), Latent (s_0 - usually I_c[u0])
        Output: [F1, F2] concatenated
        """
        
        # --- BLOCK 1 ---
        F1 = self.physics_engine(W=W, Latent=Latent, returnFeature='all', diff=False)
        s1 = self.mlp(F1).squeeze(-1)
        
        # --- BLOCK 2 ---
        F2 = self.physics_engine(W=W, Latent=s1, returnFeature='all', diff=False)
        
        # --- FEATURE SEPARATION  ---
    
        
        # 1. Get Features for F (only related to u0)
        # Aggregate u0 features from both blocks
        u0_F1 = F1[..., self.u0_idx] 
        u0_F2 = F2[..., self.u0_idx]
        out_u0 = torch.cat([u0_F1, u0_F2], dim=-1) # Output for network F
        
        # 2. Get Features for G (related to xi)
        xi_F1 = F1[..., self.xi_idx_clean]
        xi_F2 = F2[..., self.xi_idx_clean]
        out_xi = torch.cat([xi_F1, xi_F2], dim=-1) # Output for network G
        
        # Return 2 separate tensors
        return out_u0, out_xi