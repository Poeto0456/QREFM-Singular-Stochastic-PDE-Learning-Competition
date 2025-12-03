# Adapted from https://github.com/crispitagorico/torchspde
# Modified for current implementation by the authors of SPDEBench

import os
import os.path as osp
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import scipy.io
import h5py
import pandas as pd
import csv
import itertools
import matplotlib as mpl
from matplotlib.gridspec import SubplotSpec
from timeit import default_timer
from src.model import NeuralSPDE
from src.utilities import *

# Import ACFLoss (Optional)
try:
    from evaluation.loss import ACFLoss
except ImportError:
    pass

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return trainable_params

#===========================================================================
# Data Loaders for Neural SPDE
#===========================================================================

def dataloader_nspde_2d(u, xi=None, ntrain=1000, ntest=200, T=51, sub_t=1, sub_x=4, batch_size=20, dataset=None):
    if xi is None:
        print('There is no known forcing')

    if dataset=='sns':
        T, sub_t, sub_x = 51, 1, 4

    u0_train = u[:ntrain, ::sub_x, ::sub_x, 0].unsqueeze(1)
    u_train = u[:ntrain, ::sub_x, ::sub_x, :T:sub_t]

    if xi is not None:
        xi_train = xi[:ntrain, ::sub_x, ::sub_x, 0:T:sub_t].unsqueeze(1)
    else:
        xi_train = torch.zeros_like(u_train).unsqueeze(1)

    u0_test = u[-ntest:, ::sub_x, ::sub_x, 0].unsqueeze(1)
    u_test = u[-ntest:, ::sub_x, ::sub_x, 0:T:sub_t]

    if xi is not None:
        xi_test = xi[-ntest:, ::sub_x, ::sub_x, 0:T:sub_t].unsqueeze(1)
    else:
        xi_test = torch.zeros_like(u_test).unsqueeze(1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_train, xi_train, u_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_test, xi_test, u_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

#===========================================================================
# Training and Testing functionalities
#===========================================================================

def eval_nspde(model, test_dl, myloss, batch_size, device):
    ntest = len(test_dl.dataset)
    test_loss = 0.
    model.eval()
    with torch.no_grad():
        for u0_, xi_, u_ in test_dl:    
            u0_, xi_, u_ = u0_.to(device), xi_.to(device), u_.to(device)
            u_pred = model(u0_, xi_)
            
            # Dynamic batch size
            curr_bs = u_pred.shape[0]
            loss = myloss(u_pred[...,1:].reshape(curr_bs, -1), u_[...,1:].reshape(curr_bs, -1))
            test_loss += loss.item()
    return test_loss / ntest

def train_nspde(model, train_loader, test_loader, device, myloss, batch_size=20, epochs=5000,
                learning_rate=0.001, scheduler_step=100, scheduler_gamma=0.5, print_every=20,
                weight_decay=1e-4, delta=0, factor=0.1,
                plateau_patience=None, plateau_terminate=None, time_train=False, time_eval=False,
                checkpoint_file='checkpoint.pt'): 

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if plateau_patience is None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=plateau_patience, factor=factor, threshold=1e-6, min_lr=1e-7)
    
    if plateau_terminate is not None:
        early_stopping = EarlyStopping(patience=plateau_terminate, verbose=False, delta=delta, path=checkpoint_file)

    ntrain = len(train_loader.dataset)
    ntest = len(test_loader.dataset)

    losses_train = []
    losses_test = []

    times_train = [] 
    times_eval = []

    try:
        for ep in range(epochs):
            model.train()
            train_loss = 0.
            
            for batch_idx, (u0_, xi_, u_) in enumerate(train_loader):
                u0_ = u0_.to(device)
                xi_ = xi_.to(device)
                u_ = u_.to(device)

                t1 = default_timer()
                
                optimizer.zero_grad()
                
                # Forward (1 output)
                u_pred = model(u0_, xi_)
                
                # Loss
                curr_bs = u_pred.shape[0]
                loss = myloss(u_pred[..., 1:].reshape(curr_bs, -1), u_[..., 1:].reshape(curr_bs, -1))

                train_loss += loss.item()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                times_train.append(default_timer()-t1)

            # --- VALIDATION ---
            test_loss = 0.
            total_acf_score = 0.0
            
            model.eval()
            with torch.no_grad():
                for u0_, xi_, u_ in test_loader:
                    u0_ = u0_.to(device)
                    xi_ = xi_.to(device)
                    u_ = u_.to(device)

                    t1 = default_timer()
                    u_pred = model(u0_, xi_)
                    times_eval.append(default_timer()-t1)

                    curr_bs = u_pred.shape[0]
                    loss = myloss(u_pred[..., 1:].reshape(curr_bs, -1), u_[..., 1:].reshape(curr_bs, -1))
                    test_loss += loss.item()
                    
                    # ACF Monitor
                    try:
                        pred_permuted = u_pred.permute(0, 3, 1, 2)
                        target_permuted = u_.permute(0, 3, 1, 2)
                        acf_metric = ACFLoss(target_permuted, max_lag=50, stationary=True, name='val_acf')
                        acf_err = acf_metric.compute(pred_permuted)
                        total_acf_score += acf_err.mean().item()
                    except Exception as e:
                        pass # Ignore ACF errors if library missing

            avg_val_loss = test_loss / ntest
            avg_acf_score = total_acf_score / len(test_loader)
            
            if plateau_patience is None:
                scheduler.step()
            else:
                scheduler.step(avg_val_loss)
            
            if plateau_terminate is not None:
                early_stopping(avg_val_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            if ep % print_every == 0:
                losses_train.append(train_loss/ntrain)
                losses_test.append(avg_val_loss)
                print(f'Epoch {ep:04d} | Train L2: {train_loss/ntrain:.6f} | Val L2: {avg_val_loss:.6f} | Val ACF: {avg_acf_score:.6f}')

        if time_train and time_eval:
            return model, losses_train, losses_test, times_train, times_eval 
        elif time_train and not time_eval:
            return model, losses_train, losses_test, times_train
        elif time_eval and not time_train:
            return model, losses_train, losses_test, times_eval 
        else:
            return model, losses_train, losses_test
        
    except KeyboardInterrupt:
        print("Training interrupted explicitly.")
        return model, losses_train, losses_test


def hyperparameter_search_nspde_2d(train_dl, val_dl, test_dl, solver, d_h=[32], iter=[1, 2, 3], modes1=[32, 64], modes2=[32, 64],
                                epochs=500, print_every=20, lr=0.025, plateau_patience=100, plateau_terminate=100,
                                log_file='log_nspde', checkpoint_file='checkpoint.pt',
                                final_checkpoint_file='final.pt'):
    hyperparams = list(itertools.product(d_h, iter, modes1, modes2))

    loss = LpLoss(size_average=False)

    fieldnames = ['d_h', 'iter', 'modes1', 'modes2', 'nb_params', 'loss_train', 'loss_val', 'loss_test']
    with open(log_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)

    best_loss_val = 1000.

    for (_dh, _iter, _modes1, _modes2) in hyperparams:
        print('\n dh:{}, iter:{}, modes1:{}, modes2:{}'.format(_dh, _iter, _modes1, _modes2))

        model = NeuralSPDE(dim=2, in_channels=1, noise_channels=1, hidden_channels=_dh,
                           n_iter=_iter, modes1=_modes1, modes2=_modes2, solver=solver).cuda()

        nb_params = count_params(model)
        print('\n The model has {} parameters'.format(nb_params))

        _, _, _ = train_nspde(model, train_dl, val_dl, device, loss, batch_size=20, epochs=epochs, learning_rate=lr,
                              scheduler_step=500, scheduler_gamma=0.5, plateau_patience=plateau_patience,
                              plateau_terminate=plateau_terminate, print_every=print_every,
                              checkpoint_file=checkpoint_file)

        model.load_state_dict(torch.load(checkpoint_file))
        loss_test = eval_nspde(model, test_dl, loss, 20, device)
        loss_train = eval_nspde(model, train_dl, loss, 20, device)
        loss_val = eval_nspde(model, val_dl, loss, 20, device)

        if loss_val < best_loss_val:
            torch.save(model.state_dict(), final_checkpoint_file)
            best_loss_val = loss_val

        with open(log_file, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([_dh, _iter, _modes1, _modes2, nb_params, loss_train, loss_val, loss_test])


#===============================================================================
# Utilities (adapted from https://github.com/zongyi-li/fourier_neural_operator)
#===============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_grid(batch_size, dim_x, dim_y, dim_t=None):
    gridx = torch.linspace(0, 1, dim_x, dtype=torch.float)
    gridy = torch.linspace(0, 1, dim_y, dtype=torch.float)
    if dim_t:
        gridx = gridx.reshape(1, dim_x, 1, 1, 1).repeat([batch_size, 1, dim_y, dim_t, 1])
        gridy = gridy.reshape(1, 1, dim_y, 1, 1).repeat([batch_size, dim_x, 1, dim_t, 1])
        gridt = torch.linspace(0, 1, dim_t, dtype=torch.float)
        gridt = gridt.reshape(1, 1, 1, dim_t, 1).repeat([batch_size, dim_x, dim_y, 1, 1])
        return torch.cat((gridx, gridy, gridt), dim=-1).permute(0,4,1,2,3)
    gridx = gridx.reshape(1, dim_x, 1, 1).repeat([batch_size, 1, dim_y, 1])
    gridy = gridy.reshape(1, 1, dim_y, 1).repeat([batch_size, dim_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).permute(0,3,1,2)

class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()
        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float
        self.file_path = file_path
        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]
        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))
        if self.to_float:
            x = x.astype(np.float32)
        if self.to_torch:
            x = torch.from_numpy(x)
            if self.to_cuda:
                x = x.cuda()
        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

def get_gpu_mem(synchronize=True, empty_cache=True):
    return torch.cuda.memory_allocated(), torch.cuda.memory_cached()

def generate_mem_hook(handle_ref, mem, idx, hook_type, exp):
    def hook(self, *args):
        if len(mem) == 0 or mem[-1]["exp"] != exp:
            call_idx = 0
        else:
            call_idx = mem[-1]["call_idx"] + 1
        mem_all, mem_cached = get_gpu_mem()
        torch.cuda.synchronize()
        mem.append({
            'layer_idx': idx,
            'call_idx': call_idx,
            'layer_type': type(self).__name__,
            'exp': exp,
            'hook_type': hook_type,
            'mem_all': mem_all,
            'mem_cached': mem_cached,
        })
    return hook

def add_memory_hooks(idx, mod, mem_log, exp, hr):
    h = mod.register_forward_pre_hook(generate_mem_hook(hr, mem_log, idx, 'pre', exp))
    hr.append(h)
    h = mod.register_forward_hook(generate_mem_hook(hr, mem_log, idx, 'fwd', exp))
    hr.append(h)
    h = mod.register_backward_hook(generate_mem_hook(hr, mem_log, idx, 'bwd', exp))
    hr.append(h)

def log_mem(model, inp, mem_log=None, exp=None, model_type='NSPDE'):
    mem_log = mem_log or []
    exp = exp or f'exp_{len(mem_log)}'
    hr = []
    for idx, module in enumerate(model.modules()):
        add_memory_hooks(idx, module, mem_log, exp, hr)
    try:
        if model_type in ['NSPDE', 'NCDE']:
            out = model(inp[0], inp[1])
        else:
            out = model(inp)
        loss = out.sum()
        loss.backward()
    finally:
        [h.remove() for h in hr]
        return mem_log

def plot_mem(df, exps=None, normalize_call_idx=True, normalize_mem_all=True, filter_fwd=False, return_df=False, output_file=None):
    if exps is None:
        exps = df.exp.drop_duplicates()
    fig, ax = plt.subplots(figsize=(20, 10))
    for exp in exps:
        df_ = df[df.exp == exp]
        if normalize_call_idx:
            df_.call_idx = df_.call_idx / df_.call_idx.max()
        if normalize_mem_all:
            df_.mem_all = df_.mem_all - df_[df_.call_idx == df_.call_idx.min()].mem_all.iloc[0]
            df_.mem_all = df_.mem_all // 2 ** 20
        if filter_fwd:
            layer_idx = 0
            callidx_stop = df_[(df_["layer_idx"] == layer_idx) & (df_["hook_type"] == "fwd")]["call_idx"].iloc[0]
            df_ = df_[df_["call_idx"] <= callidx_stop]
        plot = df_.plot(ax=ax, x='call_idx', y='mem_all', label=exp)
        print('Maximum memory: {} MB'.format(df_['mem_all'].max()))
        if output_file:
            plot.get_figure().savefig(output_file)
    if return_df:
        return df_

def get_memory(device, reset=False, in_mb=True):
    if device is None:
        return float('nan')
    if device.type == 'cuda':
        if reset:
            torch.cuda.reset_max_memory_allocated(device)
        bytes = torch.cuda.max_memory_allocated(device)
        if in_mb:
            bytes = bytes / 1024 / 1024
        return bytes
    else:
        return float('nan')