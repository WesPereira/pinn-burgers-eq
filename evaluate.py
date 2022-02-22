import torch
import scipy.io
import argparse
import numpy as np
import torch.nn as nn
from pinn.utils.util import log, perf
from pinn.burgers_2d_net import SimpleModel
import matplotlib.pyplot as plt


def _get_args():
    parser = argparse.ArgumentParser(
        prog='Evaluate step',
        usage='%(prog)s [options] parser',
        description='Parser for hyperparams evaluate')
    
    parser.add_argument('--model_path',
                        type=str,
                        default='nu10000a1.pt',
                        help="Model path in .pt format")
    
    parser.add_argument('--data_path',
                        type=str,
                        default='burgers_data.mat',
                        help="Data path in .mat format")
    
    parser.add_argument('--t',
                        type=float,
                        default=0.25,
                        help='Timestamp to plot.')
    
    args = parser.parse_args()
    
    return args


@perf
def make_report(model_path: str,
                _input: torch.Tensor,
                gt: torch.Tensor,
                x: torch.Tensor,
                y: torch.Tensor,
                t: float = 0.25,
                device: torch.device = 'cpu') -> None:
    model2 = torch.load(model_path,  map_location=torch.device(device))
    nu_val = model_path.split('/')[-1].split('a')[0][2:]
    a_val = model_path.split('/')[-1].split('a')[1].split('.')[0]
    log.info(f'Fazendo report para nu = {nu_val} e a = {a_val}')
    net = SimpleModel()
    net.load_state_dict(model2)
    net = net.to(device)
    net.eval()
    pred = net(_input)
    mae = nn.L1Loss()(gt, pred)
    mse = nn.MSELoss()(gt, pred)
    log.info(f'MAE: {mae:.4f}')
    log.info(f'MSE: {mse:.4f}')
    
    max_pred = torch.max(pred)
    max_gt = torch.max(gt)
    min_pred = torch.min(pred)
    min_gt = torch.min(gt)
    log.info(f'Max pred value: {max_pred:.4f} | Max gt value: {max_gt:.4f}')
    log.info(f'Min pred value: {min_pred:.4f} | min gt value: {min_gt:.4f}\n')
    
    sol = pred[_input[:,2] == t]
    u = sol[:,0].reshape((101, 101))
    v = sol[:,1].reshape((101, 101))
    Y, X = np.meshgrid(y.squeeze(), x.squeeze())
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 8))
    ax.plot_surface(X, Y, u.cpu().detach().numpy())
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'plot for t = {t:.2f}')
    plt.savefig('img1.png', dpi=fig.dpi)
    plt.show()


def main():
    args = _get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Using device: {device}')
    
    data = scipy.io.loadmat(args.data_path)
    _t = data['t'].flatten()[:,None]
    _x = data['x'].flatten()[:,None]
    _y = data['y'].flatten()[:,None]
    u_sol = data['uref']
    v_sol = data['vref']
    
    X, Y, T = np.meshgrid(_x, _y, _t)
    vars_values = np.hstack((Y.flatten()[:,None], X.flatten()[:,None],
                             T.flatten()[:,None]))
    x_u = torch.tensor(vars_values[:, 1].reshape(-1, 1),
                                dtype=torch.float).to(device)
    y_u = torch.tensor(vars_values[:, 0].reshape(-1, 1),
                            dtype=torch.float).to(device)
    t_u = torch.tensor(vars_values[:, 2].reshape(-1, 1),
                            dtype=torch.float).to(device)
    _input = torch.hstack((x_u, y_u, t_u))
    gt = torch.tensor(
        np.hstack((u_sol.flatten()[:,None], v_sol.flatten()[:,None])))
    make_report(args.model_path, _input, gt, _x, _y, args.t, device)


if __name__=="__main__":
    main()
