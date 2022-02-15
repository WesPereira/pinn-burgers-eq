import time
import argparse
from pathlib import Path
import torch
import numpy as np
import scipy.io
from pyDOE import lhs
from pinn.burgers_2d_net import PinnBurgers2D
from pinn.utils.util import log, perf


def _get_args():
    parser = argparse.ArgumentParser(
        prog='Training step',
        usage='%(prog)s [options] parser',
        description='Parser for hyperparams training')
    
    parser.add_argument('--path',
                        type=str,
                        required=True,
                        help='Path to save model. Use .pt at end of path.')
    
    parser.add_argument('--nu',
                        type=int,
                        default=1000,
                        help='Number of pointzs of the simulation.')
    
    parser.add_argument('--nf',
                        type=int,
                        default=10000,
                        help='Number of points for f func.')
    
    parser.add_argument('--epochs',
                        type=int,
                        default=50000,
                        help='Number of epochs for training')
    
    parser.add_argument('--a',
                        type=float,
                        default=1,
                        help='Constant for balancing physics and pure NN.')
    
    parser.add_argument('--loss',
                        type=str,
                        default='mse',
                        help="Loss type. Values: ['mse', 'mae']")
    
    args = parser.parse_args()
    
    return args
    

@perf
def main():
    args = _get_args()
    
    log.info('Starting to modeling data...')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Using device: {device}')

    data = scipy.io.loadmat('./burger_data_2d.mat')

    nu = 0.01/np.pi

    N_u = args.nu
    N_f = args.nf
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]

    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    y = data['y'].flatten()[:,None]
    u_sol = np.real(data['uref'])
    v_sol = np.real(data['vref'])

    Y, X, T = np.meshgrid(y,x,t)

    vars_values = np.hstack((Y.flatten()[:,None], X.flatten()[:,None],
                             T.flatten()[:,None]))

    lb = vars_values.min(0)
    ub = vars_values.max(0)

    xx1 = np.vstack((Y[:,:,0:1].flatten(), X[:,:,0:1].flatten(),
                     T[:,:,0:1].flatten()))
    xx1 = np.moveaxis(xx1, -1, 0)

    xx2 = np.vstack((Y[:,0:1,:].flatten(), X[:,0:1,:].flatten(),
                     T[:,0:1,:].flatten()))
    xx2 = np.moveaxis(xx2, -1, 0)

    xx3 = np.vstack((Y[0:1,:,:].flatten(), X[0:1,:,:].flatten(),
                     T[0:1,:,:].flatten()))
    xx3 = np.moveaxis(xx3, -1, 0)
    
    xx4 = np.vstack((Y[-1:,:,:].flatten(), X[-1:,:,:].flatten(),
                     T[-1:,:,:].flatten()))
    xx4 = np.moveaxis(xx4, -1, 0)

    xx5 = np.vstack((Y[:,-1:,:].flatten(), X[:,-1:,:].flatten(),
                     T[:,-1:,:].flatten()))
    xx5 = np.moveaxis(xx5, -1, 0)
    
    pre_sol1_u = u_sol[:,:,0:1].flatten()
    pre_sol1_v = v_sol[:,:,0:1].flatten()
    sol1 = np.moveaxis(np.vstack((pre_sol1_u, pre_sol1_v)), 0, -1)

    pre_sol2_u = u_sol[:,0:1,:].flatten()
    pre_sol2_v = v_sol[:,0:1,:].flatten()
    sol2 = np.moveaxis(np.vstack((pre_sol2_u, pre_sol2_v)), 0, -1)

    pre_sol3_u = u_sol[0:1,:,:].flatten()
    pre_sol3_v = v_sol[0:1,:,:].flatten()
    sol3 = np.moveaxis(np.vstack((pre_sol3_u, pre_sol3_v)), 0, -1)

    pre_sol4_u = u_sol[-1:,:,:].flatten()
    pre_sol4_v = v_sol[-1:,:,:].flatten()
    sol4 = np.moveaxis(np.vstack((pre_sol4_u, pre_sol4_v)), 0, -1)

    pre_sol5_u = u_sol[:,-1:,:].flatten()
    pre_sol5_v = v_sol[:,-1:,:].flatten()
    sol5 = np.moveaxis(np.vstack((pre_sol5_u, pre_sol5_v)), 0, -1)

    vars_u_train = np.vstack([xx1, xx2, xx3, xx4, xx5])
    vars_f_train = lb + (ub-lb)*lhs(3, N_f)
    vars_f_train = np.vstack((vars_f_train, vars_u_train))
    u_train = np.vstack([sol1, sol2, sol3, sol4, sol5])

    idx = np.random.choice(vars_u_train.shape[0], N_u, replace=False)
    vars_u_train = vars_u_train[idx, :]
    u_train = u_train[idx,:]

    model = PinnBurgers2D(vars_u_train, u_train, vars_f_train,
                          nu, layers, device, args.a, args.epochs, args.loss)

    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time                
    log.info(f'Training time: {elapsed:.4f}s. Saving model at {args.path}...')

    save_path = Path(args.path)
    Path(str(save_path.parents[0])).mkdir(parents=True, exist_ok=True)
    torch.save(model.net.state_dict(), args.path)
    log.info('Finished training.')


if __name__=="__main__":
    main()
