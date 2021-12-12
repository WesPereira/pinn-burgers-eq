import pickle
import numpy as np
from pinn.burgers_1d_net import PinnBurgers1D
from pinn.utils.util import log, perf
from pinn.utils.plot import newfig, pgf_with_latex, savefig
from pyDOE import lhs
import fire
from scipy.interpolate import griddata
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _plot(result: torch.Tensor, X_star: np.ndarray, X: np.ndarray,
         T: np.ndarray, t: np.ndarray, x: np.ndarray,
         X_u_train: np.ndarray, u_train: np.ndarray, Exact: np.ndarray) -> None:
    
    mpl.rcParams.update(pgf_with_latex)
    u_pred = result.detach().numpy()
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    fig, ax = newfig(3.0, 1.1)
    ax.axis('off')

    ####### Row 0: u(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
                    extent=[t.min(), t.max(), x.min(), x.max()], 
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx',
            label = 'Data (%d points)' % (u_train.shape[0]),
            markersize = 4, clip_on = False)

    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)    

    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('u(t,x)', fontsize = 10)

    ####### Row 1: u(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, Exact[25,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x, U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('u(t,x)')    
    ax.set_title('t = 0.25', fontsize = 10)
    ax.axis('square')

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact[50,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('u(t,x)')
    ax.axis('square')
    ax.set_title('t = 0.50', fontsize = 10)
    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x, Exact[75,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x, U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('u(t,x)')
    ax.axis('square')
    ax.set_title('t = 0.75', fontsize = 10)
    savefig(fig, 'plots.png')
    plt.show()


@perf
def train(save_path: str):
    log.info('Starting to load data...')
    
    with open('data_burgers.pkl', 'rb') as f:
        data = pickle.load(f)
    
    nu = .07

    N_u = 100
    N_f = 10000
    
    # MLP structure
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    
    # transform (x, t) -> (t, x)
    Exact = np.real(data['usol']).T

    X, T = np.meshgrid(x,t)

    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]              

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)    

    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
    uu1 = Exact[0:1,:].T
    xx2 = np.hstack((X[:,0:1], T[:,0:1]))
    uu2 = Exact[:,0:1]
    xx3 = np.hstack((X[:,-1:], T[:,-1:]))
    uu3 = Exact[:,-1:]

    X_u_train = np.vstack([xx1, xx2, xx3])
    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.vstack([uu1, uu2, uu3])

    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx,:]

    log.info('Finish to load data and prep data...')

    pinn = PinnBurgers1D(X_u_train, u_train, X_f_train, nu, layers)

    log.info('Starting training...')
    pinn.train()
    log.info('Finish train.')
    torch.save(pinn.net.state_dict(), save_path)
    log.info(f'Model saved at {save_path}.')
    
    log.info('Starting the plot...')
    result = pinn.net(torch.from_numpy(X_star).type(torch.FloatTensor))
    _plot(result, X_star, X, T, t, x, X_f_train, u_train, Exact)
    log.info('Finish ploting')


if __name__ == '__main__' :
    fire.Fire(train)
