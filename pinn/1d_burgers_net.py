from typing import List
import torch
import torch.nn as nn
import numpy as np
from pinn.utils import log


class SimpleModel(nn.Module):
    
    def __init__(
        self, layers: List[int] = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    ) -> None:
        """[summary]

        Args:
            layers (List[int], optional): [description]. 
            Defaults to [2, 20, 20, 20, 20, 20, 20, 20, 20, 1].

        Returns:
            [type]: [description]
        """
        super().__init__()
        
        modules = []
        for _in, _out in list(zip(layers, layers[1:])):
            modules.append(nn.Linear(_in, _out))
            modules.append(nn.Tanh())
        
        # Remove last tanh
        modules.pop()
        self.model = nn.Sequential(*modules)
    
    def forward(self, x):
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        return self.model(x)


class PinnBurgers1D:
    def __init__(self, X_u: np.ndarray, u: np.ndarray, X_f, nu: float) -> None:
        """[summary]

        Args:
            X_u (np.ndarray): [description]
            u (np.ndarray): [description]
            X_f ([type]): [description]
            nu (float): [description]
        """
        self.nu = nu
        self.x_u = torch.tensor(X_u[:, 0].reshape(-1, 1),
                                dtype=torch.float,
                                requires_grad=True)
        self.t_u = torch.tensor(X_u[:, 1].reshape(-1, 1),
                                dtype=torch.float,
                                requires_grad=True)
        self.x_f = torch.tensor(X_f[:, 0].reshape(-1, 1),
                                dtype=torch.float,
                                requires_grad=True)
        self.t_f = torch.tensor(X_f[:, 1].reshape(-1, 1),
                                dtype=torch.float,
                                requires_grad=True)

        self.u = torch.tensor(u, dtype=torch.float)

        self.zeros_t =  torch.zeros((self.x_f.shape[0], 1))

        self.net = SimpleModel()
        self.net.apply(self.init_weights)

        self.loss = nn.MSELoss()

        self.optimizer = torch.optim.LBFGS(self.net.parameters(),
                                    lr=1,
                                    max_iter=50000,
                                    max_eval=50000,
                                    history_size=50,
                                    tolerance_grad=1e-05,
                                    tolerance_change=0.5 * np.finfo(float).eps,
                                    line_search_fn="strong_wolfe")

        self.ls = 0
        self.iter = 0

    def init_weights(self, m: torch.nn) -> None:
        """[summary]

        Args:
            m (torch.nn): [description]

        Returns:
            [type]: [description]
        """
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight, 0.1)
            m.bias.data.fill_(0.001)

    def net_u(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """[summary]

        Args:
            x (torch.Tensor): [description]
            t (torch.Tensor): [description]

        Returns:
            torch.Tensor: [description]
        """
        u = self.net(torch.hstack((x, t)))
        return u

    def net_f(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """[summary]

        Args:
            x (torch.Tensor): [description]
            t (torch.Tensor): [description]

        Returns:
            torch.Tensor: [description]
        """
        u = self.net_u(x, t)

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]

        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                   retain_graph=True, create_graph=True)[0]

        f = u_t + (u * u_x) - (self.nu * u_xx)

        return f

    def closure_func(self) -> torch.nn:
        """
        Closure function.
        
        Returns:
            torch.nn: loss tensor
        """
        self.optimizer.zero_grad()
        u_pred = self.net_u(self.x_u, self.t_u)
        f_pred = self.net_f(self.x_f, self.t_f)

        u_loss = self.loss(u_pred, self.u)
        f_loss = self.loss(f_pred, self.zeros_t)
        self.ls = u_loss + f_loss

        self.ls.backward()

        self.iter += 1

        if self.iter % 50 == 0:
            log.info(f'Epoch: {self.iter}, Loss: {self.ls:6.3f}')

        return self.ls

    def train(self):
        """
        Train method
        """
        self.net.train()
        self.optimizer.step(self.closure_func)
