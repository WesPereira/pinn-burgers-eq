from typing import List
import torch
import torch.nn as nn
import numpy as np
from pinn.utils.util import log, perf


class SimpleModel(nn.Module):
    
    def __init__(
        self, layers: List[int] = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    ) -> None:
        """
        Model init.

        Args:
            layers (List[int], optional): Layers structure. 
            Defaults to [2, 20, 20, 20, 20, 20, 20, 20, 20, 1].
        """
        super().__init__()
        
        modules = []
        for _in, _out in list(zip(layers, layers[1:])):
            modules.append(nn.Linear(_in, _out))
            modules.append(nn.Tanh())
        
        # Remove last tanh
        modules.pop()
        self.model = nn.Sequential(*modules)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method.

        Args:
            x (torch.Tensor): input.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)


class PinnBurgers1D:
    def __init__(self, X_u: np.ndarray, u: np.ndarray, X_f: np.ndarray,
                 nu: float, layers: List[int]) -> None:
        """
        Init function.

        Args:
            X_u (np.ndarray): X_u array.
            u (np.ndarray): u array.
            X_f (np.ndarray): X_f array.
            nu (float): nu condition float.
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

        self.net = SimpleModel(layers)

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

    def net_u(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Represents u function at Burgers equation.

        Args:
            x (torch.Tensor): spatial data.
            t (torch.Tensor): time data.

        Returns:
            torch.Tensor: u output tensor.
        """
        u = self.net(torch.hstack((x, t)))
        return u

    def net_f(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Represents f function at Burgers equation.

        Args:
            x (torch.Tensor): spatial data.
            t (torch.Tensor): time data.

        Returns:
            torch.Tensor: f output tensor.
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

    @perf
    def train(self):
        """
        Train method
        """
        self.net.train()
        self.optimizer.step(self.closure_func)
