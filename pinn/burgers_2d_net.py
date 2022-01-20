from typing import List, Tuple
import torch
import torch.nn as nn
import numpy as np
from pinn.utils.util import log, perf


class SimpleModel(nn.Module):
    
    def __init__(
        self, layers: List[int] = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
    ) -> None:
        """
        Model init.

        Args:
            layers (List[int], optional): Layers structure. 
            Defaults to [3, 20, 20, 20, 20, 20, 20, 20, 20, 2].
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
    def __init__(self, X_u: np.ndarray, u: np.ndarray, v: np.ndarray,
                 X_f: np.ndarray, nu: float, layers: List[int]) -> None:
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
        
        self.v = torch.tensor(v, dtype=torch.float)

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

    def net_u_and_v(self, x: torch.Tensor, y: torch.Tensor,
              t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Represents u function at Burgers equation.

        Args:
            x (torch.Tensor): spatial data.
            y (torch.Tensor): spatial data.
            t (torch.Tensor): time data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: u and v output tensors.
        """
        u, v = self.net(torch.hstack((x, y, t)))
        return u, v

    def net_f(self, x: torch.Tensor, y: torch.Tensor,
              t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Represents f function at Burgers equation.

        Args:
            x (torch.Tensor): spatial data.
            y (torch.Tensor): spatial data.
            t (torch.Tensor): time data.

        Returns:
            torch.Tensor: f output tensor.
        """
        u, v = self.net_u_and_v(x, y, t)

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v),
                                  retain_graph=True, create_graph=True)[0]

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v),
                                  retain_graph=True, create_graph=True)[0]
        
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                                  retain_graph=True, create_graph=True)[0]
        
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v),
                                  retain_graph=True, create_graph=True)[0]

        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                   retain_graph=True, create_graph=True)[0]
        
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x),
                                   retain_graph=True, create_graph=True)[0]
        
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y),
                                   retain_graph=True, create_graph=True)[0]
        
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y),
                                   retain_graph=True, create_graph=True)[0]

        f_one = u_t + u*u_x - v*u_y - self.nu*(u_xx + u_yy)
        f_two = v_t + u*v_x - v*v_y - self.nu*(v_xx + v_yy)

        return f_one, f_two

    def closure_func(self) -> torch.nn:
        """
        Closure function.
        
        Returns:
            torch.nn: loss tensor
        """
        self.optimizer.zero_grad()
        u_pred, v_pred = self.net_u_and_v(self.x_u, self.y_u, self.t_u)
        f_pred_one, f_pred_two = self.net_f(self.x_f, self.y_f, self.t_f)

        u_loss = self.loss(u_pred, self.u)
        v_loss = self.loss(v_pred, self.v)
        f_loss_one = self.loss(f_pred_one, self.zeros_t)
        f_loss_two = self.loss(f_pred_two, self.zeros_t)
        self.ls = u_loss + v_loss + f_loss_one + f_loss_two

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
