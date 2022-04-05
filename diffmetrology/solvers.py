from .basics import *
import numpy as np
import torch.autograd.functional as F
from skimage.restoration import unwrap_phase


class Fringe(PrettyPrinter):
    """
    Fringe image analysis to resolve displacements.
    """
    def __init__(self):
        self.PHASE_SHIFT_COUNT = 4
        self.XY_COUNT = 2

    def solve(self, fs):
        """
        ----- old -----
        ref.shape = [len(Ts), self.XY_COUNT, camera_count, img_size]
        ----- old -----

        Outputs:
        a.shape = b.shape = p.shape = 
        [2, original size]

        where 2 denotes x and y.
        """
        def single(fs):
            ax, bx, psix = self._solve(fs[0:self.PHASE_SHIFT_COUNT])
            ay, by, psiy = self._solve(fs[self.PHASE_SHIFT_COUNT:self.XY_COUNT*self.PHASE_SHIFT_COUNT])
            return np.array([ax, ay]), np.array([bx, by]), np.array([psix, psiy])
        
        # TODO: following does not work when `len(Ts) == self.XY_COUNT*self.PHASE_SHIFT_COUNT` ...
        fsize = list(fs.shape)
        xy_index = fsize.index(self.XY_COUNT*self.PHASE_SHIFT_COUNT)
        inds = [i for i in range(len(fsize))]
        inds.remove(xy_index)
        inds = [xy_index] + inds
        
        # run the algorithm
        a, b, p = single(fs.transpose(inds))
        
        return a, b, p

    def unwrap(self, fs, Ts, valid=None):
        print('unwraping ...')
        if valid is None:
            valid = 1.0
        F = valid * fs
        fs_unwrapped = np.zeros(fs.shape)
        for xy in range(fs.shape[0]):
            print(f'xy = {xy} ...')
            for T in range(fs.shape[1]):
                print(f't = {T} ...')
                t = Ts[T]
                for i in range(fs.shape[2]):
                    fs_unwrapped[xy,T,i] = unwrap_phase(F[xy,T,i,...]) * t / (2*np.pi)
        return fs_unwrapped

    @staticmethod
    def _solve(fs):
        """Solver for four-step phase shifting: f(x) = a + b cos(\phi + \psi).
          b cos(\psi) = fs[0] - a
        - b sin(\psi) = fs[1] - a
        - b cos(\psi) = fs[2] - a
          b sin(\psi) = fs[3] - a
        """
        a = np.mean(fs, axis=0)
        b = 0.0
        for f in fs:
            b += (f - a)**2
        b = b/2.0
        psi = np.arctan2(fs[3] - fs[1], fs[0] - fs[2])
        return a, b, psi
    

class Optimization(PrettyPrinter):
    """
    General class for design optimization.
    """
    def __init__(self, diff_variables):
        self.diff_variables = diff_variables
        for v in self.diff_variables:
            v.requires_grad = True
        self.optimizer = None # to be initialized from set_parameters
    
class Adam(Optimization):
    def __init__(self, diff_variables, lr, lrs=None, beta=0.99):
        Optimization.__init__(self, diff_variables)
        if lrs is None:
            lrs = [1] * len(self.diff_variables)
        self.optimizer = torch.optim.Adam(
            [{"params": v, "lr": lr*l} for v, l in zip(self.diff_variables, lrs)],
            betas=(beta,0.999), amsgrad=True
        )

    def optimize(self, func, loss, maxit=300, record=True):
        print('optimizing ...')
        ls = []
        with torch.autograd.set_detect_anomaly(False): #True
            for it in range(maxit):
                L = loss(func())
                self.optimizer.zero_grad()
                L.backward(retain_graph=True)

                if record:
                    grads = torch.Tensor([torch.mean(torch.abs(v.grad)) for v in self.diff_variables])
                    print('iter = {}: loss = {:.4e}, grad_bar = {:.4e}'.format(
                        it, L.item(), torch.mean(grads)
                    ))
                    ls.append(L.cpu().detach().numpy())
                
                self.optimizer.step()

        return np.array(ls)


class LM(Optimization):
    """
    The Levenberg–Marquardt algorithm.
    """
    def __init__(self, diff_variables, lamb, mu=None, option='diag'):
        Optimization.__init__(self, diff_variables)
        self.lamb = lamb # damping factor
        self.mu = 2.0 if mu is None else mu # dampling rate (>1)
        self.option = option

    def jacobian(self, func, inputs, create_graph=False, strict=False):
        """Constructs a M-by-N Jacobian matrix where M >> N.

        Here, computing the Jacobian only makes sense for a tall Jacobian matrix. In this case,
        column-wise evaluation (forward-mode, or jvp) is more effective to construct the Jacobian.

        This function is modified from torch.autograd.functional.jvp().
        """

        Js = []
        outputs = func()
        M = outputs.shape

        grad_outputs = (torch.zeros_like(outputs, requires_grad=True),)
        for x in inputs:
            grad_inputs = F._autograd_grad(
                (outputs,), x, grad_outputs, create_graph=True
            )
            
            F._check_requires_grad(grad_inputs, "grad_inputs", strict=strict)

            # Construct Jacobian matrix
            N = torch.numel(x)
            if N == 1:
                J = F._autograd_grad(
                    grad_inputs, grad_outputs, (torch.ones_like(x),),
                    create_graph=create_graph,
                    retain_graph=True
                )[0][...,None]
            else:
                J = torch.zeros((*M, N), device=x.device)
                v = torch.zeros(N, device=x.device)
                for i in range(N):
                    v[i] = 1.0
                    J[...,i] = F._autograd_grad(
                        grad_inputs, grad_outputs, (v.view(x.shape),),
                        create_graph=create_graph,
                        retain_graph=True
                    )[0]

                    v[i] = 0.0
            Js.append(J)
        return torch.cat(Js, axis=-1)
        
    def optimize(self, func, change_parameters, func_yref_y, maxit=300, record=True):
        """
        Inputs:
        - func: Evaluate `y = f(x)` where `x` is the implicit parameters by `self.diff_variables` (out of the class)
        - change_parameters: Change of `self.diff_variables` (out of the class)
        - func_yref_y: Compute `y_ref - y`

        Outputs:
        - ls: Loss function.
        """
        print('optimizing ...')
        Ns = [x.numel() for x in self.diff_variables]
        NS = [[*x.shape] for x in self.diff_variables]

        ls = []
        lamb = self.lamb
        with torch.autograd.set_detect_anomaly(False):
            for it in range(maxit):
                y = func()
                with torch.no_grad():
                    L = torch.mean(func_yref_y(y)**2).item()
                    if L < 1e-16:
                        print('L too small; termiante.')
                        break

                # Obtain Jacobian
                J = self.jacobian(func, self.diff_variables, create_graph=False)
                J = J.view(-1, J.shape[-1])
                JtJ = J.T @ J
                N = JtJ.shape[0]
                
                # Regularization matrix
                if self.option == 'I':
                    R = torch.eye(N, device=JtJ.device)
                elif self.option == 'diag':
                    R = torch.diag(torch.diag(JtJ).abs())
                else:
                    R = torch.diag(self.option)
                
                # Compute b = J.T @ (y_ref - y)
                bb = [
                    torch.autograd.grad(outputs=y, inputs=x, grad_outputs=func_yref_y(y), retain_graph=True)[0]
                    for x in self.diff_variables
                ]
                for i, bx in enumerate(bb):
                    if len(bx.shape) == 0: # single scalar
                        bb[i] = torch.Tensor([bx.item()]).to(y.device)
                    if len(bx.shape) > 1: # multi-dimension
                        bb[i] = torch.Tensor(bx.cpu().detach().numpy().flatten()).to(y.device)
                b = torch.cat(bb, axis=-1)
                del J, bb, y

                # Damping loop
                L_current = L + 1.0
                it_inner = 0
                while L_current >= L:
                    it_inner += 1
                    if it_inner > 20:
                        print('inner loop too many; Exiting damping loop.')
                        break

                    A = JtJ + lamb * R
                    x_delta = torch.linalg.solve(A, b)
                    if torch.isnan(x_delta).sum():
                        print('x_delta NaN; Exiting damping loop')
                        break
                    x_delta_s = torch.split(x_delta, Ns)

                    # Reshape if x is not a 1D array
                    x_delta_s = [*x_delta_s]
                    for xi in range(len(x_delta_s)):
                        x_delta_s[xi] = torch.reshape(x_delta_s[xi],  NS[xi])
                    
                    # Update `x += x_delta` (this is done in external function `change_parameters`)
                    self.diff_variables = change_parameters(x_delta_s, sign=True)

                    # Calculate new error
                    with torch.no_grad():
                        L_current = torch.mean(func_yref_y(func())**2).item()

                    del A

                    # Terminate
                    if L_current < L:
                        lamb /= self.mu
                        del x_delta_s
                        break

                    # Else, increase damping and undo the update
                    lamb *= 2.0*self.mu
                    self.diff_variables = change_parameters(x_delta_s, sign=False)
                    
                    if lamb > 1e16:
                        print('lambda too big; Exiting damping loop.')
                        del x_delta_s
                        break

                del JtJ, R, b

                if record:
                    x_increment = torch.mean(torch.abs(x_delta)).item()
                    print('iter = {}: loss = {:.4e}, |x_delta| = {:.4e}'.format(
                        it, L, x_increment
                    ))
                    ls.append(L)
                    if it > 0:
                        dls = np.abs(ls[-2] - L)
                        if dls < 1e-8:
                            print("|\Delta loss| = {:.4e} < 1e-8; Exiting LM loop.".format(dls))
                            break

                    if x_increment < 1e-8:
                        print("|x_delta| = {:.4e} < 1e-8; Exiting LM loop.".format(x_increment))
                        break
        return ls
