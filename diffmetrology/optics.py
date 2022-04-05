from .basics import *
from .shapes import *
from scipy.interpolate import LSQBivariateSpline
from datetime import datetime
import matplotlib.pyplot as plt


class Step(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, eps):
        ctx.constant = eps
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * torch.exp(-(ctx.constant*input)**2), None

def ind(x, eps=0.5):
    return Step.apply(x, eps)


class Lensgroup(Endpoint):
    """
    The Lensgroup (consisted of multiple optical surfaces) is mounted on a rod, whose
    origin is `origin`. The Lensgroup has full degree-of-freedom to rotate around the
    x/y axes, with the rotation angles defined as `theta_x`, `theta_y`, and `theta_z` (in degree).

    In Lensgroup's coordinate (i.e. object frame coordinate), surfaces are allocated
    starting from `z = 0`. There is an additional, comparatively small 3D origin shift
    (`shift`) between the surface center (0,0,0) and the origin of the mount, i.e.
    shift + origin = lensgroup_origin.
    
    There are two configurations of ray tracing: forward and backward. In forward mode,
    rays start from `d = 0` surface and propagate along the +z axis; In backward mode,
    rays start from `d = d_max` surface and propagate along the -z axis.
    """
    def __init__(self, origin, shift, theta_x=0., theta_y=0., theta_z=0., device=torch.device('cpu')):
        self.origin = torch.Tensor(origin).to(device)
        self.shift = torch.Tensor(shift).to(device)
        self.theta_x = torch.Tensor(np.asarray(theta_x)).to(device)
        self.theta_y = torch.Tensor(np.asarray(theta_y)).to(device)
        self.theta_z = torch.Tensor(np.asarray(theta_z)).to(device)
        self.device = device

        Endpoint.__init__(self, self._compute_transformation(), device)

        # TODO: in case you would like to render something ...
        self.mts_prepared = False

    def load_file(self, filename):
        LENSPATH = './lenses/'
        filename = filename if filename[0] == '.' else LENSPATH + filename
        self.surfaces, self.materials, self.r_last, d_last = self.read_lensfile(filename)
        self.d_sensor = d_last + self.surfaces[-1].d
        self._sync()

    def load(self, surfaces, materials):
        self.surfaces = surfaces
        self.materials = materials
        self._sync()

    def _sync(self):
        for i in range(len(self.surfaces)):
            self.surfaces[i].to(self.device)

    def update(self, _x=0.0, _y=0.0):
        self.to_world = self._compute_transformation(_x, _y)
        self.to_object = self.to_world.inverse()
    
    def _compute_transformation(self, _x=0.0, _y=0.0, _z=0.0):
        # we compute to_world transformation given the input positional parameters (angles)
        R = ( rodrigues_rotation_matrix(torch.Tensor([1, 0, 0]).to(self.device), torch.deg2rad(self.theta_x+_x)) @ 
              rodrigues_rotation_matrix(torch.Tensor([0, 1, 0]).to(self.device), torch.deg2rad(self.theta_y+_y)) @ 
              rodrigues_rotation_matrix(torch.Tensor([0, 0, 1]).to(self.device), torch.deg2rad(self.theta_z+_z)) )
        t = self.origin + R @ self.shift
        return Transformation(R, t)

    @staticmethod
    def read_lensfile(filename):
        surfaces = []
        materials = []
        ds = [] # no use for now
        with open(filename) as file:
            line_no = 0
            d_total = 0.
            for line in file:
                if line_no < 2: # first two lines are comments; ignore them
                    line_no += 1 
                else:
                    ls = line.split()
                    surface_type, d, r = ls[0], float(ls[1]), float(ls[3])/2
                    roc = float(ls[2])
                    if roc != 0: roc = 1/roc
                    materials.append(Material(ls[4]))
                    
                    d_total += d
                    ds.append(d)

                    if surface_type == 'O': # object
                        d_total = 0.
                        ds.pop()
                    elif surface_type == 'X': # XY-polynomial
                        del roc
                        ai = []
                        for ac in range(5, len(ls)):
                            if ac == 5:
                                b = float(ls[5])
                            else:
                                ai.append(float(ls[ac]))
                        surfaces.append(XYPolynomial(r, d_total, J=2, ai=ai, b=b))
                    elif surface_type == 'B': # B-spline
                        del roc
                        ai = []
                        for ac in range(5, len(ls)):
                            if ac == 5:
                                nx = int(ls[5])
                            elif ac == 6:
                                ny = int(ls[6])
                            else:
                                ai.append(float(ls[ac]))
                        tx = ai[:nx+8]
                        ai = ai[nx+8:]
                        ty = ai[:ny+8]
                        ai = ai[ny+8:]
                        c  = ai
                        surfaces.append(BSpline(r, d, size=[nx, ny], tx=tx, ty=ty, c=c))
                    elif surface_type == 'M': # mixed-type of X and B
                        raise NotImplementedError()
                    elif surface_type == 'S': # aspheric surface
                        if len(ls) <= 5:
                            surfaces.append(Aspheric(r, d_total, roc))
                        else:
                            ai = []
                            for ac in range(5, len(ls)):
                                if ac == 5:
                                    conic = float(ls[5])
                                else:
                                    ai.append(float(ls[ac]))
                            surfaces.append(Aspheric(r, d_total, roc, conic, ai))
                    elif surface_type == 'A': # aperture
                        surfaces.append(Aspheric(r, d_total, roc))
                    elif surface_type == 'I': # sensor
                        d_total -= d
                        ds.pop()
                        materials.pop()
                        r_last = r
                        d_last = d
        return surfaces, materials, r_last, d_last

    def reverse(self):
        # reverse surfaces
        d_total = self.surfaces[-1].d
        for i in range(len(self.surfaces)):
            self.surfaces[i].d = d_total - self.surfaces[i].d
            self.surfaces[i].reverse()
        self.surfaces.reverse()
        
        # reverse materials
        self.materials.reverse()

    # ------------------------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------------------------
    def rms(self, ps, units=1e3, option='centroid'):
        ps = ps[...,:2] * units
        if option == 'centroid':
            ps_mean = torch.mean(ps, axis=0) # centroid
        ps = ps - ps_mean[None,...] # we now use normalized ps
        spot_rms = torch.sqrt(torch.mean(torch.sum(ps**2, axis=-1)))
        return spot_rms

    def spot_diagram(self, ps, show=True, xlims=None, ylims=None, color='b.'):
        """
        Plot spot diagram.
        """
        units = 1e3
        units_str = '[um]'
        # units = 1
        # units_str = '[mm]'
        spot_rms = float(self.rms(ps, units))
        ps = ps.cpu().detach().numpy()[...,:2] * units
        ps_mean = np.mean(ps, axis=0) # centroid
        ps = ps - ps_mean[None,...] # we now use normalized ps
        
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(ps[...,1], ps[...,0], color) # permute axe 0 and 1
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('x ' + units_str)
        plt.ylabel('y ' + units_str)
        plt.title("Spot diagram, RMS = " + str(round(spot_rms,3)) + ' ' + units_str)
        if xlims is not None:
            plt.xlim(*xlims)
        if ylims is not None:
            plt.ylim(*ylims)
        ax.set_aspect(1./ax.get_data_ratio())

        fig.savefig("spotdiagram_" + datetime.now().strftime('%Y%m%d-%H%M%S-%f') + ".pdf", bbox_inches='tight')
        if show: plt.show()
        else: plt.close()
        return spot_rms
    
    # ------------------------------------------------------------------------------------
    
    # ------------------------------------------------------------------------------------
    # IO and visualizations
    # ------------------------------------------------------------------------------------
    def draw_points(self, ax, options, seq=range(3)):
        for surface in self.surfaces:
            points_world = self._generate_points(surface)
            ax.plot(points_world[seq[0]], points_world[seq[1]], points_world[seq[2]], options)

    # ------------------------------------------------------------------------------------
    
    # ------------------------------------------------------------------------------------
    
    def trace(self, ray, stop_ind=None):
        # update transformation when doing pose estimation
        if (
            self.origin.requires_grad or self.shift.requires_grad
            or
            self.theta_x.requires_grad or self.theta_y.requires_grad or self.theta_z.requires_grad
        ):
            self.update()

        # in local
        ray_in = self.to_object.transform_ray(ray)
        valid, mask_g, ray_out = self._trace(ray_in, stop_ind=stop_ind, record=False)
        
        # in world
        ray_final = self.to_world.transform_ray(ray_out)

        return ray_final, valid, mask_g

    # ------------------------------------------------------------------------------------
    
    def _refract(self, wi, n, eta, approx=False):
        """
        Snell's law (surface normal n defined along the positive z axis).
        """
        if np.prod(eta.shape) > 1:
            eta_ = eta[..., None]
        else:
            eta_ = eta
        
        cosi = torch.sum(wi * n, axis=-1)

        if approx:
            tmp = 1. - eta**2 * (1. - cosi)
            g = tmp
            valid = tmp > 0.
            wt = tmp[..., None] * n + eta_ * (wi - cosi[..., None] * n)
        else:
            cost2 = 1. - (1. - cosi**2) * eta**2
            
            # 1. get valid map; 2. zero out invalid points; 3. add eps to avoid NaN grad at cost2==0.
            g = cost2
            valid = cost2 > 0.
            cost2 = torch.clamp(cost2, min=1e-8)
            tmp = torch.sqrt(cost2)

            wt = tmp[..., None] * n + eta_ * (wi - cosi[..., None] * n)
        return valid, wt, g

    def _trace(self, ray, stop_ind=None, record=False):
        if stop_ind is None:
            stop_ind = len(self.surfaces)-1  # last index to stop
        is_forward = (ray.d[..., 2] > 0).all()

        if is_forward:
            return self._forward_tracing(ray, stop_ind, record)
        else:
            return self._backward_tracing(ray, stop_ind, record)

    def _forward_tracing(self, ray, stop_ind, record):
        wavelength = ray.wavelength
        dim = ray.o[..., 2].shape
        
        if record:
            oss = []
            for i in range(dim[0]):
                oss.append([ray.o[i,:].cpu().detach().numpy()])

        valid = torch.ones(dim, device=self.device).bool()
        mask  = torch.ones(dim, device=self.device)
        for i in range(stop_ind+1):
            eta = self.materials[i].ior(wavelength) / self.materials[i+1].ior(wavelength)

            # ray intersecting surface
            valid_o, p, g_o = self.surfaces[i].ray_surface_intersection(ray, valid)

            # get surface normal and refract 
            n = self.surfaces[i].normal(p[..., 0], p[..., 1])
            valid_d, d, g_d = self._refract(ray.d, -n, eta)
            
            # check validity
            mask = mask * ind(g_o) * ind(g_d)
            valid = valid & valid_o & valid_d
            if not valid.any():
                break

            # update ray {o,d}
            if record:
                for os, v, pp in zip(oss, valid.cpu().detach().numpy(), p.cpu().detach().numpy()):
                    if v: os.append(pp)
            ray.o = p
            ray.d = d
        
        if record:
            return valid, mask, ray, oss
        else:
            return valid, mask, ray
        
    def _backward_tracing(self, ray, stop_ind, record):
        wavelength = ray.wavelength
        dim = ray.o[..., 2].shape
        
        if record:
            oss = []
            for i in range(dim[0]):
                oss.append([ray.o[i,:].cpu().detach().numpy()])

        valid = torch.ones(dim, device=ray.o.device).bool()
        mask  = torch.ones(dim, device=ray.o.device)
        for i in np.flip(range(stop_ind+1)):
            surface = self.surfaces[i]
            eta = self.materials[i+1].ior(wavelength) / self.materials[i].ior(wavelength)

            # ray intersecting surface
            valid_o, p, g_o = surface.ray_surface_intersection(ray, valid)

            # get surface normal and refract 
            n = surface.normal(p[..., 0], p[..., 1])
            valid_d, d, g_d = self._refract(ray.d, n, eta)  # backward: no need to revert the normal

            # check validity
            mask = mask * ind(g_o) * ind(g_d)
            valid = valid & valid_o & valid_d
            if not valid.any():
                break

            # update ray {o,d}
            if record:
                for os, v, pp in zip(oss, valid.numpy(), p.cpu().detach().numpy()):
                    if v: os.append(pp)
            ray.o = p
            ray.d = d

        if record:
            return valid, mask, ray, oss
        else:
            return valid, mask, ray

    def _generate_points(self, surface, with_boundary=False):
        R = surface.r
        x = y = torch.linspace(-R, R, surface.APERTURE_SAMPLING, device=self.device)
        X, Y = torch.meshgrid(x, y)
        Z = surface.surface_with_offset(X, Y)
        valid = X**2 + Y**2 <= R**2
        if with_boundary:
            from scipy import ndimage
            tmp = ndimage.convolve(valid.cpu().numpy().astype('float'), np.array([[0,1,0],[1,0,1],[0,1,0]]))
            boundary = valid.cpu().numpy() & (tmp != 4)
            boundary = boundary[valid.cpu().numpy()].flatten()
        points_local = torch.stack(tuple(v[valid].flatten() for v in [X, Y, Z]), axis=-1)
        points_world = self.to_world.transform_point(points_local).T.cpu().detach().numpy()
        if with_boundary:
            return points_world, boundary
        else:
            return points_world

class Surface(PrettyPrinter):
    def __init__(self, r, d, device=torch.device('cpu')):
        # self.r = torch.Tensor(np.array(r))
        if torch.is_tensor(d):
            self.d = d
        else:
            self.d = torch.Tensor(np.asarray(float(d))).to(device)
        self.r = float(r)
        self.device = device
        self.NEWTONS_MAXITER = 10
        self.NEWTONS_TOLERANCE_TIGHT = 50e-6 # in [mm], i.e. 50 [nm] here (up to <10 [nm])
        self.NEWTONS_TOLERANCE_LOOSE = 300e-6 # in [mm], i.e. 300 [nm] here (up to <10 [nm])
        self.APERTURE_SAMPLING = 11
    
    # === Common methods (must not be overridden)
    def surface_with_offset(self, x, y):
        return self.surface(x, y) + self.d
    
    def normal(self, x, y):
        ds_dxyz = self.surface_derivatives(x, y)
        return normalize(torch.stack(ds_dxyz, axis=-1))

    def surface_area(self):
        return math.pi * self.r**2

    def ray_surface_intersection(self, ray, active=None):
        """
        Returns:
        - g >= 0: valid or not
        - p: intersection point
        - g: explicit funciton
        """
        solution_found, local = self.newtons_method(ray.maxt, ray.o, ray.d)
        r2 = local[..., 0]**2 + local[..., 1]**2
        g = self.r**2 - r2
        if active is None:
            valid_o = solution_found & ind(g > 0.).bool()
        else:
            valid_o = active & solution_found & ind(g > 0.).bool()
        return valid_o, local, g

    def newtons_method_impl(self, maxt, t0, dx, dy, dz, ox, oy, oz, A, B, C):
        t_delta = torch.zeros_like(oz)

        # Iterate until the intersection error is small
        t        = maxt * torch.ones_like(oz)
        residual = maxt * torch.ones_like(oz)
        it = 0
        while (torch.abs(residual) > self.NEWTONS_TOLERANCE_TIGHT).any() and (it < self.NEWTONS_MAXITER):
            it += 1
            t = t0 + t_delta
            residual, s_derivatives_dot_D = self.surface_and_derivatives_dot_D(
                t, dx, dy, dz, ox, oy, t_delta * dz, A, B, C # here z = t_delta * dz
            )
            t_delta -= residual / s_derivatives_dot_D
        t = t0 + t_delta
        valid = (torch.abs(residual) < self.NEWTONS_TOLERANCE_LOOSE) & (t <= maxt)
        return t, t_delta, valid

    def newtons_method(self, maxt, o, D, option='implicit'):
        # Newton's method to find the root of the ray-surface intersection point.
        # Two modes are supported here:
        # 
        # 1. 'explicit": This implements the loop using autodiff, and gradients will be
        # accurate for o, D, and self.parameters. Slow and memory-consuming.
        # 
        # 2. 'implicit": This implements the loop as proposed in the paper, it finds the 
        # solution without autodiff, then hook up the gradient. Less memory consumption.

        # pre-compute constants
        ox, oy, oz = (o[..., i].clone() for i in range(3))
        dx, dy, dz = (D[..., i].clone() for i in range(3))
        A = dx**2 + dy**2
        B = 2 * (dx * ox + dy * oy)
        C = ox**2 + oy**2

        # initial guess of t
        t0 = (self.d - oz) / dz
        
        if option == 'explicit':
            t, t_delta, valid = self.newtons_method_impl(
                maxt, t0, dx, dy, dz, ox, oy, oz, A, B, C
            )
        elif option == 'implicit':
            with torch.no_grad():
                t, t_delta, valid = self.newtons_method_impl(
                    maxt, t0, dx, dy, dz, ox, oy, oz, A, B, C
                )
                s_derivatives_dot_D = self.surface_and_derivatives_dot_D(
                    t, dx, dy, dz, ox, oy, t_delta * dz, A, B, C
                )[1]
            t = t0 + t_delta # re-engage autodiff

            t = t - (self.g(ox + t * dx, oy + t * dy) + self.h(oz + t * dz) + self.d)/s_derivatives_dot_D
        else:
            raise Exception('option={} is not available!'.format(option))

        p = o + t[..., None] * D
        return valid, p

    # === Virtual methods (must be overridden)
    def g(self, x, y):
        raise NotImplementedError()

    def dgd(self, x, y):
        """
        Derivatives of g: (g'x, g'y).
        """
        raise NotImplementedError()

    def h(self, z):
        raise NotImplementedError()

    def dhd(self, z):
        """
        Derivative of h.
        """
        raise NotImplementedError()

    def surface(self, x, y):
        """
        Solve z from h(z) = -g(x,y).
        """
        raise NotImplementedError()

    def reverse(self):
        raise NotImplementedError()

    # === Default methods (better be overridden)
    def surface_derivatives(self, x, y):
        """
        Returns \nabla f = \nabla (g(x,y) + h(z)) = (dg/dx, dg/dy, dh/dz).
        (Note: this default implementation is not efficient)
        """
        gx, gy = self.dgd(x, y)
        z = self.surface(x, y)
        return gx, gy, self.dhd(z)
        
    def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, z, A, B, C):
        """
        Returns g(x,y)+h(z) and dot((g'x,g'y,h'), (dx,dy,dz)).
        (Note: this default implementation is not efficient)
        """
        x = ox + t * dx
        y = oy + t * dy
        s = self.g(x,y) + self.h(z)
        sx, sy = self.dgd(x, y)
        sz = self.dhd(z)
        return s, sx*dx + sy*dy + sz*dz


class Aspheric(Surface):
    """
    Aspheric surface: https://en.wikipedia.org/wiki/Aspheric_lens.
    """
    def __init__(self, r, d, c=0., k=0., ai=None, device=torch.device('cpu')):
        Surface.__init__(self, r, d, device)
        self.c, self.k = (torch.Tensor(np.array(v)) for v in [c, k])
        self.ai = None
        if ai is not None:
            self.ai = torch.Tensor(np.array(ai))

    # === Common methods
    def g(self, x, y):
        return self._g(x**2 + y**2)

    def dgd(self, x, y):
        dsdr2 = 2 * self._dgd(x**2 + y**2)
        return dsdr2*x, dsdr2*y

    def h(self, z):
        return -z

    def dhd(self, z):
        return -torch.ones_like(z)

    def surface(self, x, y):
        return self._g(x**2 + y**2)

    def reverse(self):
        self.c = -self.c
        if self.ai is not None:
            self.ai = -self.ai

    def surface_derivatives(self, x, y):
        dsdr2 = 2 * self._dgd(x**2 + y**2)
        return dsdr2*x, dsdr2*y, -torch.ones_like(x)

    def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, z, A, B, C):
        r2 = A * t**2 + B * t + C
        return self._g(r2) - z, self._dgd(r2) * (2*A*t + B) - dz
    
    # === Private methods
    def _g(self, r2):
        tmp = r2*self.c
        total_surface = tmp / (1 + torch.sqrt(1 - (1+self.k) * tmp*self.c))
        higher_surface = 0
        if self.ai is not None:
            for i in np.flip(range(len(self.ai))):
                higher_surface = r2 * higher_surface + self.ai[i]
            higher_surface = higher_surface * r2**2
        return total_surface + higher_surface
    
    def _dgd(self, r2):
        alpha_r2 = (1 + self.k) * self.c**2 * r2
        tmp = torch.sqrt(1 - alpha_r2) # TODO: potential NaN grad
        total_derivative = self.c * (1 + tmp - 0.5*alpha_r2) / (tmp * (1 + tmp)**2)

        higher_derivative = 0
        if self.ai is not None:
            for i in np.flip(range(len(self.ai))):
                higher_derivative = r2 * higher_derivative + (i+2) * self.ai[i]
        return total_derivative + higher_derivative * r2


# ----------------------------------------------------------------------------------------

class BSpline(Surface):
    """
    Implemented according to Wikipedia.
    """
    def __init__(self, r, d, size, px=3, py=3, tx=None, ty=None, c=None, device=torch.device('cpu')): # input c is 1D
        Surface.__init__(self, r, d, device)
        self.px = px
        self.py = py
        self.size = np.asarray(size)

        # knots
        if tx is None:
            self.tx = None
        else:
            if len(tx) != size[0] + 2*(self.px + 1):
                raise Exception('len(tx) is not correct!')
            self.tx = torch.Tensor(np.asarray(tx)).to(self.device)
        if ty is None:
            self.ty = None
        else:
            if len(ty) != size[1] + 2*(self.py + 1):
                raise Exception('len(ty) is not correct!')
            self.ty = torch.Tensor(np.asarray(ty)).to(self.device)

        # c is the only differentiable parameter
        c_shape = size + np.array([self.px, self.py]) + 1
        if c is None:
            self.c = None
        else:
            c = np.asarray(c)
            if c.size != np.prod(c_shape):
                raise Exception('len(c) is not correct!')
            self.c = torch.Tensor(c.reshape(*c_shape)).to(self.device)
        
        if (self.tx is None) or (self.ty is None) or (self.c is None):
            self.tx = self._generate_knots(self.r, size[0], p=px, device=device)
            self.ty = self._generate_knots(self.r, size[1], p=py, device=device)
            self.c = torch.zeros(*c_shape, device=device)
        else:
            self.to(self.device)

    @staticmethod
    def _generate_knots(R, n, p=3, device=torch.device('cpu')):
        t = np.linspace(-R, R, n)
        step = t[1] - t[0]
        T = t[0] - 0.9 * step
        np.pad(t, p+1, 'constant', constant_values=step)
        t = np.concatenate((np.ones(p+1)*T, t, -np.ones(p+1)*T), axis=0)
        return torch.Tensor(t).to(device)

    def fit(self, x, y, z, eps=1e-3):
        x, y, z = (v.flatten() for v in [x, y, z])

        # knot positions within [-r, r]^2
        X = np.linspace(-self.r, self.r, self.size[0])
        Y = np.linspace(-self.r, self.r, self.size[1])
        bs = LSQBivariateSpline(x, y, z, X, Y, kx=self.px, ky=self.py, eps=eps)
        tx, ty = bs.get_knots()
        c = bs.get_coeffs().reshape(len(tx)-self.px-1, len(ty)-self.py-1)

        # convert to torch.Tensor
        self.tx, self.ty, self.c = (torch.Tensor(v).to(self.device) for v in [tx, ty, c])

    # === Common methods
    def g(self, x, y):
        return self._deBoor2(x, y)

    def dgd(self, x, y):
        return self._deBoor2(x, y, dx=1), self._deBoor2(x, y, dy=1)

    def h(self, z):
        return -z

    def dhd(self, z):
        return -torch.ones_like(z)

    def surface(self, x, y):
        return self._deBoor2(x, y)

    def surface_derivatives(self, x, y):
        return self._deBoor2(x, y, dx=1), self._deBoor2(x, y, dy=1), -torch.ones_like(x)

    def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, z, A, B, C):
        x = ox + t * dx
        y = oy + t * dy
        s, sx, sy = self._deBoor2(x, y, dx=-1, dy=-1)
        return s - z, sx*dx + sy*dy - dz

    def reverse(self):
        self.c = -self.c

    # === Private methods
    def _deBoor(self, x, t, c, p=3, is2Dfinal=False, dx=0):
        """
        Arguments
        ---------
        x: Position.
        t: Array of knot positions, needs to be padded as described above.
        c: Array of control points.
        p: Degree of B-spline.
        dx:
        - 0: surface only
        - 1: surface 1st derivative only
        - -1: surface and its 1st derivative
        """
        k = torch.sum((x[None,...] > t[...,None]).int(), axis=0) - (p+1)
        
        if is2Dfinal:
            inds = np.indices(k.shape)[0]
            def _c(jk): return c[jk, inds]
        else:
            def _c(jk): return c[jk, ...]

        need_newdim = (len(c.shape) > 1) & (not is2Dfinal)

        def f(a, b, alpha):
            if need_newdim:
                alpha = alpha[...,None]
            return (1.0 - alpha) * a + alpha * b
        
        # surface only
        if dx == 0:
            d = [_c(j+k) for j in range(0, p+1)]

            for r in range(-p, 0):
                for j in range(p, p+r, -1):
                    left = j+k
                    t_left  = t[left]
                    t_right = t[left-r]
                    alpha = (x - t_left) / (t_right - t_left)
                    d[j] = f(d[j-1], d[j], alpha)
            return d[p]

        # surface 1st derivative only
        if dx == 1:
            q = []
            for j in range(1, p+1):
                jk = j+k
                tmp = t[jk+p] - t[jk]
                if need_newdim:
                    tmp = tmp[..., None]
                q.append(p * (_c(jk) - _c(jk-1)) / tmp)

            for r in range(-p, -1):
                for j in range(p-1, p+r, -1):
                    left = j+k
                    t_right = t[left-r]
                    t_left_ = t[left+1]
                    alpha = (x - t_left_) / (t_right - t_left_)
                    q[j] = f(q[j-1], q[j], alpha)
            return q[p-1]
            
        # surface and its derivative (all)
        if dx < 0:
            d, q = [], []
            for j in range(0, p+1):
                jk = j+k
                c_jk = _c(jk)
                d.append(c_jk)
                if j > 0:
                    tmp = t[jk+p] - t[jk]
                    if need_newdim:
                        tmp = tmp[..., None]
                    q.append(p * (c_jk - _c(jk-1)) / tmp)

            for r in range(-p, 0):
                for j in range(p, p+r, -1):
                    left = j+k
                    t_left  = t[left]
                    t_right = t[left-r]
                    alpha = (x - t_left) / (t_right - t_left)
                    d[j] = f(d[j-1], d[j], alpha)

                    if (r < -1) & (j < p):
                        t_left_ = t[left+1]
                        alpha = (x - t_left_) / (t_right - t_left_)
                        q[j] = f(q[j-1], q[j], alpha)
            return d[p], q[p-1]

    def _deBoor2(self, x, y, dx=0, dy=0):
        """
        Arguments
        ---------
        x,  y : Position.
        dx, dy: 
        """
        if not torch.is_tensor(x):
            x = torch.Tensor(np.asarray(x)).to(self.device)
        if not torch.is_tensor(y):
            y = torch.Tensor(np.asarray(y)).to(self.device)
        dim = x.shape

        x = x.flatten()
        y = y.flatten()

        # handle boundary issue
        x = torch.clamp(x, min=-self.r, max=self.r)
        y = torch.clamp(y, min=-self.r, max=self.r)

        if (dx == 0) & (dy == 0):     # spline
            s_tmp = self._deBoor(x, self.tx, self.c, self.px)
            s = self._deBoor(y, self.ty, s_tmp.T, self.py, True)
            return s.reshape(dim)
        elif (dx == 1) & (dy == 0):  # x-derivative
            s_tmp = self._deBoor(y, self.ty, self.c.T, self.py)
            s_x = self._deBoor(x, self.tx, s_tmp.T, self.px, True, dx)
            return s_x.reshape(dim)
        elif (dy == 1) & (dx == 0):  # y-derivative
            s_tmp = self._deBoor(x, self.tx, self.c, self.px)
            s_y = self._deBoor(y, self.ty, s_tmp.T, self.py, True, dy)
            return s_y.reshape(dim)
        else:                       # return all
            s_tmpx = self._deBoor(x, self.tx, self.c, self.px)
            s_tmpy = self._deBoor(y, self.ty, self.c.T, self.py)
            s, s_x = self._deBoor(x, self.tx, s_tmpy.T, self.px, True, -abs(dx))
            s_y = self._deBoor(y, self.ty, s_tmpx.T, self.py, True, abs(dy))
            return s.reshape(dim), s_x.reshape(dim), s_y.reshape(dim)


class XYPolynomial(Surface):
    """
    General XY polynomial surface of equation of parameters:
    
    explicit:   b z^2 - z + \sum{i,j} a_ij x^i y^{j-i} = 0
    implicit:   (denote c = \sum{i,j} a_ij x^i y^{j-i})
                z = (1 - \sqrt{1 - 4 b c}) / (2b)
                
    explicit derivatives:
    (2 b z - 1) dz + \sum{i,j} a_ij x^{i-1} y^{j-i-1} ( i y dx + (j-i) x dy ) = 0

    dx = \sum{i,j} a_ij   i   x^{i-1} y^{j-i}
    dy = \sum{i,j} a_ij (j-i) x^{i}   y^{j-i-1}
    dz = 2 b z - 1
    """
    def __init__(self, r, d, J=0, ai=None, b=None, device=torch.device('cpu')):
        Surface.__init__(self, r, d, device)
        self.J  = J
        # differentiable parameters (default: all ai's and b are zeros)
        if ai is None:
            self.ai = torch.zeros(self.J2aisize(J)) if J > 0 else torch.array([0])
        else:
            if len(ai) != self.J2aisize(J):
                raise Exception("len(ai) != (J+1)*(J+2)/2 !")
            self.ai = torch.Tensor(ai).to(device)
        if b is None:
            b = 0.
        self.b = torch.Tensor(np.asarray(b)).to(device)
        print('ai.size = {}'.format(self.ai.shape[0]))
        self.to(self.device)
    
    @staticmethod
    def J2aisize(J):
        return int((J+1)*(J+2)/2)

    def center(self):
        x0 = -self.ai[2]/self.ai[5]
        y0 = -self.ai[1]/self.ai[3]
        return x0, y0

    def fit(self, x, y, z):
        x, y, z = (torch.Tensor(v.flatten()) for v in [x, y, z])
        A, AT = self._construct_A(x, y, z**2)
        coeffs = torch.solve(AT @ z[...,None], AT @ A)[0]
        self.b  = coeffs[0][0]
        self.ai = coeffs[1:].flatten()

    # === Common methods
    def g(self, x, y):
        c = torch.zeros_like(x)
        count = 0
        for j in range(self.J+1):
            for i in range(j+1):
                c = c + self.ai[count] * torch.pow(x, i) * torch.pow(y, j-i)
                count += 1
        return c

    def dgd(self, x, y):
        sx = torch.zeros_like(x)
        sy = torch.zeros_like(x)
        count = 0
        for j in range(self.J+1):
            for i in range(j+1):
                if j > 0:
                    sx = sx + self.ai[count] * i * torch.pow(x, max(i-1,0)) * torch.pow(y, j-i)
                    sy = sy + self.ai[count] * (j-i) * torch.pow(x, i) * torch.pow(y, max(j-i-1,0))
                count += 1
        return sx, sy

    def h(self, z):
        return self.b * z**2 - z

    def dhd(self, z):
        return 2 * self.b * z - torch.ones_like(z)

    def surface(self, x, y):
        x, y = (v if torch.is_tensor(x) else torch.Tensor(v) for v in [x, y])
        c = self.g(x, y)
        return self._solve_for_z(c)

    def reverse(self):
        self.b = -self.b
        self.ai = -self.ai

    def surface_derivatives(self, x, y):
        x, y = (v if torch.is_tensor(x) else torch.Tensor(v) for v in [x, y])
        sx = torch.zeros_like(x)
        sy = torch.zeros_like(x)
        c = torch.zeros_like(x)
        count = 0
        for j in range(self.J+1):
            for i in range(j+1):
                c = c + self.ai[count] * torch.pow(x, i) * torch.pow(y, j-i)
                if j > 0:
                    sx = sx + self.ai[count] * i * torch.pow(x, max(i-1,0)) * torch.pow(y, j-i)
                    sy = sy + self.ai[count] * (j-i) * torch.pow(x, i) * torch.pow(y, max(j-i-1,0))
                count += 1
        z = self._solve_for_z(c)
        return sx, sy, self.dhd(z)
        
    def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, z, A, B, C):
        x = ox + t * dx
        y = oy + t * dy
        sx = torch.zeros_like(x)
        sy = torch.zeros_like(x)
        c = torch.zeros_like(x)
        count = 0
        for j in range(self.J+1):
            for i in range(j+1):
                c = c + self.ai[count] * torch.pow(x, i) * torch.pow(y, j-i)
                if j > 0:
                    sx = sx + self.ai[count] * i * torch.pow(x, max(i-1,0)) * torch.pow(y, j-i)
                    sy = sy + self.ai[count] * (j-i) * torch.pow(x, i) * torch.pow(y, max(j-i-1,0))
                count += 1
        s = c + self.h(z)
        return s, sx*dx + sy*dy + self.dhd(z)*dz

    # === Private methods
    def _construct_A(self, x, y, A_init=None):
        A = torch.zeros_like(x) if A_init == None else A_init
        for j in range(self.J+1):
            for i in range(j+1):
                A = torch.vstack((A, torch.pow(x, i) * torch.pow(y, j-i)))
        AT = A[1:,:] if A_init == None else A
        return AT.T, AT

    def _solve_for_z(self, c):
        if self.b == 0:
            return c
        else:
            return (1. - torch.sqrt(1. - 4*self.b*c)) / (2*self.b)
        

# ----------------------------------------------------------------------------------------

def generate_test_lensgroup():
    origin_mount = np.array([0, 0, -70])
    origin_shift = np.array([0.1, 0.2, 0.3])
    theta_x = 180
    theta_y = 0

    lensname = 'Thorlabs/AL50100-A.txt'
    lensgroup = Lensgroup(lensname, origin_mount, origin_shift, theta_x, theta_y)
    return lensgroup

# ----------------------------------------------------------------------------------------


if __name__ == "__main__":
    init()

    lensgroup = generate_test_lensgroup()
    # print(lensgroup)

    ray = generate_test_rays()
    ray_out, valid, mask = lensgroup.trace(ray)

    ray_out.d = -ray_out.d
    ray_out.update()

    ray_new, valid_, mask_ = lensgroup.trace(ray_out)
    # assert np.sum(np.abs(ray.d.numpy() + ray_new.d.numpy())) < 1e-5
