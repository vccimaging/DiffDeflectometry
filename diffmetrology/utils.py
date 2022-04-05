from .scene import *
from .solvers import *
import scipy
import scipy.optimize
import scipy.io

from matplotlib.image import imread
import matplotlib.ticker as plticker
import time


def var2string(variable):
    for name in globals():
        if eval(name) == variable:
            return name
    return ''

class DiffMetrology(PrettyPrinter):
    """
    Major class to handle all situations
    """
    def __init__(self,
        calibration_path, rotation_path,
        origin_shift, lut_path=None, thetas=None, angles=0.,
        scale=1, device=torch.device('cpu')):

        self.device = device
        self.MAX_VAL = 2**16 - 1.0 # 16-bit image
        
        # geometry setup
        mat_g = scipy.io.loadmat(calibration_path + 'cams.mat')
        cameras = self._init_camera(mat_g, scale)
        screen = self._init_screen(mat_g)
        self.scene = Scene(cameras, screen, device=device)
        
        # cache calibration checkerboard image for testing
        self._checkerboard = imread(calibration_path + 'checkerboard.png')
        self._checkerboard = np.mean(self._checkerboard, axis=-1) # for now we use grayscale
        self._checkerboard = np.flip(self._checkerboard, axis=1).copy()
        self._checkerboard_wh = np.array([mat_g['w'], mat_g['h']])

        # lensgroup metrology part setup
        mat_r = scipy.io.loadmat(rotation_path)
        p_rotation = torch.Tensor(np.stack((mat_r['p1'][0], mat_r['p2'][0]), axis=-1)).T
        origin_mount = self._compute_mount_geometry(p_rotation*scale, verbose=True)
        if thetas is None:
            self.scene.lensgroup = Lensgroup(origin_mount, origin_shift, 0.0, 0.0, 0.0, device)
        else:
            self.scene.lensgroup = Lensgroup(origin_mount, origin_shift, thetas[0], thetas[1], 0.0, device)

        if type(angles) is not list:
            angles = [angles]
        self.angles = angles
        
        # load sensor LUT
        if lut_path is not None:
            self.lut = scipy.io.loadmat(lut_path)['Js'][:,:self.scene.camera_count]
            self.bbd = scipy.io.loadmat(lut_path)['bs'].reshape((2,self.scene.camera_count))

        self.ROTATION_ANGLE = -25.0

    # === Utility methods ===
    def solve_for_intersections(self, fs_cap, fs_ref, Ts):
        """
        Obtain the intersection points from phase-shifting images.
        """
        FR = Fringe()
        a_ref, b_ref, psi_ref = FR.solve(fs_ref)
        a_cap, b_cap, psi_cap = FR.solve(fs_cap)

        VERBOSE = False
        # VERBOSE = True

        def find_center(valid_cap):
            x, y = np.argwhere(valid_cap==1).sum(0)/ valid_cap.sum()
            return np.array([x, y])

        # get valid map for two cameras
        valid_map = []
        A = np.mean(a_cap, axis=(0,1))
        B = np.mean(b_cap, axis=(0,1))
        I = np.abs(np.mean(fs_ref - fs_cap, axis=(0,1)))
        for j in range(self.scene.camera_count):
            # thres = 0.005
            # thres = 0.1
            thres = 0.07
            valid_ab = (A[j] > thres) & (B[j] > thres) & (I[j] < 2.0*thres)
            if VERBOSE:
                plt.imshow(valid_ab); plt.show()
            label, num_features = scipy.ndimage.label(valid_ab)
            if num_features < 2:
                label_target = 1
            else: # count labels and get the area as the target measurement area
                counts = np.array([np.count_nonzero(label == i) for i in range(num_features)])
                label_targets = np.where((200**2 < counts) & (counts < 500**2))[0]
                if len(label_targets) > 1: # find which label target is our lens
                    Dm = np.inf
                    for l in label_targets:
                        c = find_center(label == l)
                        D = np.abs(self.scene.cameras[0].filmsize/2 - c).sum()
                        if D < Dm:
                            Dm = D
                            label_target = l
                else:
                    label_target = label_targets[0]
            V = label == label_target
            valid_map.append(V)
        valid_map = np.array(valid_map)

        psi_unwrap = FR.unwrap(psi_cap - psi_ref, Ts, valid=valid_map[None,None,...])

        # Given the unwrapped phase, we try to remove the DC term so that |psi_unwrap|^2 is minimized
        # NOTE: This method is not yet robust
        # remove_DC = False
        remove_DC = True
        if remove_DC:
            k_DC = np.arange(-10,11,1)
            for t in range(len(Ts)):
                DCs = k_DC * Ts[t]
                psi_current = psi_unwrap[:,t,:,...]
                psi_with_dc = valid_map[:,None,:,:,None] * (psi_current[...,None] + DCs[None,None,None,None,...])
                DC_target = DCs[np.argmin(np.sum(psi_with_dc**2, axis=(2,3)), axis=-1)]
                print("t = {}, DC_target =\n{}".format(Ts[t], DC_target))
                psi_unwrap[:,t,:,...] += valid_map[None,:,...] * np.transpose(DC_target,(0,1))[...,None,None]
        
        if VERBOSE:
            for t in range(len(Ts)):
                for ii in range(2):
                    plt.figure()
                    plt.imshow(psi_unwrap[0,t,ii,...], cmap='coolwarm')
            plt.show()

        # Convert unit from [pixel number] to [mm] 
        psi_x = psi_unwrap[0, ...] * self.scene.screen.pixelsize.item()
        psi_y = psi_unwrap[1, ...] * self.scene.screen.pixelsize.item()

        # Get median of the values across different Ts
        psi_x = np.mean(psi_x, axis=0)
        psi_y = np.mean(psi_y, axis=0)
        
        # Compute intersection points when there are no elements
        ps_ref = torch.stack(self.trace(with_element=False, angles=0.0)[0])
        ps_ref = valid_map[...,None] * ps_ref[...,0:2].cpu().detach().numpy()

        # Compute final shift (here, the valid map is the valid map of x of the first T)
        # NOTE: here we flip the sequence of (x,y) to fit the format for lateral processings ...
        p = ps_ref - np.stack((psi_y, psi_x), axis=-1)

        # Find valid map centers
        xs = []
        ys = []
        for i in range(len(valid_map)):
            x, y = np.argwhere(valid_map[i]==1).sum(0)/ valid_map[i].sum()
            xs.append(x)
            ys.append(y)
        centers = np.stack((np.array(xs), np.array(ys)), axis=-1)
        centers = np.fliplr(centers).copy() # flip to fit our style

        return torch.Tensor(p).to(self.device), torch.Tensor(valid_map).bool().to(self.device), torch.Tensor(centers).to(self.device)


    def simulation(self, sinusoid_path, Ts, i=None, angles=None, to_numpy=False):
        """
        Render fringe images.

        img.shape = [ len(Ts), 8, self.camera_count, img_size ]
        """
        print('Simulation ...')

        # cache current screen
        screen_org = self.scene.screen
        pixelsize = screen_org.pixelsize.item()

        def single_impl(with_element):
            """
            imgs_all.shape = len(Ts) * [0-8] * camera_count * img_size.
            """
            imgs_all = []
            for T in Ts:
                print(f'Now at T = {T}')
                img_path = sinusoid_path + 'T=' + str(T) + '/'
                
                # with elements
                imgs = []
                for i in range(8):
                    # read sinusoid images
                    im = imread(img_path + str(i) + '.png')
                    im = np.mean(im, axis=-1) # for now we use grayscale
                    im = np.flip(im).copy() # NOTE: (i) our display is rotated by 90 deg;
                    # (ii) to be consistent with XY convention here. So the flip.

                    # set screen to be sinusoid patterns
                    sizenew = pixelsize * np.array(im.shape)
                    t = np.array([sizenew[0]/2-50.0, sizenew[1]/2-80.0, 0])
                    self.scene.screen = Screen(Transformation(np.eye(3), t), sizenew, pixelsize, im, self.device)

                    # render
                    tmp = self.scene.render(with_element=with_element, to_numpy=True)
                    imgs.append(np.array(tmp))
                imgs_all.append(np.array(imgs))

            return np.array(imgs_all)

        def single(angle):
            self.scene.lensgroup.update(_y=angle)
            imgs = single_impl(True)
            self.scene.lensgroup.update(_y=-angle)
            return imgs

        # we only capture reference once
        print('Simulating reference ...')
        refs = single_impl(False)

        # here, we measure testing part in each angle
        print('Simulating measurements ...')
        if angles is None:
            """
            imgs_rendered.shape = len(angles) * len(Ts) * [0-8] * camera_count * img_size.
            """
            ims = []
            for angle in self.angles:
                ims.append(single(angle))
            ims = np.array(ims)
        else:
            print(f'Now at angle = {angles}')
            ims = single(angles)

        # revert back to the original screen
        self.scene.screen = screen_org

        print('Done ...')
        return ims, refs

    def render(self, i=None, with_element=True, mask=None, angles=None, to_numpy=False):
        """
        Rendering.
        """
        def single(angle):
            self.scene.lensgroup.update(_y=angle)
            im = self.scene.render(i, with_element, mask, to_numpy)
            self.scene.lensgroup.update(_y=-angle)
            return im

        if angles is None:
            ims = []
            for angle in self.angles:
                ims += single(angle)
        else:
            ims = single(angles)
        return ims

    def trace(self, i=None, with_element=True, mask=None, angles=None, to_numpy=False):
        """
        Perform ray tracing.
        """
        def single(angle):
            self.scene.lensgroup.update(_y=angle)
            ps, valid, mask_g = self.scene.trace(i, with_element, mask, to_numpy)
            self.scene.lensgroup.update(_y=-angle)
            return ps, valid, mask_g
        
        if angles is None:
            ps = []
            valid = []
            mask_g = []
            for angle in self.angles:
                pss, valids, mask_gs = single(angle)
                ps += pss
                valid += valids
                mask_g += mask_gs
        else:
            ps, valid, mask_g = single(angles)
        return ps, valid, mask_g

    # =====================

    def to(self, device=torch.device('cpu')):
        super().to(device)
        self.device = device
        self.scene.to(device)

    # === Visualizations ===
    def imshow(self, imgs):
        N = self.scene.camera_count
        self._imshow(imgs[0:N], title='front')
        if len(imgs) > N:
            self._imshow(imgs[N:2*N], title='back')
        plt.show()
        
    def _imshow(self, imgs, title=''):
        ax = plt.subplots(1,len(imgs))[1]
        for i, img in enumerate(imgs):
            ax[i].imshow(img, cmap='gray')
            ax[i].set_title(title + ': camera ' + str(i+1))

    def spot_diagram(self, ps_ref, ps_cap, valid=True, angle=None, with_grid=True):
        """
        Plot spot diagram.
        """
        N = self.scene.camera_count
        for j, a in enumerate(self.angles):
            if angle == a:
                i = j
        try:
            i
        except NameError:
            i = 0
        figure = self._spot_diagram(ps_ref[N*i:N*(i+1)], ps_cap[N*i:N*(i+1)], valid[N*i:N*(i+1)], title=f'angle={angle}', with_grid=with_grid)
        figure.suptitle('Spot Diagram')
        return figure

    def _spot_diagram(self, ps_ref, ps_cap, valid=True, title='', with_grid=False):
        """
        Plot spot diagram.
        """
        figure, ax = plt.subplots(1, self.scene.camera_count)

        def sub_sampling(x):
            Ns = [8,8]
            return x[::Ns[0],::Ns[1],...]
        for i in range(len(ax)):
            mask = sub_sampling(valid[i])
            ref = sub_sampling(ps_ref[i])[mask].cpu().detach().numpy()
            cap = sub_sampling(ps_cap[i])[mask].cpu().detach().numpy()
            ax[i].plot(ref[...,0], ref[...,1], 'b.', label='Measurement')
            ax[i].plot(cap[...,0], cap[...,1], 'r.', label='Modeled (reprojection)')
            ax[i].legend()
            ax[i].set_xlabel('[mm]')
            ax[i].set_ylabel('[mm]')
            ax[i].set_aspect(1)
            ax[i].set_title(title + ': camera ' + str(i+1))

            # Add the grid
            if with_grid:
                loc = plticker.MultipleLocator(base=4*self.scene.screen.pixelsize.item())
                ax[i].xaxis.set_major_locator(loc)
                ax[i].yaxis.set_major_locator(loc)
                ax[i].grid(which='major', axis='both', linestyle='-')
                ax[i].tick_params(axis='both', which='minor', width=0)
        return figure

    def generate_grid(self, R):
        N = 513
        x = y = torch.linspace(-R, R, N, device=self.device)
        X, Y = torch.meshgrid(x, y)
        valid = X**2 + Y**2 <= R**2
        return X, Y, valid

    def show_surfaces(self, verbose=True):
        if verbose:
            ax = plt.subplots(1, len(self.scene.lensgroup.surfaces))[1]
        Zs = []
        valids = []
        for i, surface in enumerate(self.scene.lensgroup.surfaces):
            X, Y, valid = self.generate_grid(surface.r)
            Z = surface.surface(X, Y)
            Z_mean = Z[valid].mean().item()
            Z = torch.where(valid, Z - Z_mean, torch.zeros_like(Z)).cpu().detach().numpy()
            valids.append(valid)
            Zs.append(Z)
            if verbose:
                im = ax[i].imshow(Z, cmap='jet')
                ax[i].set_title('surface ' + str(i))
                plt.colorbar(im, ax=ax[i])
        if verbose:
            plt.show()
        return Zs, valids

    def print_surfaces(self):
        if self.scene.lensgroup == None:
            print('No surfaces found; Please initialize lensgroup!')
        else:
            for i, s in enumerate(self.scene.lensgroup.surfaces):
                print("surface[{}] = {}".format(i, s))

    # =====================

    # === Optimizations ===
    def change_parameters(self, diff_parameters_names, xs, sign=True):
        diff_parameters = []
        for i, name in enumerate(diff_parameters_names):
            if sign:
                exec('self.scene.{name} = self.scene.{name} + xs[{i}]'.format(name=name,i=i))
            else:
                exec('self.scene.{name} = self.scene.{name} - xs[{i}]'.format(name=name,i=i))
            exec('diff_parameters.append(self.scene.{})'.format(name))
        return diff_parameters

    def solve(self, diff_parameters_names, forward, loss, func_yref_y=None, option='LM', R=None):
        """
        Solve for unknown parameters.
        """
        # def loss(I):
        #     return (I - I0).mean()

        # def func_yref_y(I):
        #     return I0 - I

        time_start = time.time()
        
        diff_parameters = []
        for name in diff_parameters_names:
            try:
                exec('self.scene.{}.requires_grad = True'.format(name))
            except:
                exec('self.scene.{name} = self.scene.{name}.detach()'.format(name=name))
                exec('self.scene.{}.requires_grad = True'.format(name))
            exec('diff_parameters.append(self.scene.{})'.format(name))

        if option == 'Adam':
            O = Adam(
                diff_variables=diff_parameters,
                lr=1e-1,
                beta=0.99
            )
            ls = O.optimize(
                forward,
                loss,
                maxit=200
            )

        elif option == 'LM':
            if func_yref_y is None:
                raise Exception("func_yref_y is not given!")
            
            if R is None:
                Ropt = 'I'
            else:
                Ropt = R
            O = LM(
                diff_variables=diff_parameters,
                lamb=1e-1, # 1e-4
                option=Ropt
            )
            ls = O.optimize(
                forward,
                lambda xs, sign: self.change_parameters(diff_parameters_names, xs, sign),
                func_yref_y=func_yref_y,
                maxit=100
            )
        
        else:
            raise NotImplementedError()
        
        for name in diff_parameters_names:
            print('self.scene.{} = '.format(name), end='')
            exec('print(self.scene.{}.cpu().detach().numpy())'.format(name))
            
        torch.cuda.synchronize()
        time_end = time.time()

        print('Elapsed time = {:e} seconds'.format(time_end - time_start))

        return ls
    # =====================


    # === Optimizations ===
    def init_diff_parameters(self, dicts=None, pose_dict=None):
        self.diff_parameters = {}
        print('Initializing differentiable parameters:')
        if dicts is not None:
            for i, dictionary in enumerate(dicts):
                if dictionary is None:
                    continue # skip if there is none
                if type(dictionary) is dict:
                    for key, value in dictionary.items():
                        keystr = 'surfaces[{}].{}'.format(i, key)
                        full_keystr = 'self.scene.lensgroup.' + keystr
                        print('--- ' + full_keystr)
                        self.diff_parameters[keystr] = torch.Tensor(np.asarray(value)).to(self.device)
                        exec('{} = self.diff_parameters[keystr].clone()'.format(full_keystr))
                        exec('{}.requires_grad = True'.format(full_keystr))
                elif type(dictionary) is set:
                    for key in dictionary:
                        keystr = 'surfaces[{}].{}'.format(i, key)
                        full_keystr = 'self.scene.lensgroup.' + keystr
                        print('--- ' + full_keystr)
                        exec('self.diff_parameters[keystr] = {}.clone()'.format(full_keystr))
                        exec('{}.requires_grad = True'.format(full_keystr))
                else:
                    raise Exception("wrong type dicts!")
        if pose_dict is not None:
            if type(pose_dict) is dict:
                for keystr, value in pose_dict.items():
                    full_keystr = 'self.scene.lensgroup.' + keystr
                    print('--- ' + full_keystr)
                    self.diff_parameters[keystr] = torch.Tensor(np.asarray(value)).to(self.device)
                    exec('{} = self.diff_parameters[keystr].clone()'.format(full_keystr))
                    exec('{}.requires_grad = True'.format(full_keystr))
            elif type(pose_dict) is set:
                for keystr in pose_dict:
                    full_keystr = 'self.scene.lensgroup.' + keystr
                    print('--- ' + full_keystr)
                    exec('self.diff_parameters[keystr] = {}.clone()'.format(full_keystr))
                    exec('{}.requires_grad = True'.format(full_keystr))
            else:
                raise Exception("wrong type dicts!")
        print('... Done.')

    def print_diff_parameters(self):
        for key in self.diff_parameters.keys():
            full_keystr = 'self.scene.lensgroup.' + key
            exec("print('-- {{}} = {{}}'.format(key, {}.cpu().detach().numpy()))".format(full_keystr))

    @staticmethod
    def compRMS(Zs, Zs_gt, valid):
        def RMS(x, y, valid):
            x = x[valid]
            y = y[valid]
            x -= x.mean()
            y -= y.mean()
            tmp = (x - y)**2
            return np.sqrt(np.mean(tmp))
        
        rmss = []
        for i in range(len(Zs_gt)):
            rms = RMS(Zs[i], Zs_gt[i], valid[i].cpu().detach().numpy())
            print('RMS = {} [mm]'.format(rms))
            rmss.append(rms)
        return rmss

    def ploterror(self, Zs, Zs_gt, verbose=True):
        figure, ax = plt.subplots(1, len(self.scene.lensgroup.surfaces), figsize=(9,3.5))
        for i, s in enumerate(self.scene.lensgroup.surfaces):
            tmp = Zs[i] - Zs_gt[i]
            im = ax[i].imshow(tmp, cmap='jet')
            ax[i].set_title('surface ' + str(i))
            plt.colorbar(im, ax=ax[i])
        if verbose:
            plt.show()
        return figure
    
    # =====================

    def set_texture(self, textures):
        if len(textures.shape) > 2:
            texture = textures[0]
        else:
            texture = textures
        pixelsize = self.scene.screen.pixelsize.item()
        sizenew = pixelsize * np.array(texture.shape)
        t = np.zeros(3)
        self.scene.screen = Screen(Transformation(np.eye(3), t), sizenew, pixelsize, np.flip(texture).copy(), self.device)

    # === Tests ===
    def test_setup(self, verbose=True):
        """
        Test if the setup is correct: check render and measurement images for consistency
        """
        # cache current screen
        screen_org = self.scene.screen
        lensgroup = self.scene.lensgroup
        self.scene.lensgroup = None

        # set screen to be checkerboard
        pixelsize = screen_org.pixelsize.item()
        sizenew = pixelsize * np.array(self._checkerboard.shape)
        t = np.array([sizenew[0]/2, sizenew[1]/2, 0])
        t[0] -= sizenew[0]/self._checkerboard_wh[0]
        t[1] -= sizenew[1]/self._checkerboard_wh[1]
        self.scene.screen = Screen(Transformation(np.eye(3), t), sizenew, pixelsize, self._checkerboard, self.device)

        # render
        imgs_rendered = self.scene.render()
        if verbose:
            self.scene.plot_setup()
            plt.show()
        
        # revert back to the original screen
        self.scene.screen = screen_org
        self.scene.lensgroup = lensgroup
        return imgs_rendered
    # =====================

    # === Internal methods ===
    # parse cameras and screen parameters
    def _init_camera(self, mat, scale=1):
        filmsize = (mat['filmsize'] * scale).astype(int)
        f = mat['f'] * scale
        c = mat['c'] * scale
        k = mat['k'] * scale
        p = mat['p'] * scale
        R = mat['R']
        t = mat['t']
        def matlab2ours(R, t):
            """Convert MATLAB [R | t] convention to ours:
            In MATLAB (https://www.mathworks.com/help/vision/ug/camera-calibration.html):
            w_scale [x_obj y_obj 1] = [x_world y_world z_world]  R_matlab   +   t_matlab
                    [     1x3     ]   [          1x3          ]    [3x3]          [1x3]
            
            Ours:
            [x_obj y_obj 1]' = R_object  [x_world y_world z_world]'  +  t_object
            [     3x1     ]    [  3x3 ]  [           3x1         ]       [3x1]
            
            We would like to get the world-coordinate [R | t]. This requires:
            R_world =  R_object.T
            t_world = -R_object.T @ t_object

            where we conclude R_world = R_matlab.
            """
            return R, -R @ t
        return [Camera(
                    Transformation(*matlab2ours(R[...,i], t[...,i])),
                    filmsize[i], f[i], c[i], k[i], p[i], self.device
                ) for i in range(len(f))]

    def _init_screen(self, mat):
        pixelsize = 1e-3 * mat['display_pixel_size'][0][0] # in [mm]
        size = pixelsize * np.array([1600, 2560]) # in [mm] (MacBook Pro 13.3")
        im = imread('./imgs/checkerboard.png')
        im = np.mean(im, axis=-1) # for now we use grayscale

        return Screen(Transformation(np.eye(3), np.zeros(3)), size, pixelsize, im, self.device)

    def _compute_mount_geometry(self, p_rotation, verbose=True):
        """
        We would like to estimate the intersection between two lines:
        
        L1 = o1 + t1 d1
        L2 = o2 + t2 d2
        
        where we need to solve for t1 and t2, by an over-determined least-squares
        (know R^3, solve for R^2):
        
         min   || L1 - L2 ||^2 = || (o1-o2) + [d1,-d2] [t1;t2] ||^2
        t1,t2                        [3x1]      [3x2]   [2x1]
        
        =>
        
         min   ||  o   +   d     t  ||^2
          t      [3x1]   [3x2] [2x1]
        
        whose solution is t = (d.T d)^{-1} d.T (-o).
        """
        N = self.scene.camera_count
        rays = [self.scene.cameras[i].sample_ray(
            p_rotation[i][None,None,...].to(self.device), is_sampler=False) for i in range(N)]

        t, r = np.linalg.lstsq(
            torch.stack((rays[0].d, -rays[1].d), axis=-1).cpu().detach().numpy(),
            -(rays[0].o - rays[1].o).cpu().detach().numpy(), rcond=None
        )[0:2]

        t_pt = torch.Tensor(t).to(self.device)
        os = [rays[i](t_pt[i]) for i in range(N)]
        if verbose:
            for i, o in enumerate(os):
                print('intersection point {}: {}'.format(i, o))
            print('|intersection points distance error| = {} mm'.format(np.sqrt(r[0])))
        return torch.mean(torch.stack(os), axis=0).cpu().detach().numpy()
    # =====================