import numpy as np
import torch
import matplotlib.pyplot as plt
import diffmetrology as dm
from matplotlib.image import imread

# load setup information
data_path = './20210403'
device = dm.init()
# device = torch.device('cpu')

print("Initialize a DiffMetrology object.")
origin_shift = np.array([0.0, 0.0, 0.0])
DM = dm.DiffMetrology(
    calibration_path = data_path + '/calibration/',
    rotation_path = data_path + '/rotation_calibration/rotation.mat',
    lut_path = data_path + '/gamma_calibration/gammas.mat',
    origin_shift = origin_shift,
    scale=1.0,
    device=device
)

print("Crop the region of interst in the original images.")
filmsize = np.array([768, 768])
# filmsize = np.array([2048, 2048])
crop_offset = ((2048 - filmsize)/2).astype(int)
for cam in DM.scene.cameras:
    cam.filmsize = filmsize
    cam.crop_offset = torch.Tensor(crop_offset).to(device)
def crop(x):
    return x[..., crop_offset[0]:crop_offset[0]+filmsize[0], crop_offset[1]:crop_offset[1]+filmsize[1]]

DM.test_setup()

# ==== Read measurements
lens_name = 'LE1234-A'

DM.scene.lensgroup.load_file('Thorlabs/' + lens_name + '.txt')

def show_parameters():
    for i in range(len(DM.scene.lensgroup.surfaces)):
        print(f"Lens radius of curvature at surface[{i}]: {1.0/DM.scene.lensgroup.surfaces[i].c.item()}")
    print(DM.scene.lensgroup.surfaces[1].d)


print("Ground Truth Lens Parameters:")
show_parameters()


angle = 0.0
Ts = np.array([70, 100, 110]) # period of the sinusoids
t = 0

# load data
option = 'experiment'
if option == 'experiment':
    data = np.load(data_path + '/measurement/' + lens_name + '/data_new.npz')
    imgs = data['imgs']
    refs = data['refs']
    imgs = crop(imgs)
    refs = crop(refs)
    del data


# solve for ps and valid map
ps_cap, valid_cap, C = DM.solve_for_intersections(imgs, refs, Ts[t:])

# set display pattern
# xs = [0, 4]
xs = [0]
sinusoid_path = './camera_acquisitions/images/sinusoids/T=' + str(Ts[t])
ims = [ np.mean(imread(sinusoid_path + '/' + str(x) + '.png'), axis=-1) for x in xs ] # use grayscale
ims = np.array([ im/im.max() for im in ims ])
ims = np.sum(ims, axis=0)
DM.set_texture(ims)
del ims
if option == 'experiment':
    # Obtained from running `metrology_calibrate.py`
    # DM.scene.screen.texture_shift = torch.Tensor([1.7445182, 1.1107264]).to(device) # LE1234-A
    DM.scene.screen.texture_shift = torch.Tensor([0.       , 1.1106231]).to(device) # LE1234-A



print("Shift `origin` by an estimated value")
origin = DM._compute_mount_geometry(C, verbose=True)
DM.scene.lensgroup.origin = torch.Tensor(origin).to(device)
DM.scene.lensgroup.update()
print(origin)


print("Load real images")
FR = dm.Fringe()
a_cap, b_cap, psi_cap = FR.solve(imgs)
imgs_sub = np.array([imgs[0,x,...] for x in xs])
imgs_sub = imgs_sub - a_cap[:,0,...]
imgs_sub = np.sum(imgs_sub, axis=0)
imgs_sub = valid_cap * torch.Tensor(imgs_sub).to(device)
I0 = valid_cap * len(xs) * (imgs_sub - imgs_sub.min().item()) / (imgs_sub.max().item() - imgs_sub.min().item())


# Utility functions
def forward():
    ps = torch.stack(DM.trace(with_element=True, mask=valid_cap, angles=angle)[0])[..., 0:2]
    return ps

def render():
    I = valid_cap*torch.stack(DM.render(with_element=True, angles=angle))
    I[torch.isnan(I)] = 0.0
    return I

def visualize(ps_current, save_string):
    print("Showing spot diagrams at display.")
    DM.spot_diagram(ps_cap, ps_current, valid=valid_cap, angle=angle, with_grid=False)
    plt.show()
    
    print("Showing images (measurement & modeled & |measurement - modeled|).")

    # Render images from parameters
    I = render()
    
    fig, axes = plt.subplots(2, 3)
    for i in range(2):
        im = axes[i,0].imshow(I0[i].cpu(), vmin=0, vmax=1, cmap='gray')
        axes[i,0].set_title(f"Camera {i+1}\nMeasurement")
        axes[i,0].set_xlabel('[pixel]')
        axes[i,0].set_ylabel('[pixel]')
        plt.colorbar(im, ax=axes[i,0])

        im = axes[i,1].imshow(I[i].cpu().detach(), vmin=0, vmax=1, cmap='gray')
        plt.colorbar(im, ax=axes[i,1])
        axes[i,1].set_title(f"Camera {i+1}\nModeled")
        axes[i,1].set_xlabel('[pixel]')
        axes[i,1].set_ylabel('[pixel]')
        
        im = axes[i,2].imshow(I0[i].cpu() - I[i].cpu().detach(), vmin=-1, vmax=1, cmap='coolwarm')
        plt.colorbar(im, ax=axes[i,2])
        axes[i,2].set_title(f"Camera {i+1}\nError")
        axes[i,2].set_xlabel('[pixel]')
        axes[i,2].set_ylabel('[pixel]')
        
        fig.suptitle(save_string)
        fig.savefig(save_string + str(i) + ".jpg", bbox_inches='tight')
    plt.show()


print("Initialize lens parameters.")
DM.scene.lensgroup.surfaces[0].c = torch.Tensor([0.00]).to(device) # 1st surface curvature
DM.scene.lensgroup.surfaces[1].c = torch.Tensor([0.00]).to(device) # 2nd surface curvature
DM.scene.lensgroup.surfaces[1].d = torch.Tensor([3.00]).to(device) # lens thickness
DM.scene.lensgroup.theta_x = torch.Tensor([0.00]).to(device) # lens X-tilt angle
DM.scene.lensgroup.theta_y = torch.Tensor([0.00]).to(device) # lens Y-tilt angle
DM.scene.lensgroup.update()

print("Visualize initial status.")
ps_current = forward()
visualize(ps_current, save_string="initial")


print("Set optimization parameters.")
diff_names = [
    'lensgroup.surfaces[0].c',
    'lensgroup.surfaces[1].c',
    'lensgroup.surfaces[1].d',
    'lensgroup.origin',
    'lensgroup.theta_x',
    'lensgroup.theta_y'
]
def loss(ps):
    return torch.sum((ps[valid_cap,...] - ps_cap[valid_cap,...])**2, axis=-1).mean()

def func_yref_y(ps):
    b = valid_cap[...,None] * (ps_cap - ps)
    b[torch.isnan(b)] = 0.0 # handle NaN ... otherwise LM won't work!
    return b

# Optimize
ls = DM.solve(diff_names, forward, loss, func_yref_y, option='LM', R='I')
print("Done. Show results (Spot RMS loss):")
show_parameters()

plt.figure()
plt.semilogy(ls, '-o', color='k')
plt.xlabel('LM iteration')
plt.ylabel('Loss')
plt.title("Opitmization Loss")

print("Visualize optimized status.")
ps_current = forward()
visualize(ps_current, save_string="optimized")

# Print mean displacement error
T = ps_current - ps_cap
E = torch.sqrt(torch.sum(T[valid_cap, ...]**2, axis=-1)).mean()
print("error = {} [um]".format(E*1e3))

