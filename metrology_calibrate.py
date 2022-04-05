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
filmsize = np.array([1024, 1024])
crop_offset = ((2048 - filmsize)/2).astype(int)
for cam in DM.scene.cameras:
    cam.filmsize = filmsize
    cam.crop_offset = torch.Tensor(crop_offset).to(device)
def crop(x):
    return x[..., crop_offset[0]:crop_offset[0]+filmsize[0], crop_offset[1]:crop_offset[1]+filmsize[1]]


# ==== Read measurements
lens_name = 'LE1234-A'

# load data
data = np.load(data_path + '/measurement/' + lens_name + '/data_new.npz')
refs = data['refs']
refs = crop(refs)
del data

Ts = np.array([70, 100, 110])  # period of the sinusoids
t = 0

# change display pattern
xs = [0]
sinusoid_path = './camera_acquisitions/images/sinusoids/T=' + str(Ts[t])
ims = [ np.mean(imread(sinusoid_path + '/' + str(x) + '.png'), axis=-1) for x in xs ] # for now we use grayscale
ims = np.array([ im/im.max() for im in ims ])
ims = np.sum(ims, axis=0)
DM.set_texture(ims)
ims = torch.Tensor(ims).to(device)

# reference image
I0 = torch.Tensor(np.array([refs[t,x,...] for x in xs])).to(device)
I0 = torch.sum(I0, axis=0)

# define functions
def forward():
    I = torch.stack(DM.render(with_element=False, angles=0.0))
    return I #/ I.max() * I0.max()
def loss(I):
    return (I - I0).mean()
def func_yref_y(I):
    return I0 - I

def show_img(I, string):
    fig = plt.figure()
    plt.imshow(I[0].cpu().detach(), vmin=0, vmax=1, cmap='gray')
    plt.colorbar()
    plt.title(string)
    plt.axis('off')
    fig.savefig("img_" + string + ".jpg", bbox_inches='tight')

def show_error(I, string):
    fig = plt.figure()
    plt.imshow(I[0].cpu().detach() - I0[0].cpu(), vmin=-1, vmax=1, cmap='coolwarm')
    plt.colorbar()
    plt.title(string)
    plt.axis('off')
    fig.savefig("photo_" + string + ".jpg", bbox_inches='tight')


# initialize parameters
DM.scene.screen.texture_shift = torch.Tensor([0.0, 0.0]).to(device)

# parameters
diff_names = ['screen.texture_shift']

# initial
I = forward()
show_img(I0, 'Measurement')
show_img(I, 'Modeled')
show_error(I, 'Initial')

# optimize
ls = DM.solve(diff_names, forward, loss, func_yref_y, option='LM')

# plot loss
plt.figure()
plt.semilogy(ls, '-o', color='k')
plt.xlabel('LM iteration')
plt.ylabel('Loss')
plt.title("Opitmization Loss")

I = forward()
show_img(I0, 'Measurement')
show_img(I, 'Modeled')
show_error(I, 'Optimized')

plt.show()
