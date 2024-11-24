"""Gather data needed to fit a GLM.

"""

import numpy as np
from glob import glob

from osl_dynamics.analysis import power, connectivity

base_dir = "/well/woolrich/users/wlo995/Gohil2024_AgeEffectsRSNs"

# Source data file and subjects IDs
files = sorted(glob(f"{base_dir}/1_preproc_and_source_recon/data/src/*/sflip_parc-raw.fif"))
ids = np.array([file.split("/")[-2].split("-")[1] for file in files])

# Load target data
f = np.load(f"{base_dir}/4_time_averaged_networks/data/f.npy")
psd = np.load(f"{base_dir}/4_time_averaged_networks/data/psd.npy")
coh = np.load(f"{base_dir}/4_time_averaged_networks/data/coh.npy")
aec = np.load(f"{base_dir}/4_time_averaged_networks/data/aec.npy")

freq_bands = [[1, 4], [4, 8], [8, 13], [13, 24], [30, 45]]
m, n = np.triu_indices(coh.shape[-2], k=1)

def get_targets(id):
    i = np.squeeze(np.argwhere(ids == id))
    p = psd[i]
    c = coh[i]
    a = aec[i]
    p = np.array([power.variance_from_spectra(f, p, frequency_range=b) for b in freq_bands]).T
    c = np.array([connectivity.mean_coherence_from_spectra(f, c, frequency_range=b) for b in freq_bands])
    mc = connectivity.mean_connections(c).T
    c = c.T[m, n]
    ma = connectivity.mean_connections(a.T).T
    a = a[m, n]
    return p, c, mc, a, ma

# Load confound data
confound_ids = np.load(f"{base_dir}/3_design_matrix/data/id.npy")
age = np.load(f"{base_dir}/3_design_matrix/data/age.npy")
cog = np.load(f"{base_dir}/3_design_matrix/data/cog.npy")
sex = np.load(f"{base_dir}/3_design_matrix/data/sex.npy")
brain_vol = np.load(f"{base_dir}/3_design_matrix/data/brain_vol.npy")
gm_vol = np.load(f"{base_dir}/3_design_matrix/data/gm_vol.npy")
wm_vol = np.load(f"{base_dir}/3_design_matrix/data/wm_vol.npy")
headsize = np.load(f"{base_dir}/3_design_matrix/data/headsize.npy")
x = np.load(f"{base_dir}/3_design_matrix/data/x.npy")
y = np.load(f"{base_dir}/3_design_matrix/data/y.npy")
z = np.load(f"{base_dir}/3_design_matrix/data/z.npy")

# Lists to hold target data
pow_ = []
coh_ = []
mean_coh_ = []
aec_ = []
mean_aec_ = []

# Lists to hold regressor data
age_ = []
cog_ = []
sex_ = []
brain_vol_ = []
gm_vol_ = []
wm_vol_ = []
headsize_ = []
x_ = []
y_ = []
z_ = []

for id in ids:
    if id not in confound_ids:
        print(f"sub-{id} not found")
        continue
    print(f"sub-{id}")
    i = confound_ids == id

    # Get data
    p, c, mc, a, ma = get_targets(id)

    # Add to target data lists
    pow_.append(p)
    coh_.append(c)
    mean_coh_.append(mc)
    aec_.append(a)
    mean_aec_.append(ma)

    # Add to regressor lists
    age_.append(age[i][0])
    cog_.append(cog[i][0])
    sex_.append(sex[i][0])
    brain_vol_.append(brain_vol[i][0])
    gm_vol_.append(gm_vol[i][0])
    wm_vol_.append(wm_vol[i][0])
    headsize_.append(headsize[i][0])
    x_.append(x[i][0])
    y_.append(y[i][0])
    z_.append(z[i][0])

# Save data
np.save("data/glm_pow.npy", pow_)
np.save("data/glm_coh.npy", coh_)
np.save("data/glm_mean_coh.npy", mean_coh_)
np.save("data/glm_aec.npy", aec_)
np.save("data/glm_mean_aec.npy", mean_aec_)
np.save("data/glm_age.npy", age_)
np.save("data/glm_cog.npy", cog_)
np.save("data/glm_sex.npy", sex_)
np.save("data/glm_brain_vol.npy", brain_vol_)
np.save("data/glm_gm_vol.npy", gm_vol_)
np.save("data/glm_wm_vol.npy", wm_vol_)
np.save("data/glm_headsize.npy", headsize_)
np.save("data/glm_x.npy", x_)
np.save("data/glm_y.npy", y_)
np.save("data/glm_z.npy", z_)
