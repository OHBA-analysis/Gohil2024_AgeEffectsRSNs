"""Fit a GLM and do stats.

"""

import numpy as np
import glmtools as glm
from scipy import stats

do_pow = True
do_mean_coh = True

def do_stats(design, data, model, contrast_idx, metric="copes"):
    perm = glm.permutations.MaxStatPermutation(
        design=design,
        data=data,
        contrast_idx=contrast_idx,
        nperms=1000,
        metric=metric,
        tail=0,  # two-tailed t-test
        pooled_dims=(1,2),  # pool over channels and frequencies
        nprocesses=16,
    )
    if metric == "tstats":
        tstats = abs(model.tstats[0])
        percentiles = stats.percentileofscore(perm.nulls, tstats)
    elif metric == "copes":
        copes = abs(model.copes[0])
        percentiles = stats.percentileofscore(perm.nulls, copes)
    return 1 - percentiles / 100

def fit_glm_and_do_stats(target):
    data = glm.data.TrialGLMData(
        data=target,
        category_list=np.load("data/category_list.npy"),
        age=np.load("data/age.npy"),
        sex=np.load("data/sex.npy"),
        brain_vol=np.load("data/brain_vol.npy"),
        gm_vol=np.load("data/gm_vol.npy"),
        hip_vol=np.load("data/hip_vol.npy"),
        headsize=np.load("data/headsize.npy"),
        x=np.load("data/x.npy"),
        y=np.load("data/y.npy"),
        z=np.load("data/z.npy"),
        dim_labels=["Subjects", "Channels", "Frequencies"],
    )

    DC = glm.design.DesignConfig()
    DC.add_regressor(name="Top", rtype="Categorical", codes=1)
    DC.add_regressor(name="Bottom", rtype="Categorical", codes=2)
    DC.add_regressor(name="Age", rtype="Parametric", datainfo="age", preproc="z")
    DC.add_regressor(name="Sex", rtype="Parametric", datainfo="sex", preproc="z")
    DC.add_regressor(name="Brain Vol.", rtype="Parametric", datainfo="brain_vol", preproc="z")
    DC.add_regressor(name="GM Vol.", rtype="Parametric", datainfo="gm_vol", preproc="z")
    DC.add_regressor(name="Hippo. Vol.", rtype="Parametric", datainfo="hip_vol", preproc="z")
    DC.add_regressor(name="Head Size", rtype="Parametric", datainfo="headsize", preproc="z")
    DC.add_regressor(name="x", rtype="Parametric", datainfo="x", preproc="z")
    DC.add_regressor(name="y", rtype="Parametric", datainfo="y", preproc="z")
    DC.add_regressor(name="z", rtype="Parametric", datainfo="z", preproc="z")

    DC.add_contrast(name="Top-Bottom", values=[1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    design = DC.design_from_datainfo(data.info)
    design.plot_summary(savepath="plots/glm_design.png", show=False)
    design.plot_leverage(savepath="plots/glm_leverage.png", show=False)
    design.plot_efficiency(savepath="plots/glm_efficiency.png", show=False)

    model = glm.fit.OLSModel(design, data)

    copes = model.copes[0]
    pvalues = do_stats(design, data, model, contrast_idx=0)
    return copes, pvalues

if do_pow:
    target = np.load("data/pow.npy")
    copes, pvalues = fit_glm_and_do_stats(target)
    np.save("data/glm_pow.npy", copes)
    np.save("data/glm_pow_pvalues.npy", pvalues)

if do_mean_coh:
    target = np.load("data/mean_coh.npy")
    copes, pvalues = fit_glm_and_do_stats(target)
    np.save("data/glm_mean_coh.npy", copes)
    np.save("data/glm_mean_coh_pvalues.npy", pvalues)