import numpy as np

def computeLRM_fromUnscaled(data, scalerOtpt, max_diss_rate_unscaled=None, clip=False):
    alpha = data[:,3]
    beta = data[:,4]
    gamma = data[:,5]
    cvar = data[:,1]   
    lrm = 0.32 * np.sqrt(alpha**2 + beta**2 + gamma**2) * np.clip(cvar, a_min=0, a_max=None)   
    lrm_scaled = scalerOtpt.transform(np.reshape(lrm, (-1,1)))
    if clip:
        lrm_scaled = clipLRM_fromScaled(lrm_scaled, max_diss_rate_unscaled, scalerOtpt)
    return lrm_scaled

def computeLRM_fromScaled(data, scalerInpt, scalerOtpt, max_diss_rate_unscaled=None, clip=False):
    data_unscaled = scalerInpt.inverse_transform(data)
    lrm_scaled = computeLRM_fromUnscaled(data_unscaled, scalerOtpt)
    if clip:
        lrm_scaled = clipLRM_fromScaled(lrm_scaled, max_diss_rate_unscaled, scalerOtpt)
    return lrm_scaled

def clipLRM_fromUnscaled(lrm, max_diss_rate_unscaled, scalerOtpt):
    lrm_clipped = np.clip(lrm, a_min=0, a_max=1.5*max_diss_rate_unscaled)
    lrm_clipped_scaled = scalerOtpt.transform(np.reshape(lrm_clipped, (-1,1)))
    return lrm_clipped_scaled
   
def clipLRM_fromScaled(lrm, max_diss_rate_unscaled, scalerOtpt):
    lrm_unscaled = scalerOtpt.inverse_transform(lrm)
    lrm_clipped_scaled = clipLRM_fromUnscaled(lrm_unscaled, max_diss_rate_unscaled, scalerOtpt)
    return lrm_clipped_scaled
