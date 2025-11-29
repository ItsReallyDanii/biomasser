import numpy as np
from skimage.measure import label

def compute_flow_metrics(img):
    """Mini pseudo-physics simulator for fast feedback."""
    img = (img > 0.5).astype(float)  # binary mask
    porosity = img.mean()

    # Approximate permeability using connected component length
    labels = label(img)
    num_components = labels.max()
    mean_size = np.mean([np.sum(labels == i) for i in range(1, num_components + 1)]) if num_components > 0 else 0
    K = mean_size / (img.size + 1e-5)

    # Dummy flow rate proxy (Darcy-like)
    FlowRate = K * porosity * 0.5

    return {
        "Porosity": porosity,
        "FlowRate": FlowRate,
        "K": K
    }
