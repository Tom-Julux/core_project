from itertools import combinations
from typing import Tuple

import numpy as np
from napari.utils.transforms import Affine


def is_orthogonal(affine: Affine, ndim: int, order: Tuple[int], ndisplay: int) -> bool:
    """
    affine = layer.affine
    ndim = layer.ndim
    order = viewer.dims.order
    ndisplay = viewer.dims.ndisplay
    """
    displayed = list(order[-ndisplay:])
    not_displayed = list(order[:-ndisplay])

    non_displayed_subspace = np.zeros(ndim)
    for d in not_displayed:
        non_displayed_subspace[d] = 1
    # Map subspace through inverse transform, ignoring translation
    world_to_data = Affine(
        ndim=ndim,
        linear_matrix=affine.linear_matrix,
        translate=None,
    )
    mapped_nd_subspace = world_to_data(non_displayed_subspace)
    # Look at displayed subspace
    displayed_mapped_subspace = (mapped_nd_subspace[d] for d in displayed)
    # Check that displayed subspace is null
    return all(abs(v) < 1e-8 for v in displayed_mapped_subspace)


def compute_affine(ndims, scale=None, translate=None, rotate=None, shear=None):
    """
    scale == spacing
    translate == origin
    rotate == direction

    """
    # Replace with default transformation if not given
    scale = np.ones(ndims) if scale is None else scale
    translate = np.ones(ndims) if translate is None else translate
    rotate = np.eye(ndims) if rotate is None else rotate
    shear = np.zeros(ndims) if shear is None else shear

    # Scaling
    scale_matrix = np.diag(scale)

    # Shearing
    shear_matrix = np.eye(ndims)
    upper_triangle_indices = list(combinations(range(ndims), 2))  # (i, j) pairs
    for shear_value, (i, j) in zip(shear, upper_triangle_indices):
        shear_matrix[i, j] = shear_value  # Shear in one direction
        shear_matrix[j, i] = shear_value  # Ensure symmetry for 2D and above

    # Build affine
    affine = np.eye(ndims + 1)
    affine[:ndims, :ndims] = rotate @ scale_matrix @ shear_matrix
    affine[:ndims, ndims] = translate

    return affine
