"""
This file contains a custom implementation of the metrics used in the evaluation of the TrackRad2025 Challenge.
The metrics are implemented as subclasses of a replica of the IterationMetric class from MONAI.
This allows us to not rely on the MONAI library for the evaluation of the algorithms,
resulting in a much more lightweight and self-contained solution.

The metrics implemented are:
- DiceMetric: computes the Dice coefficient between the predicted and ground truth segmentations.
- SurfaceDistance95Metric: computes the 95th percentile of the Hausdorff distance between the predicted and ground truth segmentations.
- SurfaceDistanceAvgMetric: computes the average surface distance between the predicted and ground truth segmentations.

The monai_metrics_test.ipynb notebook contains tests for the implemented metrics,
comparing the results with the MONAI library to ensure correctness/equality.
"""
import numpy as np
import pandas as pd
from typing import Any, Sequence
import numpy as np
import scipy
import pandas as pd
import traceback
from typing import Any, Sequence
from scipy.ndimage import distance_transform_edt as scipy_distance_transform_edt, binary_erosion

TensorOrList = Sequence[np.ndarray] | np.ndarray

class IterationMetric():
    """
    Base class for metrics computation at the iteration level, that is, on a min-batch of samples
    usually using the model outcome of one iteration.

    `__call__` is designed to handle `y_pred` and `y` (optional) in np.ndarrays or a list/tuple of np.ndarrays.

    Subclasses typically implement the `_compute_tensor` function for the actual tensor computation logic.
    """

    def __call__(
        self, y_pred: TensorOrList, y: TensorOrList | None = None, **kwargs: Any
    ) -> np.ndarray | Sequence[np.ndarray | Sequence[np.ndarray]]:
        """
        Execute basic computation for model prediction `y_pred` and ground truth `y` (optional).
        It supports inputs of a list of "channel-first" Tensor and a "batch-first" Tensor.

        Args:
            y_pred: the raw model prediction data at one iteration, must be a list of `channel-first` Tensor
                or a `batch-first` Tensor.
            y: the ground truth to compute, must be a list of `channel-first` Tensor
                or a `batch-first` Tensor.
            kwargs: additional parameters for specific metric computation logic (e.g. ``spacing`` for SurfaceDistanceMetric, etc.).

        Returns:
            The computed metric values at the iteration level.
            The output shape could be a `batch-first` tensor or a list of `batch-first` tensors.
            When it's a list of tensors, each item in the list can represent a specific type of metric.

        """
        # handling a list of channel-first data
        if isinstance(y_pred, (list, tuple)) or isinstance(y, (list, tuple)):
            return self._compute_list(y_pred, y, **kwargs)
        # handling a single batch-first data
        if isinstance(y_pred, np.ndarray):
            return self._compute_tensor(y_pred, y, **kwargs)
        raise ValueError("y_pred or y must be a list/tuple of `channel-first` Tensors or a `batch-first` Tensor.")

    def _compute_list(
        self, y_pred: TensorOrList, y: TensorOrList | None = None, **kwargs: Any
    ) -> np.ndarray | list[np.ndarray | Sequence[np.ndarray]]:
        """
        Execute the metric computation for `y_pred` and `y` in a list of "channel-first" tensors.

        The return value is a "batch-first" tensor, or a list of "batch-first" tensors.
        When it's a list of tensors, each item in the list can represent a specific type of metric values.
        """
        if y is not None:
            ret = [
                self._compute_tensor(p.detach().unsqueeze(0), y_.detach().unsqueeze(0), **kwargs)
                for p, y_ in zip(y_pred, y)
            ]
        else:
            ret = [self._compute_tensor(p_.detach().unsqueeze(0), None, **kwargs) for p_ in y_pred]

        # concat the list of results (e.g. a batch of evaluation scores)
        if isinstance(ret[0], np.ndarray):
            return np.concatenate(ret, dim=0)  # type: ignore[arg-type]
        # the result is a list of sequence of tensors (e.g. a batch of multi-class results)
        if isinstance(ret[0], (list, tuple)) and all(isinstance(i, np.ndarray) for i in ret[0]):
            return [np.concatenate(batch_i, axis=0) for batch_i in zip(*ret)]
        return ret

    def _compute_tensor(self, y_pred: np.ndarray, y: np.ndarray | None = None, **kwargs: Any) -> TensorOrList:
        """
        Computation logic for `y_pred` and `y` of an iteration, the data should be "batch-first" Tensors.
        A subclass should implement its own computation logic.
        The return value is usually a "batch_first" tensor, or a list of "batch_first" tensors.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class DiceMetric(IterationMetric):
    def __init__(self) -> None:
        super().__init__()

    def _compute_tensor(self, y_pred: np.ndarray, y_true: np.ndarray | None = None, **kwargs):
        B, C, H, W = y_pred.shape
        dice = np.empty((B, C), dtype=np.float32)
        
        for b, c in np.ndindex(B, C):
            seg_gt = y_true[b, c]
            seg_pred = y_pred[b, c]
            volume_sum = seg_gt.sum() + seg_pred.sum()
            if volume_sum == 0:
                dice[b, c] = np.nan
            volume_intersect = np.logical_and(seg_gt, seg_pred).sum()
            dice[b, c] = 2*volume_intersect / volume_sum
        
        return dice
       
class HausdorffDistanceMetric(IterationMetric):
    def __init__(self, percentile=95, spacing=[1.0, 1.0], directed=False) -> None:
        super().__init__()
        self.percentile = percentile
        self.spacing = spacing
        self.directed = directed

    def _compute_tensor(self, y_pred: np.ndarray, y_true: np.ndarray | None = None, **kwargs):
        assert y_true.shape == y_pred.shape, f"y_pred and y_true should have same shapes, got {y_pred.shape} and {y_true.shape}."
        
        B, C, H, W = y_pred.shape
        
        hd = np.empty((B, C), dtype=np.float32)

        for b, c in np.ndindex(B, C):
            seg_pred, seg_gt = y_pred[b, c], y_true[b, c]
            
            seg_union = np.logical_or(seg_pred,seg_gt)
            if not seg_union.any(): # not seg_pred.any() or not seg_gt.any():
                hd[b, c] = np.nan
                continue
            
            
            # compute the bounding box of the union of the two segmentations
            a = np.argwhere(seg_union)
            bb_union = np.array((np.min(a[:, 0]), np.min(a[:, 1]), np.max(a[:, 0]), np.max(a[:, 1])))
            # add a margin to the bounding box
            margin = 1
            bb_union += np.array((-margin, -margin, margin, margin))
            # clip to the image size
            bb_union = np.clip(bb_union, 0, seg_pred.shape[-2])

            # crop the bounding box to the minimum size for efficiency
            s = np.s_[bb_union[0]:bb_union[2], bb_union[1]:bb_union[3]]
            seg_pred, seg_gt = seg_pred[s], seg_gt[s]
            
            # compute the edges of the segmentations
            edges_pred = np.logical_xor(binary_erosion(seg_pred), seg_pred)
            edges_gt = np.logical_xor(binary_erosion(seg_gt), seg_gt)

            # if no edges are present, the distance is infinite
            if not edges_gt.any() or not edges_pred.any():
                hd[b, c] = np.inf
                continue

            # compute the distance transform with scipy distance_transform_edt
            distances = scipy_distance_transform_edt(
                input=(~edges_gt), 
                sampling=self.spacing
            )
            # compute the houseforff distance
            distances = distances.astype(np.float32)[edges_pred]
            hd[b, c] = np.quantile(distances, self.percentile / 100)

            # if directed, compute the distance from the other direction and take the maximum
            if not self.directed:
                distances2 = scipy_distance_transform_edt(
                    input=(~edges_pred), 
                    sampling=self.spacing
                )
                distances2 = distances2.astype(np.float32)[edges_gt]
                hd[b, c] = max(hd[b, c], np.quantile(distances2, self.percentile / 100))

        return hd

class SurfaceDistanceMetric(IterationMetric):
    def __init__(self, spacing = [1.0, 1.0]) -> None:
        super().__init__()
        self.spacing = spacing

    def _compute_tensor(self, y_pred: np.ndarray, y_true: np.ndarray | None = None, **kwargs):
        assert y_true.shape == y_pred.shape, f"y_pred and y_true should have same shapes, got {y_pred.shape} and {y_true.shape}."
        

        B, C, H, W = y_pred.shape
        asd = np.empty((B, C), dtype=np.float32)

        for b, c in np.ndindex(B, C):
            seg_pred, seg_gt = y_pred[b, c], y_true[b, c]
            
            seg_union = np.logical_or(seg_pred,seg_gt)
            if not seg_union.any(): # not seg_pred.any() or not seg_gt.any():
                asd[b, c] = np.nan
                continue
            
            # compute the bounding box of the union of the two segmentations
            a = np.argwhere(seg_union)
            bb_union = np.array((np.min(a[:, 0]), np.min(a[:, 1]), np.max(a[:, 0]), np.max(a[:, 1])))
            # add a margin to the bounding box
            margin = 1
            bb_union += np.array((-margin, -margin, margin, margin))
            # clip to the image size
            bb_union = np.clip(bb_union, 0, seg_pred.shape[-2])

            # crop the bounding box to the minimum size for efficiency
            s = np.s_[bb_union[0]:bb_union[2], bb_union[1]:bb_union[3]]
            seg_pred, seg_gt = seg_pred[s], seg_gt[s]
            
            # compute the edges of the segmentations
            edges_pred = np.logical_xor(binary_erosion(seg_pred), seg_pred)
            edges_gt = np.logical_xor(binary_erosion(seg_gt), seg_gt)
            
            # if no edges are present, the distance is infinite
            if not edges_gt.any() or not edges_pred.any():
                asd[b, c] = np.inf
                continue

            # compute the distance transform with scipy distance_transform_edt
            distances = scipy_distance_transform_edt(
                input=(~edges_gt), 
                sampling=self.spacing
            )
            
            # compute the average surface distance
            distances = distances.astype(np.float32)[edges_pred]
            asd[b, c] = np.mean(distances)

        return asd

class EuclideanCenterDistanceMetric(IterationMetric):
    def __init__(self) -> None:
        super().__init__()

    def _compute_tensor(self, y_pred: np.ndarray, y_true: np.ndarray | None = None, **kwargs):
        """
        Computation logic for `y_pred` and `y` of an iteration, the data should be "batch-first" Tensors.
        A subclass should implement its own computation logic.
        The return value is usually a "batch_first" tensor, or a list of "batch_first" tensors.
        """
        B,C,H,W = y_pred.shape
        true_com_path = np.array([scipy.ndimage.center_of_mass(y_true[i,0,:,:]) for i in range(B)])
        pred_com_path = np.array([scipy.ndimage.center_of_mass(y_pred[i,0,:,:]) for i in range(B)])

        # L2 norm of the difference between the predicted and true center of mass
        return np.linalg.norm(true_com_path - pred_com_path, axis=1)

# region Dosimetric metrics helper functions
def shift_by_centroid_diff(pred_centroid, true_centroid, true_seg):
    """Shift input segmentation by difference between predicted and true centroids.

    Args:
        pred_centroid (list): predicted centroids in SI and AP
        true_centroid (list): true centroids in SI and AP
        true_seg (arr): true binary mask with shape (h,w)

    Returns:
        arr: shifted binary mask
    """
    
    # Difference in centroids positions
    delta_centroids_SI = pred_centroid[0][0] - true_centroid[0][0]  
    delta_centroids_AP = pred_centroid[0][1] - true_centroid[0][1] 
    
    # Take last input segmentation and shift it by delta centroids  
    shifted_seg = scipy.ndimage.shift(true_seg, 
                                    shift=[delta_centroids_SI, delta_centroids_AP],
                                    order=3, mode='nearest')

    return shifted_seg

def calculate_dvh(dose_arr, label_arr, bins=1001):
    """Calculates a dose-volume histogram.
    Adapted from https://github.com/pyplati/platipy/blob/master/platipy/imaging/dose/dvh.py

    Args:
        dose_arr (numpy.ndarray): The dose grid.
        label_arr (numpy.ndarray): The (binary) label defining a structure.
        bins (int | list | np.ndarray, optional): Passed to np.histogram,
            can be an int (number of bins), or a list (specifying bin edges). Defaults to 1001.

    Returns:
        bins (numpy.ndarray): The points of the dose bins
        values (numpy.ndarray): The DVH values
    """

    # Check that dose and label array have the same shape
    if dose_arr.shape != label_arr.shape:
        raise ValueError("Dose grid size does not match label, please resample!") 

    # Get dose values for the structure
    dose_vals = dose_arr[np.where(label_arr)]

    # Calculate the histogram
    counts, bin_edges = np.histogram(dose_vals, bins=bins)

    # Get mid-points of bins
    bins = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    # Calculate the actual DVH values
    values = np.cumsum(counts[::-1])[::-1]
    
    if np.all(values == 0):
        return bins, values
    else:
        values = values / values.max()
        return bins, values

def calculate_dvh_for_labels(dose_array, labels, bin_width=0.1, max_dose=None, spacing=(1,1,1)):
    """Calculate the DVH for multiple labels.
    Adapted from https://github.com/pyplati/platipy/blob/master/platipy/imaging/dose/dvh.py

    Args:
        dose_array (np.ndarray): Dose grid
        labels (dict): Dictionary of labels with the label name as key and binary mask np.ndarray as
            value.
        bin_width (float, optional): The width of each bin of the DVH (Gy). Defaults to 0.1.
        max_dose (float, optional): The maximum dose of the DVH. If not set then maximum dose from
            dose grid is used.Defaults to None.
        spacing (tuple, optional): The voxel spacing of the dose grid. Defaults to (1,1,1). Should be the same as for the masks!

    Returns:
        pandas.DataFrame: The DVH for each structure along with the mean dose and size in cubic
            centimetres as a data frame.
    """

    dvh = []

    label_keys = labels.keys()

    if not max_dose:
        max_dose = dose_array.max()

    for k in label_keys:
        # Get mask from dict
        mask_array = labels[k]

        # Compute cubic centimetre volume of structure
        cc = mask_array.sum() * np.prod([a / 10 for a in spacing])

        bins, values = calculate_dvh(
            dose_array, mask_array, bins=np.arange(-bin_width / 2, max_dose + bin_width, bin_width)
        )

        # Remove rounding error
        bins = np.round(
            bins.astype(float),
            decimals=10,
        )

        mean_dose = dose_array[mask_array > 0].mean()
        entry = {
            **{
                "label": k,
                "cc": cc,
                "mean": mean_dose,
            },
            **dict(zip(bins, values)),
        }

        dvh.append(entry)

    return pd.DataFrame(dvh)

def calculate_d_x(dvh, x, label=None):
    """Calculate the dose which x percent of the volume receives

    Args:
        dvh (pandas.DataFrame): DVH DataFrame as produced by calculate_dvh_for_labels
        x (float|list): The dose threshold (or list of dose thresholds) which x percent of the
            volume receives
        label (str, optional): The label to compute the metric for. Computes for all metrics if not
            set. Defaults to None.

    Returns:
        pandas.DataFrame: Data frame with a row for each label containing the metric and value.
    """

    if label:
        dvh = dvh[dvh.label == label]

    if not isinstance(x, list):
        x = [x]

    bins = np.array([b for b in dvh.columns if isinstance(b, float)])
    values = np.array(dvh[bins])

    metrics = []
    for idx in range(len(dvh)):
        d = dvh.iloc[idx]

        m = {"label": d.label}

        for threshold in x:
            value = np.interp(threshold / 100, values[idx][::-1], bins[::-1])
            if values[idx, 0] == np.sum(values[idx]):
                value = 0

            # Interp will return zero when computing D100, do compute this separately
            if threshold == 100:
                i, j = np.where(values == 1.0)
                value = bins[j][i == idx][-1]

            m[f"D{threshold}"] = value

        metrics.append(m)

    return pd.DataFrame(metrics)

def calculate_v_x(dvh, x, label=None):
    """Get the volume (in cc) which receives x dose

    Args:
        dvh (pandas.DataFrame): DVH DataFrame as produced by calculate_dvh_for_labels
        x (float|list): The dose threshold (or list of dose thresholds) to get the volume for.
        label (str, optional): The label to compute the metric for. Computes for all metrics if not
            set. Defaults to None.

    Returns:
        pandas.DataFrame: Data frame with a row for each label containing the metric and value.
    """

    if label:
        dvh = dvh[dvh.label == label]

    if not isinstance(x, list):
        x = [x]

    bins = np.array([b for b in dvh.columns if isinstance(b, float)])
    values = np.array(dvh[bins])

    metrics = []
    for idx in range(len(dvh)):
        d = dvh.iloc[idx]

        m = {"label": d.label}

        for threshold in x:
            value = np.interp(threshold, bins, values[idx]) * d.cc

            metric_name = f"V{threshold}"
            if threshold - int(threshold) == 0:
                metric_name = f"V{int(threshold)}"

            m[metric_name] = value

        metrics.append(m)

    return pd.DataFrame(metrics)

def calculate_d_cc_x(dvh, x, label=None):
    """Compute the dose which is received by cc of the volume

    Args:
        dvh (pandas.DataFrame): DVH DataFrame as produced by calculate_dvh_for_labels
        x (float|list): The cc (or list of cc's) to compute the dose at.
        label (str, optional): The label to compute the metric for. Computes for all metrics if not
            set. Defaults to None.

    Returns:
        pandas.DataFrame: Data frame with a row for each label containing the metric and value.
    """

    if label:
        dvh = dvh[dvh.label == label]

    if not isinstance(x, list):
        x = [x]

    metrics = []
    for idx in range(len(dvh)):

        d = dvh.iloc[idx]
        m = {"label": d.label}

        for threshold in x:
            cc_at = (threshold / dvh[dvh.label == d.label].cc.iloc[0]) * 100
            cc_at = min(cc_at, 100)
            cc_val = calculate_d_x(dvh[dvh.label == d.label], cc_at)[f"D{cc_at}"].iloc[0]

            m[f"D{threshold}cc"] = cc_val

        metrics.append(m)

    return pd.DataFrame(metrics)

#endregion

class DoseMetric(IterationMetric):
    def __init__(self) -> None:
        super().__init__()

    def _compute_tensor(self, y_pred: np.ndarray, y_true: np.ndarray | None = None, sigma=4, **kwargs):
        """
        Computation logic for `y_pred` and `y` of an iteration, the data should be "batch-first" Tensors.
        A subclass should implement its own computation logic.
        The return value is usually a "batch_first" tensor, or a list of "batch_first" tensors.
        """
        # Filter out empty targets
        empty_targets = np.sum(y_true, axis=(1,2,3)) == 0
        y_pred = y_pred[~empty_targets]
        y_true = y_true[~empty_targets]

        B,C,H,W = y_pred.shape
        
        initial_target_mask = y_true[0,0,:,:]

        # Add an isotropic 3 pixel/mm margin to the target
        expanded_initial_target_mask = scipy.ndimage.binary_dilation(initial_target_mask, structure=scipy.ndimage.generate_binary_structure(2, 1), iterations=3).astype(initial_target_mask.dtype)
        
        # Smooth the expanded target mask with a gaussian filter
        expanded_initial_target_mask = (scipy.ndimage.gaussian_filter(expanded_initial_target_mask.astype(np.float32), sigma=sigma)> 0.5).astype(initial_target_mask.dtype) # sigma=4 or 6
    
        labels = {"GTV": initial_target_mask}
        d98_original = calculate_d_x(calculate_dvh_for_labels(expanded_initial_target_mask, labels), [2, 98]).loc[0, 'D98']

        targets_com = np.array([scipy.ndimage.center_of_mass(y_true[i,0,:,:]) for i in range(B)])
        outputs_com = np.array([scipy.ndimage.center_of_mass(y_pred[i,0,:,:]) for i in range(B)])
    
        # Store final summed up shifted dose for current patient
        final_shifted_dose = np.zeros(shape=(H,W), dtype=np.float32)
        
        for i in range(B):
            # Shift by centroid difference and sum up current dose to final one
            final_shifted_dose += shift_by_centroid_diff(outputs_com[i:i+1], targets_com[i:i+1], expanded_initial_target_mask)
        
        # Divide by number of shifts to normalize dose
        final_shifted_dose /= B

        # Compute DVH for final dose
        d98_final = calculate_d_x(calculate_dvh_for_labels(final_shifted_dose, labels), [2, 98]).loc[0, 'D98']

        # Get realtive d98
        relative_d98_dose = d98_final  / d98_original
        return np.array((relative_d98_dose,))

# Create instances of the metrics
dice_metric = DiceMetric()
hausdorff_distance_95_metric = HausdorffDistanceMetric(percentile=95)
surface_distance_avg_metric = SurfaceDistanceMetric()
center_distance_metric = EuclideanCenterDistanceMetric()
dose_metric = DoseMetric()
