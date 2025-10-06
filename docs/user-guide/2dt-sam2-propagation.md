# 2D+t SAM2 Based Propagation

The 2D+t SAM2 based propagation feature allows you to propagate segmentation masks across time-series data. This is ideal for videos or time-lapse microscopy images.

## How to Use 2D+t Propagation

1. **Load a Time-Series Image**:
   - Ensure the image layer is a time-series (2D+t) dataset.

2. **Add Initial Prompts**:
   - Use points, bounding boxes, or masks to define the segmentation for the first frame.

3. **Run Propagation**:
   - Click the "Propagate" button to propagate the segmentation to the next frame.
   - Use the "Run" button to propagate until the end of the time-series.

4. **Adjust Settings**:
   - Use the "Propagation Dimension" spinbox to select the dimension to propagate over.
   - Enable "Reverse Direction" to propagate backward.
   - Check "Overwrite Existing" to replace existing masks.

5. **Review Results**:
   - Use the Preview Layer to review the propagated masks.

## Placeholder for Images
- Add images showing the propagation process and results.