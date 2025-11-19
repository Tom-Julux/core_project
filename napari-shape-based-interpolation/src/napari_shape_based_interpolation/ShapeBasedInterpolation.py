#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:24:19 2019

@author: saduque
"""
import SimpleITK
from scipy.interpolate import interpn
from skimage.morphology import opening, square
import numpy as np
import matplotlib.pyplot as plt


def shape_based_interpolation(img, img_voxel_size, dose_voxel_size, img_axes=None, dose_axes=None, \
                              nn_at_edges=True, ax_dim = 2, cor_dim = 0, sag_dim = 1):
    '''
      Description
      -----------
      Smart interpolation between slices using shape based interpolation methods from Herman et al. (1992).
      Binary structure slices in ct coordinates are interpolated to a target grid (e.g. dose geometry) by  
      first creating a signed distance map which contains the distance of every pixel position to the structure boundary. 
      These several 2D distance map slices are then interpolated linearly in 3D and then converted back into a 3D binary mask.
      
      If there is a large discrepancy between img_voxel_size and dose_voxel_size, there are slices missing at the edges 
      (assuming axial slices extend 1/2 of a voxel outside of the last slice position). To replace these slices, linear 
      3D interpolation (non-shape-based!) can be used by setting nn_at_edges to false. This takes longer, the quicker approach
      is to use nearest-neighbour interpolation at edges by setting nn_at_edges to true. There the last slice at the boundary 
      is copied as often as necessary to fill the whole new structure geometry in the dose space.
      
      This function is trying to reproduce the contour interpolation method in RayStation, which is also based on the paper 
      of Herman et al. and solves interpolation at edges by adding a "hat" (for more details please refer to the RayStation
      Reference Manual). 
      
      The current version of the code has been extended to account for 
          - Multiple tumor regions seprated by air gaps in SI direction
          - Single axial slices, which are linearly interpolated assuming an empty slice above and below
          - Holes inside the input binary mask, which are filled using SITK iterative hole filling function
          - Single voxel islands outside the binary mask, which are erased using an opening morphological filter
      
      Before using this function, perform a test by running test1.py or test2.py using the example data supplied with the function
    
      Parameters
      ----------
      img: numpy-array
          image to be interpolated (e.g. a binary mask as created with plastimatch)
      img_axes: tuple or list (optional: if physical coordinates of images are given)
          axes along image extensions, stored as a tuple or list of 3 numpy arrays for the three axes
      img_voxel_size: numpy-array
          voxel size of the image
      dose_axes: tuple or list (optional: if physical coordinates of images are given) 
          axes along dose cube extensions, stored as a tuple or list of 3 numpy arrays for the three axes
      dose_voxel_size: numpy-array
          voxel size of the dose cube (i.e. target voxel size of resampling procedure)
      nn_at_edges: boolean 
          True -> use nearest-neighbourinterpolation at edges; false -> use linear interpolation at edges; 
          None --> do not apply correction at edges (suggested if tumor has multiple separated regions --> only first and last edge are corrected)
      ax_dim, cor_dim, sag_dim: integer
          index of the axial, coronal and sagittal orientation (must be the same for image, axes and voxel sizes!)
      
      Returns
      ----------
      A shape-based interpolated version of the image in dose cube space (or any other target space)
    '''

    # check if physical axes are given, if not create arbitrary ones
    if img_axes is None or dose_axes is None:
        # arbitrary image axes from its shape
        img_axes = []
        for dim in range(img.ndim):
            img_axes.append(np.linspace(start=0, stop=img.shape[dim], num=img.shape[dim], endpoint=False))
        
        # arbitrary dose axes matching the target spacing
        dose_axes = []
        for dim in range(img.ndim):
            dose_axes.append(np.linspace(start=0, stop=img.shape[dim], \
                                         num=int(img.shape[dim]*img_voxel_size[dim]/dose_voxel_size[dim]), \
                                         endpoint=False))       
        
    
    # determine indices of axial slices containing the structure and initialize the distance map 
    axial_slices = np.unique(np.where(img>0)[ax_dim])
    # dose space meshgrid for interpolation
    x, y, z = np.meshgrid(dose_axes[cor_dim], dose_axes[sag_dim], dose_axes[ax_dim]) # shape e.g. (3, 600, 600, 420)
    
    # case in which tumor is made of  consecutive axial slices 
    # --> single region
    if len(axial_slices) == (axial_slices[-1] - axial_slices[0] + 1):
        print('Handling a single tumor region in SI direction.')
        
        distance_map = np.zeros((img.shape[cor_dim],img.shape[sag_dim],len(axial_slices)))
      
        # get the distance map for all slices containing the contour
        for slice in np.arange(0, len(axial_slices)):
            part_img = img[:,:,slice + axial_slices[0]]

            sitk_img = SimpleITK.GetImageFromArray(part_img)
            try:
                sitk_distance_map = SimpleITK.ApproximateSignedDistanceMap(sitk_img) # unfortunately not applicable to 3D image
            except RuntimeError:
#                plt.imshow(part_img)
#                plt.show()
                # fill holes first, as distance map is not well defined with them and RuntimeError is thrown
                sitk_no_holes = SimpleITK.VotingBinaryIterativeHoleFilling(sitk_img)
                print('Hole(s) successfully filled.')
#                plt.imshow(SimpleITK.GetArrayFromImage(sitk_no_holes))
#                plt.show()
                try:
                    sitk_distance_map = SimpleITK.ApproximateSignedDistanceMap(sitk_no_holes) # unfortunately not applicable to 3D image, only 2D slices
                except RuntimeError:
                    # define structuring element for morphological filter
                    selem = square(2)
                    # perform opening operation
                    opened_part_img = opening(part_img, selem)
                    print('Hole(s) were not filled, opening successfully applied instead.')
#                    plt.imshow(opened_part_img)
#                    plt.show() 
                    
                    sitk_img = SimpleITK.GetImageFromArray(opened_part_img)
                    # compute distance map
                    sitk_distance_map = SimpleITK.ApproximateSignedDistanceMap(sitk_img)
                    
            # write computed distance map to current axial slice of 3D distance_map      
            distance_map[:,:,slice] = SimpleITK.GetArrayFromImage(sitk_distance_map)
        
        # linearly interpolate the distance map 
        distance_map_interp = interpn((img_axes[sag_dim], img_axes[cor_dim], img_axes[ax_dim][axial_slices]), distance_map, np.array([y, x, z]).T, bounds_error=False, fill_value=0)
        distance_map_interp = distance_map_interp.T
        
        # make interpolated distance map a binary image
        #print('Unique elements in distance_map_interp:' + str(np.unique(distance_map_interp)))
        distance_map_interp[np.where(distance_map_interp > 0)] = 0
        distance_map_interp[np.where(distance_map_interp < 0)] = 1
        #print('Unique elements in distance_map_interp after thresholding:' + str(np.unique(distance_map_interp)))
 

    # case in which tumor is not made of consecutive axial slices 
    # --> multiple regions with no structure in between    
    else:
        print('Handling multiple tumor regions in SI direction.')

        # find out number of axial slices (both with and without tumor)
        nr_axial_slices = axial_slices[-1] - axial_slices[0] + 1
        #print('Non-empty axial slices: ' +  str(axial_slices))
        
        # determine indices with no structure within the axial slice range of the tumor
        empty_axial_slices =  np.setdiff1d(np.arange(axial_slices[0],axial_slices[-1]+1), axial_slices)
        #print('Empty axial slices in between: ' + str(empty_axial_slices))

        # empty list where current slices for image region will be stored
        current_img_slices = []
        # empty list where current slices for distance map region will be stored
        current_slices = []
       
        # define zero-valued distance map 
        distance_map = np.zeros((img.shape[cor_dim],img.shape[sag_dim], nr_axial_slices))
        # empty list where different distant maps for different regions will be stored 
        distance_map_interp_regions = []
      
        
        # get the distance maps fot the different regions
        for slice in np.arange(0, nr_axial_slices): 
            current_img_slice = slice + axial_slices[0]
            part_img = img[:,:,current_img_slice]
 
#            print('Current image slice:' + str(current_img_slice))           
#            print('Unique elements in axial part_img: ' + str(np.unique(part_img)))             
#            if current_img_slice == 53:
#                plt.imshow(part_img)
#                plt.show()
#                print(part_img)
#                sitk_img = SimpleITK.GetImageFromArray(part_img)
#                print(sitk_img)
#                print(sitk_img.GetSize())
            
            # case with structure
            if current_img_slice in axial_slices:
                sitk_img = SimpleITK.GetImageFromArray(part_img)
                try:
                    sitk_distance_map = SimpleITK.ApproximateSignedDistanceMap(sitk_img) # unfortunately not applicable to 3D image
                except RuntimeError:
                    # fill holes first, as distance map is not well defined with them and RuntimeError would be thrown
                    sitk_no_holes = SimpleITK.VotingBinaryIterativeHoleFilling(sitk_img)
                    print('Hole(s) successfully filled.')
#                    plt.imshow(SimpleITK.GetArrayFromImage(sitk_no_holes))
#                    plt.show()
                    try:
                        sitk_distance_map = SimpleITK.ApproximateSignedDistanceMap(sitk_no_holes) # unfortunately not applicable to 3D image
                    except RuntimeError:
                        # define structuring element for morphological filter
                        selem = square(2)
                        # perform opening operation
                        opened_part_img = opening(part_img, selem)
                        print('Hole(s) were not filled, opening successfully applied instead.')
#                        plt.imshow(opened_part_img)
#                        plt.show() 
                        
                        sitk_img = SimpleITK.GetImageFromArray(opened_part_img)
                        # compute distance map
                        sitk_distance_map = SimpleITK.ApproximateSignedDistanceMap(sitk_img)

                # write computed distance map to current axial sliceof 3D distance_map      
                distance_map[:,:,slice] = SimpleITK.GetArrayFromImage(sitk_distance_map)     
                
                # append image slices and distance map slice for current region to list, to be used for interpolation
                current_img_slices.append(current_img_slice)
                current_slices.append(slice)


            # linearly interpolate the distance map when point of crossing from structure to no-structure is reached
            if (current_img_slice in axial_slices) and \
               ((current_img_slice+1) in empty_axial_slices):
                #print('Structure to no-structure crossing found.')
                
                if (current_img_slice in axial_slices) and \
                   ((current_img_slice+1) in empty_axial_slices) and \
                   ((current_img_slice-1) in empty_axial_slices):                   
                    print('Single slice structure found.')
                    
                    # replace the single distance map slice with the original slice in image space
                    distance_map[:,:,slice] = part_img
                   
#                    print(f"img_axes shape: {np.shape(img_axes[ax_dim][current_img_slices[0]-1:current_img_slices[0]+2])} ") # (3,)
#                    print(f"distance_map shape: {np.shape(distance_map[:,:,current_slices[0]-1:current_slices[0]+2])} ") # (512, 512, 3)

                    # perform linear interpolation with original image by using empty slice above and below
                    distance_map_interp = interpn((img_axes[sag_dim], img_axes[cor_dim], img_axes[ax_dim][current_img_slices[0]-1:current_img_slices[0]+2]), \
                                                  distance_map[:,:,current_slices[0]-1:current_slices[0]+2], np.array([y, x, z]).T, bounds_error=False, fill_value=0)

                    # as image is in [0,1], values have to be changed into fake signed distance map values (negative  inside contour)
                    # the 0.2 threshold is used to avoid a stacking of slices with all the same contour
                    distance_map_interp[distance_map_interp>0.2] = -1
                    
                    distance_map_interp_regions.append(distance_map_interp.T) 
                    
                elif (current_img_slice in axial_slices) and \
                   ((current_img_slice+1) in empty_axial_slices) and \
                   ((current_img_slice-1) < axial_slices[0]):                   
                    print('Single slice structure found at the beginning of tumor region.')
                    
                    # replace the single diatance map slice with the original slice in image space
                    distance_map[:,:,slice] = part_img

                    # perform linear interpolation with original image by using empty slice above
                    distance_map_interp = interpn((img_axes[sag_dim], img_axes[cor_dim], img_axes[ax_dim][current_img_slices[0]:current_img_slices[0]+2]), \
                                                  distance_map[:,:,current_slices[0]:current_slices[0]+2], np.array([y, x, z]).T, bounds_error=False, fill_value=0)
 
                    # as image is in [0,1], values have to be changed into fake signed distance map values (negative  inside contour)
                    distance_map_interp[distance_map_interp>0.2] = -1
                    
                    distance_map_interp_regions.append(distance_map_interp.T)                

                else:              
                    # interpolate the distance map using only the current_img_slices for image
                    # and current_slices for distance map 
                    distance_map_interp = interpn((img_axes[sag_dim], img_axes[cor_dim], img_axes[ax_dim][np.array(current_img_slices)]), \
                                                  distance_map[:,:,np.array(current_slices)], np.array([y, x, z]).T, bounds_error=False, fill_value=0)
                    # append interpolated distance map containing the axial slices of the current region plus empty slice
                    # for every other region in the target space
                    distance_map_interp_regions.append(distance_map_interp.T)  
                
                # re-define empty list for possible next round of regions
                current_img_slices = []
                current_slices = []
                
            # linearly interpolate the distance map when overall last slice with structure is reached
            if (current_img_slice in axial_slices) and \
               ((current_img_slice+1) > axial_slices[-1]):
                #print('End of tumor reached (superiorest part).')

                if (current_img_slice in axial_slices) and \
                   ((current_img_slice-1) in empty_axial_slices):                   
                    print('Single slice structure found at the end of the tumor region.')
            
                    # replace the single diatance map slice with the original slice in image space
                    distance_map[:,:,slice] = part_img
                    #distance_map_padded = np.stack([np.zeros(distance_map.shape),distance_map],axis=ax_dim)
                
                    # perform linear interpolation with original image by using empty slice below
                    distance_map_interp = interpn((img_axes[sag_dim], img_axes[cor_dim], img_axes[ax_dim][current_img_slices[0]-1:current_img_slices[0]+1]), \
                                                  distance_map[:,:,current_slices[0]-1:], np.array([y, x, z]).T, bounds_error=False, fill_value=0)
                    
                    # as image is in [0,1], values have to be changed into fake signed distance map values (negative  inside contour)
                    distance_map_interp[distance_map_interp>0.2] = -1
                    
                    distance_map_interp_regions.append(distance_map_interp.T) 
                
                else:
                    # interpolate the distance map using only the current_img_slices for image
                    # and current_slices for distance map 
                    distance_map_interp = interpn((img_axes[sag_dim], img_axes[cor_dim], img_axes[ax_dim][np.array(current_img_slices)]), \
                                                  distance_map[:,:,np.array(current_slices)], np.array([y, x, z]).T, bounds_error=False, fill_value=0)
                    distance_map_interp_regions.append(distance_map_interp.T)  


        # make interpolated distance maps a binary image
        #print('Unique elements in distance_map_interp:' + str(np.unique(distance_map_interp)))
        for region in distance_map_interp_regions:
            region[np.where(region > 0)] = 0
            region[np.where(region < 0)] = 1
        #print('Unique elements in distance_map_interp after thresholding:' + str(np.unique(distance_map_interp_regions))) # [0. 1.]
    
        # after loop over regions is done, sum (as the sum will be either 0 or 1 for a given voxel) 
        # different axial regions along axial dimension
        distance_map_interp = np.sum(distance_map_interp_regions, axis=0)
        #print('Shape after summing: ' + str(np.shape(distance_map_interp))) # e.g. (600, 600, 363)
        
        

    # in the above code: slices are cut off at boundaries --> the next part corrects for that
    if nn_at_edges is not None:    
        ############################################################
    
        indices_structure = np.unique(np.where(img>0)[ax_dim])  
        indices_structure_int = np.unique(np.where(distance_map_interp>0)[ax_dim])  
        
        # get the coordinates of the structure extensions in axial dimension, both in image and dose space
        maxcoord_structure = img_axes[ax_dim][np.max(indices_structure)]
        mincoord_structure = img_axes[ax_dim][np.min(indices_structure)]
        maxcoord_structure_int = dose_axes[ax_dim][np.max(indices_structure_int)]
        mincoord_structure_int = dose_axes[ax_dim][np.min(indices_structure_int)]
        
        x, y = np.meshgrid(dose_axes[cor_dim], dose_axes[sag_dim])
        
        # both structure edges are added sequentially, starting with the maximum index values
        # d corresponds to the new structure coordinate in dose space where a slice is added
        # slices are added as long as the distance between the last structure coordinate and the new coordinate 
        # is smaller than half the image voxel size
        d = maxcoord_structure_int + dose_voxel_size[ax_dim]
        index_structure_int = np.max(indices_structure_int) + 1
        img_int = interpn((img_axes[sag_dim],img_axes[cor_dim]),img[:,:,np.max(indices_structure)],np.array([y, x]).T, bounds_error=False, fill_value=0)        
        
        while (abs(d - maxcoord_structure) < img_voxel_size[ax_dim]/2):
            distance_map_interp[:,:,index_structure_int] = img_int.T
            print("Slice added at boundary")
            d = d + dose_voxel_size[ax_dim]
            
        # see above    
        d = mincoord_structure_int - dose_voxel_size[ax_dim]
        index_structure_int = np.min(indices_structure_int) - 1
        img_int = interpn((img_axes[sag_dim],img_axes[cor_dim]),img[:,:,np.min(indices_structure)],np.array([y, x]).T, bounds_error=False, fill_value=0)

        while (abs(d - mincoord_structure) < img_voxel_size[ax_dim]/2):
            distance_map_interp[:,:,index_structure_int] = img_int.T
            print("Slice added at boundary")
            d = d - dose_voxel_size[ax_dim]
            
        # make whole distance map binary, including caps at the ends
        distance_map_interp[distance_map_interp < 0.5] = 0
        distance_map_interp[distance_map_interp >= 0.5] = 1

    # return shape based interpolated image
    return distance_map_interp
