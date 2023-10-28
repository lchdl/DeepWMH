from deepwmh.utilities.parallelization import run_parallel
from deepwmh.utilities.data_io import load_nifti, save_nifti, try_load_nifti
from typing import Union
import warnings
import numpy as np
import numpy.ma as ma
import scipy.ndimage
from scipy.ndimage import label
from scipy.ndimage import zoom
from scipy.ndimage import binary_erosion
from skimage.filters import threshold_otsu

def masked_mean(data, mask):
    mask = (mask>0.5).astype('int')
    masked_data = ma.masked_array(data,mask=1-mask)
    return masked_data.mean()

def masked_std(data, mask):
    mask = (mask>0.5).astype('int')
    masked_data = ma.masked_array(data,mask=1-mask)
    return masked_data.std()

def average_contiguous_labels(labels: list):
    '''
    averaging multiple label maps with contiguous label id.
    '''
    num_channels = 0
    label_shape = labels[0].shape
    for label in labels:
        if num_channels < int(np.max(label)) + 1:
            num_channels = int(np.max(label)) + 1
    channels = np.zeros([num_channels, *label_shape])
    for label in labels:
        int_label = label.astype('int')
        for ch in range(num_channels):
            channels[ch] += (int_label==ch).astype('float32')
    averaged_label = np.argmax(channels, axis=0)
    return averaged_label

def map_label(label: np.ndarray, src_ids: list, dst_ids: list):
    '''
    mapping label from a set of ids to another set of ids.
    src_ids: a list of ints representing source ids
    dst_ids: a list of ints representing target ids
    for example, src_ids = [1,5] and dst_ids = [2,4] will
    map id from 1 to 2 and 5 to 4.
    '''
    assert len(src_ids) == len(dst_ids), 'invalid id mapping.'

    i_label = np.around(label).astype('int')
    i_label_new = np.zeros(i_label.shape).astype('int')
    for src_id , dst_id in zip(src_ids, dst_ids):
        i_label_new[i_label == src_id] = dst_id
    return i_label_new

def mean_std_grid(data: np.ndarray, patch_size: Union[list, tuple], 
    order=1, mask: Union[np.ndarray, None] = None):
    '''
    
    Description
    ------------

    Calculate coarse mean & std value estimation grid for a given image.

    Usage
    ------------

    >>> mean_interp, std_interp = mean_std_grid(data, [64,64,64], order = 1)
    >>> mean_interp, std_interp = mean_std_grid(data, [64,64,64], order = 1, mask = mask)

    Parameters
    ------------

    data: np.ndarray
        input image data

    patch_size: list | tuple
        a list/tuple of ints representing the patch size when calculating
        local mean values using sliding window (in voxels). For example, 
        [64, 64, 64]. Any dimension size in patch_size must be even (odd
        number will be converted to even number before calculation).

    order: int
        interpolation order. 0 = nearest, 1 = linear (default), 
        this value can be up to 4, see scipy.ndimage.interpolation.zoom(...) 
        for more info. If you don't want any overshoot just leave it to 1 
        (default).

    mask: np.ndarray [optional]
        if given, then only voxels inside the mask (where mask value = 1) 
        will be counted when calculating mean and std (regions where mask
        value = 0 will be ignored).
    
    Returns
    ------------

    mean_interp: np.ndarray
        interpolated coarse local mean value estimation

    std_interp: np.ndarray
        interpolated coarse local std value estimation
    
    '''

    # ensure each dim in patch size is even.
    patch_size = list( 2 * np.ceil( np.array(patch_size)/2 ).astype('int')) 
    step_size = [patch_size[0]//2, patch_size[1]//2, patch_size[2]//2]
    
    # pad images
    # data = _pre_pad_volume(data, patch_size)
    # if mask is not None:
    #     mask = _pre_pad_volume(mask, patch_size)

    # pad images again, so that shape is divisible by patch size.
    data_shape = data.shape
    padded_data_shape = list( np.array(patch_size) * np.ceil(np.array( data_shape ) / np.array( patch_size )).astype('int'))
    padded_data = np.zeros(padded_data_shape).astype('float32')
    padded_data[:data_shape[0],:data_shape[1],:data_shape[2]] = data # fill data, out of bound voxels are filled with 0 
    
    if mask is not None:
        padded_mask = np.zeros_like(padded_data)
        padded_mask[:data_shape[0],:data_shape[1],:data_shape[2]] = (mask > 0.5).astype('float32')

    grid_shape = [ padded_data_shape[0] // step_size[0], 
                   padded_data_shape[1] // step_size[1],
                   padded_data_shape[2] // step_size[2] ]
    mean_grid, std_grid = np.zeros(grid_shape), np.zeros(grid_shape)
    for ib in range(0, padded_data_shape[0], step_size[0]):
        for jb in range(0, padded_data_shape[1], step_size[1]):
            for kb in range(0, padded_data_shape[2], step_size[2]):
                ie, je, ke = ib + patch_size[0], jb + patch_size[1], kb + patch_size[2]
                i, j, k = ib//step_size[0], jb//step_size[1], kb//step_size[2]
                if mask is not None:
                    data_block = padded_data[ib:ie, jb:je, kb:ke]
                    mask_block = padded_mask[ib:ie, jb:je, kb:ke]
                    if np.sum(mask_block) > 0:
                        data_block_masked = np.ma.masked_array(data_block, mask=1-mask_block)
                        mu = data_block_masked.mean()
                        sigma = data_block_masked.std()
                    else:
                        mu, sigma = 0, 0.00001
                else:
                    data_block = padded_data[ib:ie, jb:je, kb:ke]
                    mu = np.mean(data_block)
                    sigma = np.max([ np.std(data_block), 0.00001 ]) # to avoid division by zero

                mean_grid[i,j,k] = mu
                std_grid[i,j,k] = sigma

    _grid_shape = [grid_shape[0]+2, grid_shape[1]+2, grid_shape[2]+2]
    _mean_grid, _std_grid = np.zeros(_grid_shape), np.zeros(_grid_shape)
    _mean_grid[1:1+grid_shape[0], 1:1+grid_shape[1], 1:1+grid_shape[2]] = mean_grid # surround by zeros
    _std_grid[1:1+grid_shape[0], 1:1+grid_shape[1], 1:1+grid_shape[2]] = std_grid
    _mean_interp = zoom(_mean_grid, step_size, order=order)
    _std_interp = zoom(_std_grid, step_size, order=order)
    
    grid_shape = [ mean_grid.shape[0]*step_size[0], 
               mean_grid.shape[1]*step_size[1], 
               mean_grid.shape[2]*step_size[2] ]
    offset = [ step_size[0]//2, step_size[1]//2, step_size[2]//2 ]
    mean_interp = _mean_interp[ offset[0]:offset[0]+grid_shape[0],
                                offset[1]:offset[1]+grid_shape[1],
                                offset[2]:offset[2]+grid_shape[2] ]
    std_interp = _std_interp[ offset[0]:offset[0]+grid_shape[0],
                              offset[1]:offset[1]+grid_shape[1],
                              offset[2]:offset[2]+grid_shape[2] ]
    mean_interp = mean_interp[:data_shape[0], :data_shape[1], :data_shape[2]]
    std_interp = std_interp[:data_shape[0], :data_shape[1], :data_shape[2]]

    return mean_interp, std_interp

def z_score(data, mask=None):
    '''
    perform z-score normalization for image data.
    '''
    data_mean = np.mean(data) if mask is None else masked_mean(data, mask)
    data_std = np.std(data) if mask is None else masked_std(data, mask)
    data_std = np.max( [ data_std, 0.00001 ] ) # avoid division by zero
    return (data - data_mean) / data_std

def median_filter(data, kernel_size):
    data = scipy.ndimage.median_filter(data, size=kernel_size, mode='constant', cval=0)
    return data

def mean_filter(data, kernel_size):
    data = scipy.ndimage.uniform_filter(data, size=kernel_size, mode='constant', cval=0)
    return data

def min_filter(data, kernel_size):
    data = scipy.ndimage.minimum_filter(data, size=kernel_size, mode='constant', cval=0)
    return data

def max_filter(data, kernel_size):
    data = scipy.ndimage.maximum_filter(data, size=kernel_size, mode='constant', cval=0)
    return data

def group_std(data_list, masks=None): # mask=1 means count, mask=0 means not count
    if masks is None:
        masks = [None] * len(data_list)
    else:
        assert len(masks) == len(data_list)
    dshape_runtime = None
    all_data = []
    for data, mask in zip(data_list,masks):
        if mask is not None:
            data=np.where(mask<0.5,np.nan, data) # ignore masked data values by replacing them to nan
        dshape_runtime = data.shape
        data=np.reshape(data,[-1])
        all_data.append(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data_std = np.reshape(np.nanstd(np.vstack(all_data),axis=0),dshape_runtime)
    return data_std

def group_mean(data_list, masks=None):
    if masks is None:
        masks = [None] * len(data_list)
    else:
        assert len(masks) == len(data_list)
    dshape_runtime = None
    all_data = []
    for data,mask in zip(data_list,masks):
        if mask is not None:
            data=np.where(mask<0.5, np.nan, data)
        dshape_runtime = data.shape
        data=np.reshape(data,[-1])
        all_data.append(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data_std = np.reshape(np.nanmean(np.vstack(all_data),axis=0),dshape_runtime)
    return data_std

def connected_components(mask, return_labeled=True):
    '''
    Description
    -----------
    Get number of connected components and their volumes.
    0 is considered as background and is not counted in 
    connected components. If "return_volumes" is True, a
    list will be returned containing volumes of each component,
    otherwise a total number of connected component (int) 
    is returned.

    Usage
    -----------
    >>> num_parts, labeled_array = connected_comps(mask)
    >>> num_parts = connected_comps(mask, return_labeled = False)
    '''
    mask = (mask>0.5).astype('int')
    labeled_array, num_parts = label(mask)
    if return_labeled:
        return num_parts, labeled_array
    else:
        return num_parts

def component_filtering(mask: np.ndarray, voxel_size: list, return_type='float32', erosion=True):
    '''
    quickly refines the calculated brain mask (get rid of sparks near the brain)
    '''

    def _max_volume_component(mask, return_type='float32'):
        mask = (mask>0.5).astype('int')
        labeled_array, num_features = label(mask)
        max_v,max_i = 0, None
        if num_features == 0:
            return np.zeros_like(labeled_array).astype(return_type)
        else:
            for i in range(1,num_features+1):
                v = np.sum(labeled_array == i)
                if max_v < v:
                    max_v, max_i = v, i
            assert max_i is not None
            return (labeled_array==max_i).astype(return_type)

    volume_shape = mask.shape # [S, C, A]

    # we just need to filter one direction for thick-slice data
    min_size, max_size = np.min(voxel_size), np.max(voxel_size)
    do_filtering = [False, False, False]
    if max_size / min_size > 3: # thick-slice data
        do_filtering[np.argmax(voxel_size)] = True # do filtering for recon dir only
    else: # thin-slice data
        do_filtering = [True, True, True]
    
    # sagittal
    volume_sag = np.zeros_like(mask)
    for sag in range(volume_shape[0]):
        if do_filtering[0]:
            volume_sag[sag,:,:] = _max_volume_component( binary_erosion(mask[sag,:,:])  )
        else:
            volume_sag[sag,:,:] = mask[sag,:,:]
    # coronal
    volume_cor = np.zeros_like(mask)
    for cor in range(volume_shape[1]):
        if do_filtering[1]:
            volume_cor[:,cor,:] = _max_volume_component( binary_erosion(mask[:,cor,:]) )
        else:
            volume_cor[:,cor,:] = mask[:,cor,:]
    # axial
    volume_axi = np.zeros_like(mask)
    for axi in range(volume_shape[2]):
        if do_filtering[2]:
            volume_axi[:,:,axi] = _max_volume_component( binary_erosion(mask[:,:,axi]) )
        else:
            volume_axi[:,:,axi] = mask[:,:,axi]
    #
    volume_union = ((volume_axi + volume_cor + volume_sag) > 0.5).astype(return_type)
    return volume_union

def otsu_thresholding(image, mask=None):
    '''
    voxels where mask=1 will be counted during thresholding,
    while mask=0 will be ignored. 
    '''
    if mask is None:
        return threshold_otsu(image)
    else:
        mask = (mask>0.5).astype('int')
        if np.sum(mask) < 1:
            return None
        else:
            image_m = ma.masked_array(image, mask=1-mask)
            return threshold_otsu( image_m.compressed() )

def remove_sparks(mask, min_volume=3, verbose=False):
    '''
    remove sparks for a given (binarized) image.
    any component smaller than min_volume will be discarded.
    '''
    mask = (mask>0.5).astype('int')
    if verbose:
        print('calculating cc...')
    labeled_array, num_features = label(mask)
    if verbose:
        print('%d cc detected.' % num_features)
    filtered_mask = np.zeros_like(mask) 
    for i in range(1, num_features+1):
        v = (labeled_array==i).sum()
        if v>=min_volume:
            filtered_mask[labeled_array==i] = 1
    if verbose:
        _, n = label(filtered_mask)
        print('cc after filtering: %d.' % n)
    return filtered_mask

def remove_3mm_sparks(mask, voxel_size):
    '''
    remove components with volume smaller than 3mm^3.
    '''
    if not isinstance(voxel_size, list):
        raise RuntimeError('voxel_size should be a list of 3 floats.')

    voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2]
    voxel_aniso = np.max(voxel_size) / np.min(voxel_size)
    
    if voxel_aniso > 3.0: 
        # maybe thick slice data
        # we remove components with volume less than 3 voxels instead of 3mm^3.
        mask0 = remove_sparks(mask, min_volume=3)
        return mask0
    else:
        # maybe thin slice data
        min_volume = int(np.around(3.0 / voxel_volume))
        if min_volume < 2:
            min_volume = 2 # remove all components with 1 voxel
        mask0 = remove_sparks( mask, min_volume=min_volume )
        return mask0

def gaussian_noise_2x2x2(shape, noise_std):
    noise_size = 2 # 2mm isotropic
    # generate 1x1x1 noise
    noise_1x1x1 = np.random.normal(0,noise_std, shape)
    # resample to 2x2x2
    zoom_order = 0
    noise_2x2x2 = zoom( noise_1x1x1, noise_size, order=zoom_order)
    noise_2x2x2 = noise_2x2x2[:shape[0],:shape[1],:shape[2]]
    return noise_2x2x2
    
def median_3mm(data, physical_voxel_size):
    # smoothing image using 3mm median filtering kernel

    # check if voxel has isotropic resolution
    data_smooth = np.zeros(data.shape)
    maxl, minl = np.max(physical_voxel_size), np.min(physical_voxel_size)
    if maxl / minl > 4.0:
        # anisotropic resolution
        max_axis = np.argmax(physical_voxel_size) # find main axis. 0: sagittal, 1:coronal, 2:axial
        if max_axis == 0:
            vox2d_res= [physical_voxel_size[1],physical_voxel_size[2]]
        elif max_axis == 1:
            vox2d_res= [physical_voxel_size[0],physical_voxel_size[2]]
        elif max_axis == 2:
            vox2d_res= [physical_voxel_size[0],physical_voxel_size[1]]
        else:
            raise RuntimeError('invalid axis "%d".' % max_axis)
        kernel_size = [
            int(3.0 / vox2d_res[0]), int(3.0 / vox2d_res[1])
        ]
        if kernel_size[0]<3: kernel_size[0]=3
        if kernel_size[1]<3: kernel_size[1]=3

        for s in range(data.shape[max_axis]):        
            if max_axis == 0:
                data_smooth[s,:,:] = median_filter(data[s,:,:], kernel_size)
            elif max_axis == 1:
                data_smooth[:,s,:] = median_filter(data[:,s,:], kernel_size)
            elif max_axis == 2:
                data_smooth[:,:,s] = median_filter(data[:,:,s], kernel_size)
    else:
        # isotropic resolution
        kernel_size = [
            int(3.0 / physical_voxel_size[0]), 
            int(3.0 / physical_voxel_size[1]),
            int(3.0 / physical_voxel_size[2])
        ]
        if kernel_size[0]<3: kernel_size[0]=3
        if kernel_size[1]<3: kernel_size[1]=3
        if kernel_size[2]<3: kernel_size[2]=3

        data_smooth = median_filter(data, kernel_size)
    return data_smooth

class ComponentSelection(object):
    '''
    Extract a set of components using a selection mask.
    '''
    @staticmethod
    def _parallel_component_selection(params):
        '''
        Parallel worker function used by ComponentSelection class, 
        do not call this function directly.
        '''
        in_nifti, selection, select_method, out_nifti, skip_existing = params

        if try_load_nifti(out_nifti) == True and skip_existing:
            return

        in_dat, in_hdr = load_nifti(in_nifti,return_type='int32')
        if in_dat.shape != selection.shape:
            raise RuntimeError("in.shape (%s) != selection.shape (%s)." % (str(in_dat.shape),str(selection.shape)))

        n_parts, labeled = connected_components(in_dat)
        out_dat = np.zeros(in_dat.shape).astype('float32')

        if select_method == 'window':
            for i in range(1, n_parts+1):
                if np.sum(selection * (labeled == i)) == np.sum(labeled == i):
                    out_dat += (labeled==i).astype('float32')
        elif select_method == 'crossing':
            for i in range(1, n_parts+1):
                if np.sum(selection * (labeled == i)) > 0:
                    out_dat += (labeled==i).astype('float32')
        elif select_method == 'masking':
            out_dat = (in_dat * selection).astype('float32')

        out_dat = (out_dat > 0.5).astype('float32')
        save_nifti(out_dat, in_hdr, out_nifti)

    def __init__(self, selection: np.ndarray, select_method: str = 'crossing', skip_existing = True,
        in_niftis: list = [], out_niftis: list = []):
        '''
        Note
        =======
        Selection mode:
        * window: in this mode, components that completely fall into the selection will be selected.
        * crossing: in this mode, components that have any crossing section with the selection will
            be selected.
        * masking: in this mode, all components will be simply masked by selection.
        '''
        assert select_method in ['window', 'crossing', 'masking'], \
            'select_method can be "window", "crossing" or "masking", got "%s".' % str(select_method)

        if len(in_niftis) > 0 or len(out_niftis)>0:
            assert len(in_niftis) == len(out_niftis), 'length of in_niftis != out_niftis.'

        self.selection = (selection > 0.5).astype('int')
        self.select_method = select_method
        self.tasks = []
        self.skip_existing = skip_existing

        if len(in_niftis) > 0:
            self.run_task(in_niftis, out_niftis)

    def run_task(self, in_niftis, out_niftis, num_workers = 8):
        self.tasks = []
        for in_nifti, out_nifti in zip(in_niftis, out_niftis):
            self.tasks.append((in_nifti, self.selection, self.select_method ,out_nifti, self.skip_existing))
        run_parallel(ComponentSelection._parallel_component_selection, self.tasks, num_workers=num_workers, 
            progress_bar_msg='selection (%s)' % self.select_method)

def binarize_image(nii_file, threshold, save_file = None, as_type = 'float32'):
    '''
    binarize an image using a given threshold and return binarized data
    '''
    data, header = load_nifti(nii_file)
    data = (data > threshold).astype(as_type)
    if save_file:
        save_nifti(data, header, save_file)
    return data
