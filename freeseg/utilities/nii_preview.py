from freeseg.utilities.file_ops import gd, mkdir
from freeseg.utilities.colormaps import find_colormap_func, get_valid_color_mappings
from freeseg.utilities.data_io import get_nifti_pixdim, load_nifti, load_nifti_simple, try_load_nifti
import matplotlib
matplotlib.use('agg') # use matplotlib in command line mode
import warnings
import numpy as np
from matplotlib.pyplot import imsave
import nibabel as nib
from nibabel import processing as nibproc # used for resample images
from scipy.ndimage import zoom
import imageio

#################################
# NIFTI visualization utilities #
#################################

# hard coded character '0'~'9'.
# 8x6 character glyph
glyph_number = [
    np.transpose(np.array([[0,0,0,0,0,0],[0,0,1,1,0,0],[0,1,0,0,1,0],[0,1,0,0,1,0],[0,1,0,0,1,0],[0,1,0,0,1,0],[0,0,1,1,0,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,0,1,1,0,0],[0,1,0,1,0,0],[0,0,0,1,0,0],[0,0,0,1,0,0],[0,0,0,1,0,0],[0,1,1,1,1,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,0,1,1,0,0],[0,1,0,0,1,0],[0,0,0,0,1,0],[0,0,0,1,0,0],[0,0,1,0,0,0],[0,1,1,1,1,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,1,1,1,0,0],[0,0,0,0,1,0],[0,0,1,1,0,0],[0,0,0,0,1,0],[0,0,0,0,1,0],[0,1,1,1,0,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,0,0,1,0,0],[0,0,1,1,0,0],[0,1,0,1,0,0],[1,1,1,1,1,0],[0,0,0,1,0,0],[0,0,0,1,0,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,1,1,1,1,0],[0,1,0,0,0,0],[0,1,1,1,0,0],[0,0,0,0,1,0],[0,0,0,0,1,0],[0,1,1,1,0,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,0,1,1,1,0],[0,1,0,0,0,0],[0,1,1,1,0,0],[0,1,0,0,1,0],[0,1,0,0,1,0],[0,0,1,1,0,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,1,1,1,1,0],[0,1,0,0,1,0],[0,0,0,0,1,0],[0,0,0,1,0,0],[0,0,1,0,0,0],[0,1,0,0,0,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,0,1,1,0,0],[0,1,0,0,1,0],[0,0,1,1,0,0],[0,1,0,0,1,0],[0,1,0,0,1,0],[0,0,1,1,0,0],[0,0,0,0,0,0]])),
    np.transpose(np.array([[0,0,0,0,0,0],[0,0,1,1,0,0],[0,1,0,0,1,0],[0,1,0,0,1,0],[0,0,1,1,1,0],[0,0,0,0,1,0],[0,1,1,1,0,0],[0,0,0,0,0,0]]))
]

# original answer from:
# https://stackoverflow.com/questions/45027220/expanding-zooming-in-a-numpy-array
# ratio must be an integer
def _int_zoom(array, ratio):
    return np.kron(array, np.ones((ratio,ratio)))

# _rect_intersect():
# utility function to calculate the intersection of the two rectangles (x,y,w,h)
# original source is from:
# https://answers.opencv.org/question/90455/how-to-perform-intersection-or-union-operations-on-a-rect-in-python/
def _rect_intersect(a, b):
    '''
    a: (x1,y1,w1,h1)
    b: (x2,y2,w2,h2)
    '''
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return (x,y,0,0) # or (0,0,0,0) ?
    return (x, y, w, h)

# paste_slice(): copy and paste an image slice to another image
# here (x,y) indicates the upper-left coordinates of the destination position 
# this function is designed to be robust and it can handle out of bound values
# carefully.
def paste_slice(src, dst, x, y):
    '''
    Description
    ------------
    Copy and paste an image slice to another image. (x,y) indicates the 
    upper-left coordinates of the destination position. This function is 
    designed to be robust and it can handle out of bound values carefully.

    Parameters
    ------------
    src: np.ndarray
        source image slice with shape MxN (grayscale) or MxNx3 (RGB)
    dst: np.ndarray
        target image with shape MxNx3
    x,y: int
        upper-left coordinates of the destination position
    '''
    shp = src.shape
    if x>=0 and y>=0 and x+shp[0]<=dst.shape[0] and y+shp[1]<=dst.shape[1]:
        if len(src.shape)==2:
            for ch in range(3): # channel broadcast
                dst[x:x+shp[0],y:y+shp[1],ch]=src
        else:
            dst[x:x+shp[0],y:y+shp[1]]=src
        return dst
    else:
        # out of bounds
        src_rect = (x,y,shp[0],shp[1])
        dst_rect = (0,0,dst.shape[0],dst.shape[1])
        ist_rect = _rect_intersect(src_rect, dst_rect)
        if ist_rect[2] == 0 or ist_rect[3] == 0:
            return dst # no intersection
        src_offset = ( ist_rect[0]-x, ist_rect[1]-y )
        if len(src.shape)==2:
            for ch in range(3): # channel broadcast
                dst[ist_rect[0]:ist_rect[0]+ist_rect[2], ist_rect[1]:ist_rect[1]+ist_rect[3], ch ]=\
                    src[src_offset[0]:src_offset[0]+ist_rect[2], src_offset[1]:src_offset[1]+ist_rect[3]]
        else:
            dst[ist_rect[0]:ist_rect[0]+ist_rect[2], ist_rect[1]:ist_rect[1]+ist_rect[3]]=\
                src[src_offset[0]:src_offset[0]+ist_rect[2], src_offset[1]:src_offset[1]+ist_rect[3]]
        return dst

def lightbox(
        # 1. basic options:
        nii_file:str, n_rows:int, n_cols:int, save_path:str, 
        slice_range=None, slice_step=None, view_axis='axial', 
        # 2. color overlay options:
        # use these options if you want to draw color overlays (ie, draw lesion overlay)
        # you can assign each value in mask to have a different color (except black)
        nii_mask=None, color_palette=None, blend_weight=0.5,
        # 3. resample options:
        # use this option if your image resolution is not isotropic, ie: resample=1.0 to 
        # resample image to have 1mm isotropic resolution. resample_order can be a integer
        # from 0~5. resample_order=0 means nearest interpolation, 1 means linear interpolation.
        # maximum order is up to 5.
        resample=None, resample_order=1, # resample = 1.0 or resample = [1.0,1.0,1.0]
        # 4. intensity normalization options:
        # default normalization strategy is to scale the whole image (image.min(), image.max()) between 0~1.
        # howewer you can specify a custom range.
        intensity_range = None, # e,g: intensity_range = [0.0, 1000.0]
        # 5. miscellaneous options below
        show_slice_number=True, font_size=1):
    '''
    Description
    -------------
    Visualizing NIFTI image.
    '''
    
    nii_data = load_nifti_simple(nii_file)

    assert view_axis in ['sagittal', 'coronal', 'axial'], '"view_axis" should be "sagittal", "coronal" or "axial"(default).'
    assert len(nii_data.shape) == 3, 'Only support 3D NIFTI data.'
    assert slice_range is None or isinstance(slice_range, (tuple, list)), 'Invalid "slice_range" setting.'
    assert len(save_path)>4 and save_path[-4:]=='.png', '"save_path" must ends with ".png".'
    assert isinstance(font_size,int), '"font_size" must be "int".'

    if nii_mask is not None:
        nii_mask_data = np.around(load_nifti_simple(nii_mask)).astype('int')
        assert nii_mask_data.shape == nii_data.shape, 'data shape is not equal to mask shape.'
        assert color_palette is not None, 'must assign color palette when mask is provided.'
        assert nii_mask_data.max() == len(color_palette), 'Invalid color palette.'
    
    # nii_data: intensity image
    # nii_mask_data: color overlay, color is assigned by user defined color palette

    # resample image if needed.
    # this happens when you want to show a NIFTI image with anisotropic resolution
    # if you don't resample the image to a isotropic resolution then the image will 
    # appears to be stretched
    if resample is not None:
        if slice_range is not None or slice_step is not None:
            warnings.warn('Image will be resampled. "slice_range" and "slice_step" settings '
                          'are no longer accurate.')
        resample_3d = [resample, resample, resample] if isinstance(resample, float) else resample
        # reload nii data from resampled image
        resampled_data = nibproc.resample_to_output(nib.load(nii_file), resample_3d, order=resample_order) # linear interpolation
        nii_data = resampled_data.get_fdata().astype('float32') 
        if nii_mask is not None:
            # reload nii mask from resampled image
            resampled_mask = nibproc.resample_to_output(nib.load(nii_mask), resample_3d, order=0) # force using nearest interpolation
            nii_mask_data = np.around(resampled_mask.get_fdata()).astype('int') # to avoid floating point rounding error, here we used np.around()

    # OK, now all preparation works have been done

    if view_axis == 'sagittal':
        nii_slices = nii_data.shape[0]
        nii_data_tr = np.transpose(nii_data, [0,1,2])[::-1,::-1,::-1]
        if nii_mask is not None:
            nii_mask_data_tr = np.transpose(nii_mask_data, [0,1,2])[::-1,::-1,::-1]
    elif view_axis == 'coronal':
        nii_slices = nii_data.shape[1]
        nii_data_tr = np.transpose(nii_data, [1,0,2])[::-1,::-1,::-1]
        if nii_mask is not None:
            nii_mask_data_tr = np.transpose(nii_mask_data, [1,0,2])[::-1,::-1,::-1]
    elif view_axis == 'axial':
        nii_slices = nii_data.shape[2]
        nii_data_tr = np.transpose(nii_data, [2,0,1])
        if nii_mask is not None:
            nii_mask_data_tr = np.transpose(nii_mask_data, [2,0,1])

    # calculate slice start and slice end
    if slice_range is None:
        slice_start, slice_end = 0, nii_slices-1
    else:
        slice_start, slice_end = slice_range[0], slice_range[1]
    if slice_end is None or slice_end<0 or slice_end>=nii_slices:
        slice_end = nii_slices-1
    if slice_start is None or slice_start<0 or slice_start>=nii_slices:
        slice_end = 0
    if slice_start > slice_end:
        slice_start = slice_end
    total_slices = slice_end - slice_start + 1
    view_slices = n_rows * n_cols
    
    # calculate slice step
    if view_slices >= total_slices:
        slice_step = 1
    else:
        if slice_step is None:
            slice_step = float(total_slices) / float(view_slices)
    
    # calculate slice shape and image size
    slice_shape = (nii_data_tr.shape[1], nii_data_tr.shape[2])
    image_height = n_rows * slice_shape[1]
    image_width = n_cols * slice_shape[0]

    image = np.zeros([image_width, image_height, 3]) # RGB channel

    # normalize nii intensity
    if intensity_range is not None:
        lo, hi = intensity_range
        nii_data_tr = np.where(nii_data_tr > hi, hi, nii_data_tr)
        nii_data_tr = np.where(nii_data_tr < lo, lo, nii_data_tr)
    nii_data_tr = (nii_data_tr - nii_data_tr.min()) / (nii_data_tr.max()-nii_data_tr.min() + 0.00001)

    current_slice = slice_start
    for iy in range(n_rows):
        for ix in range(n_cols):
            if current_slice < nii_data_tr.shape[0]:
                # paste slice data
                slice_data = nii_data_tr[current_slice]
                if nii_mask is not None:
                    ind_data = nii_mask_data_tr[current_slice]
                    color_data = np.zeros([slice_data.shape[0], slice_data.shape[1], 3])
                    for ic in range(len(color_palette)):
                        color_data[:,:,0] = np.where(ind_data==ic+1, slice_data * (1-blend_weight) + color_palette[ic][0]/255.0 * blend_weight , slice_data) # fill R
                        color_data[:,:,1] = np.where(ind_data==ic+1, slice_data * (1-blend_weight) + color_palette[ic][1]/255.0 * blend_weight , slice_data) # fill G
                        color_data[:,:,2] = np.where(ind_data==ic+1, slice_data * (1-blend_weight) + color_palette[ic][2]/255.0 * blend_weight , slice_data) # fill B 
                    paste_slice(color_data, image, slice_shape[0]*ix, slice_shape[1]*iy)
                else:
                    paste_slice(nii_data_tr[current_slice], image, slice_shape[0]*ix, slice_shape[1]*iy)
                if show_slice_number:
                    # paste slice number
                    slice_str = '%03d' % current_slice
                    numbering_pos = (slice_shape[0]*ix+2, slice_shape[1]*iy+2)
                    for ig in range(3): # max 3 digits
                        selected_glyph = glyph_number[int(slice_str[ig])]
                        selected_glyph = _int_zoom(selected_glyph, font_size)
                        paste_slice(selected_glyph, image, numbering_pos[0]+ig*6*font_size, numbering_pos[1])
            current_slice = int(current_slice + slice_step)

    imsave(save_path, np.transpose(image,[1,0,2])) # don't forget to transpose the matrix before saving!

def nii_save_slice_as_image(
    slice_data: np.ndarray,
    output_path: str,
    colormap = 'grayscale',
    min_intensity = None,
    max_intensity = None,
    slice_number = None,
    font_zoom = None
):
    assert len(slice_data.shape) == 2, 'Slice data dimension incorrect (expected 2).'
    assert colormap in get_valid_color_mappings(), 'Unknown colormap setting "%s".' % colormap
    if slice_number is not None:
        assert isinstance(slice_number, int), 'slice number must be "int" or None.'

    # normalize data to zero and one
    norm_data = (slice_data - min_intensity) / (max_intensity - min_intensity)
    norm_data = np.where(norm_data<0,0,norm_data)
    norm_data = np.where(norm_data>1,1,norm_data)

    h, w = slice_data.shape
    rgb_data = np.zeros([h,w,3])
    colormap_func = find_colormap_func( colormap )
    for y in range(h):
        for x in range(w):
            rgb_data[y,x,:] = colormap_func(0,1,norm_data[y,x])

    if slice_number is not None:
        # paste slice number
        slice_str = '%d' % slice_number
        numbering_pos = (0,0)
        if font_zoom == None:
            font_zoom = 1
        for ig in range(len(slice_str)):
            selected_glyph = glyph_number[int(slice_str[ig])]
            selected_glyph = _int_zoom(selected_glyph, font_zoom)
            paste_slice(selected_glyph, rgb_data, numbering_pos[0]+ig*6*font_zoom, numbering_pos[1])
    
    imsave( output_path, np.transpose(rgb_data, [1,0,2]), vmin=min_intensity, vmax=max_intensity )

def nii_view_slice(nii_file, output_image, axis='axial', 
    slice_num=None,  reverse_slice_order = False, show_slice_number = False, hflip = False, vflip = False,
    intensity_range = None, colormap = 'grayscale',
    crop = None, anisotropic_resize = True, global_zoom = 1):
    '''
    Description
    -----------
    Save single slice from a NIFTI file to common image format.

    Parameters
    -----------
    nii_file: str
        NIFTI file path.
    output_image: str
        output image path, file name must end with common image extension names such as "*.png", "*.bmp", "*.jpg" ...
    axis: str
        view axis, can be one of "sagittal", "coronal", "axial"
    slice_num: int
        slice number
    reverse_slice_order: bool
        if slice order needs reverse then set it to True
    hflip: bool
        flip slice horizontally when saving
    vflip: bool
        flip slice vertically when saving
    crop: [x1, y1, x2, y2]
        crop slice before resizing
    intensity_range: [min, max]
        assign intensity range when visualizing a image
    '''

    assert axis in ['sagittal', 'coronal', 'axial'], 'Incorrect axis setting "%s".' % axis
    assert isinstance(global_zoom, int), 'Global zoom must be integer.'
    assert isinstance(slice_num, int), 'must specify slice_num.'

    dat = load_nifti_simple(nii_file)
    res = get_nifti_pixdim(nii_file)

    s = slice_num
    if axis == 'sagittal':
        if reverse_slice_order:
            total_slices = dat.shape[0]
            s = total_slices - slice_num - 1
        slice_data = dat[s,:,:]
        slice_res = [res[1], res[2]]
    elif axis == 'coronal':
        if reverse_slice_order:
            total_slices = dat.shape[1]
            s = total_slices - slice_num - 1
        slice_data = dat[:,s,:]
        slice_res = [res[0], res[2]]
    elif axis == 'axial':
        if reverse_slice_order:
            total_slices = dat.shape[2]
            s = total_slices - slice_num - 1
        slice_data = dat[:,:,s]
        slice_res = [res[0], res[1]]
    
    if hflip:
        slice_data = slice_data[:,::-1]
    if vflip:
        slice_data = slice_data[::-1,:]
    if crop:
        slice_data = slice_data[crop[0]:crop[2], crop[1]: crop[3]]

    if anisotropic_resize:
        aspect_ratio = slice_res[0] / slice_res[1]
        slice_data = zoom(slice_data, [ aspect_ratio, 1.0 ], order = 3)

    slice_data = zoom(slice_data, global_zoom, order=0)

    if intensity_range is None:
        min_intensity, max_intensity = np.min(dat), np.max(dat)
    else:
        if intensity_range[0] is not None and intensity_range[1] is not None: # [float/int, float/int]
            min_intensity = intensity_range[0]
            max_intensity = intensity_range[1]
        elif intensity_range[0] is not None: # [None, float/int]
            min_intensity = intensity_range[0]
            max_intensity = np.max(dat)
        elif intensity_range[1] is not None: # [float/int, None]
            max_intensity = intensity_range[1]
            min_intensity = np.min(dat)
        else: # [None, None]
            min_intensity, max_intensity = np.min(dat), np.max(dat)

    nii_save_slice_as_image( slice_data, output_image, colormap=colormap,
        min_intensity=min_intensity, max_intensity=max_intensity,
        slice_number=None if show_slice_number == False else slice_num,
        font_zoom=global_zoom)

def nii_draw_colorbar(output_image, colormap='grayscale', size=[256,48]):
    assert colormap in get_valid_color_mappings(), 'Unknown colormap setting "%s".' % colormap
    bardata = np.zeros([*size,3])
    colormap_func = find_colormap_func(colormap)
    for i in range(size[0]):
        for j in range(size[1]):
            bardata[i,j,:] = colormap_func(0,1,i/size[0])

    imsave( output_image, np.transpose(bardata, [1,0,2]), vmin=0.0, vmax=1.0)

def nii_as_gif(filepath, outpath, axis = 'axial', resample_image = True, 
    slice_range = None, slice_step = 1, dt = 0.2, intensity_range = None,
    colormap = 'grayscale', draw_slice_number = True, lesion_mask=None, side_by_side = False):
    '''
    Description
    ------------
    Saving NIFTI file (*.nii) to a GIF animation.

    Examples
    ------------
    >>> nii_output_gif('.../input.nii.gz', '.../output.gif')
    >>> nii_output_gif('.../input.nii.gz', '.../output.gif', intensity_range = [20, 120], slice_range = [50,80], slice_step = 2, dt = 0.5, colormap = "grayscale")

    Parameters
    ------------
    filepath: str
        NIFTI file path, can be *.nii or *.nii.gz format.
    outpath: str
        Output GIF file path, must end with ".gif".
    axis: str (default = "axial")
        View axis. Must be one of the followings: "sagittal", "coronal" or "axial".
    resample_image: bool (default = True)
        Resample image to isotropic resolution. 
    slice_range: list | None (default = None)
        Specify slice range [start, end). Leave it to None will display all slices.
    slice_step: int (default = 1)
        Slice step.
    dt: float (default = 0.2)
        Duration of each frame (sec).
    intensity_range: list | None (default = None)
        Clamp NIFTI image intensities to this range before generating GIF.
    colormap: str (default = "grayscale")
        Colormap used when mapping intensity value to RGB color. Can be one of the 
        followings: "grayscale", "grayscale2", "rainbow" or "metalheat".
    draw_slice_number: bool (default = True)
        Display slice number on the upper left corner of each slice when generating
        GIF preview.
    lesion_mask: str | None (default = None)
        Specify lesion mask file (NIFTI format) to overlay onto the original image 
        slice.
    side_by_side: bool (default = False) 
        Draw image and lesion overlay side by side. Can be enabled only when the lesion 
        mask is given. When lesion mask is not given, this parameter will be ignored.

    '''
    assert axis in ['sagittal', 'coronal', 'axial'], 'invalid axis setting.'
    if lesion_mask is not None:
        assert try_load_nifti(lesion_mask), 'cannot load lesion mask "%s", file has no access or not exist.' % lesion_mask
    if isinstance(slice_range, list):
        assert len(slice_range) == 2 and isinstance(slice_range[0], int) and isinstance(slice_range[1], int), \
            'invalid slice_range setting. slice_range should be None or a list with two integers indicates the ' \
            'start/end slice.'
    if isinstance(intensity_range, list):
        assert len(intensity_range) == 2 and isinstance(intensity_range[0], int) and isinstance(intensity_range[1], int), \
            'invalid intensity_range setting. intensity_range should be None or a list with two integers indicates the ' \
            'min/max intensities.'
    assert colormap in get_valid_color_mappings(), 'invalid colormap.'

    colormap_func = find_colormap_func(colormap)

    data, _ = load_nifti(filepath, nan=0.0)

    if lesion_mask is not None:
        lesion, _ = load_nifti(lesion_mask)
        lesion = (lesion>0.5).astype('float32')
        assert lesion.shape == data.shape , 'shapes of lesion mask and image are not equal.'
    else:
        lesion = np.zeros(data.shape) # generate empty lesion mask

    # normalize data intensity
    if intensity_range == None:
        data = (data-data.min()) / (data.max()-data.min())
    else:
        data = (data-intensity_range[0]) / (intensity_range[1]-intensity_range[0])
        data = np.where(data<0,0,data)
        data = np.where(data>1,1,data)
    

    # calculate start and end slice
    if slice_range is not None:
        slice_start, slice_end = slice_range
    else:
        if axis == 'sagittal':
            slice_start, slice_end = 0, data.shape[0]
        elif axis == 'coronal':
            slice_start, slice_end = 0, data.shape[1]
        elif axis == 'axial':
            slice_start, slice_end = 0, data.shape[2]

    # reorient lesion and image
    voxlen = get_nifti_pixdim(filepath)
    if axis == 'sagittal':
        pixlen = np.array([voxlen[1],voxlen[2]])
        data = data[::-1,::-1,::-1]
        lesion = lesion[::-1,::-1,::-1]
    elif axis == 'coronal':
        pixlen = np.array([voxlen[0],voxlen[2]])
        data = data[::-1,::-1,::-1]
        lesion = lesion[::-1,::-1,::-1]
    elif axis == 'axial':
        pixlen = np.array([voxlen[0],voxlen[1]])

    zoomfact = list(pixlen)

    with imageio.get_writer(outpath, mode='I', duration=dt) as writer:

        for sid in range(slice_start, slice_end, slice_step):

            # get slice data
            if axis == 'sagittal':
                data_slice = data[sid,:,:]
                lesion_slice = lesion[sid,:,:]
            elif axis == 'coronal':
                data_slice = data[:,sid,:]
                lesion_slice = lesion[:,sid,:]
            elif axis == 'axial':
                data_slice = data[:,:,sid]
                lesion_slice = lesion[:,:,sid]
            if resample_image:
                data_slice = zoom(data_slice, zoomfact, order = 3)
                lesion_slice = zoom(lesion_slice, zoomfact, order = 3)
                lesion_slice = (lesion_slice>0.5).astype('float32')
            
            if lesion_mask == None or side_by_side == False:
                # map colors
                color_slice = np.zeros([*data_slice.shape, 3])
                for x in range(color_slice.shape[0]):
                    for y in range(color_slice.shape[1]):
                        color_slice[x,y,:] = colormap_func(0,1,data_slice[x,y])
                        if lesion_slice[x,y] > 0.5:
                            color_slice[x,y,:] = [1,0,0] # pure red
                            
                # draw slice number
                if draw_slice_number:
                    slice_str = '%d' % sid
                    numbering_pos = (0,0)
                    font_zoom = 1
                    for ig in range(len(slice_str)):
                        selected_glyph = glyph_number[int(slice_str[ig])]
                        selected_glyph = _int_zoom(selected_glyph, font_zoom)
                        paste_slice(selected_glyph, color_slice, numbering_pos[0]+ig*6*font_zoom, numbering_pos[1])
                            
                writer.append_data(( np.transpose(color_slice, [1,0,2]) * 255).astype('uint8'))

            else:

                # map colors
                X, Y = data_slice.shape[0], data_slice.shape[1]
                color_slice = np.zeros([2*X, Y, 3])
                for x in range(X):
                    for y in range(Y):
                        val = colormap_func(0,1,data_slice[x,y])
                        color_slice[x,y,:] = val
                        color_slice[x+X,y,:] = val
                for x in range(X):
                    for y in range(Y):
                        if lesion_slice[x,y] > 0.5:
                            color_slice[x+X,y,:] = [1,0,0] # pure red
                            
                # draw slice number
                if draw_slice_number:
                    slice_str = '%d' % sid
                    numbering_pos = (0,0)
                    font_zoom = 1
                    for ig in range(len(slice_str)):
                        selected_glyph = glyph_number[int(slice_str[ig])]
                        selected_glyph = _int_zoom(selected_glyph, font_zoom)
                        paste_slice(selected_glyph, color_slice, numbering_pos[0]+ig*6*font_zoom, numbering_pos[1])
                        paste_slice(selected_glyph, color_slice, numbering_pos[0]+ig*6*font_zoom+X, numbering_pos[1])

                writer.append_data(( np.transpose(color_slice, [1,0,2]) * 255).astype('uint8'))


def nii_slice_range(filepath, axis = "axial", value = 0.001, percentage = 0.999):
    '''
    Description
    ------------
    calculate slice_start and slice_end, ignore empty slices.

    slice_range = [ slice_start, slice_end )
    
    * A slice is considered empty if the ratio between the number of voxels with 
      intensity below :param "value" and the total number of voxels in this slice 
      is above :param "percentage".

    '''
    assert axis in ['sagittal', 'coronal', 'axial'], 'invalid axis setting.'

    data, _ = load_nifti(filepath)
    iStart = 0
    if axis == 'sagittal': iEnd = data.shape[0]
    elif axis == 'coronal': iEnd = data.shape[1]
    elif axis == 'axial': iEnd = data.shape[2]

    slice_start, slice_end = -1,-1

    for iSlice in range(iStart, iEnd):
        if axis == 'sagittal': data_slice = data[iSlice,:,:]
        elif axis == 'coronal': data_slice = data[:,iSlice,:]
        elif axis == 'axial': data_slice = data[:,:,iSlice]
        if np.sum(data_slice < value) / data_slice.size < percentage:
            slice_start = iSlice
            break

    for iSlice in range(iEnd-1, iStart, -1):
        if axis == 'sagittal': data_slice = data[iSlice,:,:]
        elif axis == 'coronal': data_slice = data[:,iSlice,:]
        elif axis == 'axial': data_slice = data[:,:,iSlice]
        if np.sum(data_slice < value) / data_slice.size < percentage:
            slice_end = iSlice
            break
    
    slice_end += 1
    
    if slice_start < 0: 
        # the whole image is empty
        slice_start = 0
    
    return slice_start, slice_end


class SimpleNiftiPreview:
    '''
    Visualizing a single Nifti slice using custom options.
    '''
    def __init__(self,min_intensity = 'auto', max_intensity = 'auto',colormap = 'grayscale'):
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.colormap = colormap
        if self.colormap not in get_valid_color_mappings():
            raise RuntimeError("invalid colormap: '%s', colormaps can be one of the following: " \
                % (self.colormap, ' '.join(get_valid_color_mappings())))
        if self.min_intensity != 'auto':
            assert isinstance(self.min_intensity, (float,int)), 'must assign a float/int to "min_intensity".'
        if self.max_intensity != 'auto':
            assert isinstance(self.max_intensity, (float,int)), 'must assign a float/int to "max_intensity".'

    def plot(self, nifti_file, axis, slice_num, output_image, output_colormap=None,
        vflip = False, hflip = False):

        assert axis in ['sagittal', 'coronal', 'axial'], 'invalid direction setting.'
        mkdir(gd(output_image))
    
        # draw color bar if needed
        if output_colormap is not None:
            mkdir(gd(output_colormap))
            nii_draw_colorbar(output_colormap, colormap = self.colormap)

        # visualize nifti slice
        min_intensity = None if self.min_intensity == 'auto' else self.min_intensity
        max_intensity = None if self.max_intensity == 'auto' else self.max_intensity

        nii_view_slice(nifti_file, output_image, 
            axis = axis, slice_num=slice_num, intensity_range=[min_intensity, max_intensity],
            colormap=self.colormap,vflip=vflip, hflip=hflip)
