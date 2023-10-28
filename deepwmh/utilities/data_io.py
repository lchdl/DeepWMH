# defines some utility functions for reading/writing 
# nifti(*.nii)/pickle(*.pkl)/csv(*.csv) files

import os, csv, gzip, tarfile, pickle, json, xlsxwriter, shutil, imageio, openpyxl
import nibabel as nib
import numpy as np
from typing import Union
from xlsxwriter.format import Format
from copy import deepcopy
from scipy.io import loadmat
from nibabel import processing as nibproc
from deepwmh.utilities.file_ops import abs_path, dir_exist, file_empty, file_exist, join_path, gd, gn, mkdir

# utility function used to compress a file into "*.gz"
# (not suitable for compressing folders)
def gz_compress(file_path, out_path=None, compress_level:int=9, verbose=False, overwrite = True):
    assert compress_level>=0 and compress_level<=9, 'invalid compress level (0~9 accepted, default is 9).'
    assert file_exist(file_path), 'file "%s" not exist or it is a directory. '\
        'gzip can be only used to compress files, if you want to compress folder structure, '\
        'please use targz_compress(...) instead.' % file_path
    f = open(file_path,"rb")
    data = f.read()
    bindata = bytearray(data)
    gz_path = join_path( gd(file_path) , gn(file_path) + '.gz' ) if out_path is None else out_path
    if file_exist(gz_path) and not overwrite:
        if verbose:
            print('skip %s' % (file_path))
    else:
        if verbose:
            print('%s >>> %s' % (file_path, gz_path))
        with gzip.GzipFile(filename=gz_path, mode='wb', compresslevel=compress_level) as f:
            f.write(bindata)
    return gz_path

def gz_uncompress(gz_path, out_path=None, verbose=False):
    out_path0 = ''
    if out_path is not None:
        out_path0 = out_path
    else:
        if gz_path[-3:] == '.gz':
            out_path0 = gz_path[:-3]
        else:
            raise RuntimeError(
                'Incorrect gz file name. Input file name must '
                'end with "*.gz" if out_path is not set.')
    out_path0 = abs_path(out_path0)

    with gzip.open(gz_path, 'rb') as f_in:
        with open(out_path0, 'wb') as f_out:
            if verbose: print('%s >>> %s' % (gz_path, out_path))
            shutil.copyfileobj(f_in, f_out)

# compress a file or a folder structure into *.tar.gz format.
def targz_compress(file_or_dir_path, out_file=None, compress_level:int=9, verbose=False):
    assert compress_level>=0 and compress_level<=9, 'invalid compress level (0~9 accepted, default is 9).'
    assert file_exist(file_or_dir_path) or dir_exist(file_or_dir_path), \
        'file or directory not exist: "%s".' % file_or_dir_path
    targz_path = join_path(gd(file_or_dir_path) , gn(file_or_dir_path) + '.tar.gz') if out_file is None else out_file
    if file_exist(file_or_dir_path):
        # target path is a file
        with tarfile.open( targz_path , "w:gz" , compresslevel=compress_level) as tar:
            tar.add(file_or_dir_path, arcname=gn(file_or_dir_path) )
            if verbose:
                print('>>> %s' % file_or_dir_path)
    elif dir_exist(file_or_dir_path):
        # target path is a folder
        with tarfile.open( targz_path , "w:gz" , compresslevel=compress_level) as tar:
            for name in os.listdir( file_or_dir_path ):
                tar.add( join_path(file_or_dir_path, name) , recursive=True, arcname=name)
                if verbose:
                    print('>>> %s' % name)
    else:
        raise RuntimeError('only file or folder can be compressed.')

def targz_uncompress(targz_file, out_path):
    '''
    out_path: str
        path to output folder.
    '''
    targz = tarfile.open(targz_file)
    targz.extractall(out_path)
    targz.close()


def load_csv_simple(file_path, key_names = None):
    '''
    load a csv file as python dict.
    '''
    parsed_dataset = {}

    if key_names is None:
        with open(file_path, 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file,delimiter=',',quotechar='"')
            for row in csv_reader:
                key_names = row
                break
        key_names = [key for key in key_names if len(key) > 0] # remove ''
    
    with open(file_path, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',',quotechar='"')
        table_head = None
        for row in csv_reader:
            if table_head is None: 
                table_head = row
            else:
                for key in key_names:
                    if key not in table_head:
                        raise Exception('Cannot find key name "%s" in CSV file "%s". Expected key names can be %s.' % \
                            (key,file_path, table_head))
                    else:
                        column_index = table_head.index(key)
                        if key not in parsed_dataset:
                            parsed_dataset[key] = [] # create list
                        parsed_dataset[key].append(row[column_index])

    return parsed_dataset

def write_csv_simple(file_path, csv_dict):
    keys = list(csv_dict.keys())
    lines=0
    for key in keys: # calculate max lines
        if len(csv_dict[key]) > lines:
            lines = len(csv_dict[key])
    mkdir(gd(file_path))
    with open(file_path,'w') as f:
        table_head = ''
        for key in keys: table_head += key+','
        f.write(table_head+'\n')
        for i in range(lines):
            for key in keys:
                if i >= len(csv_dict[key]):
                    f.write(',')
                else:
                    f.write('%s,' % csv_dict[key][i])
            f.write('\n')

def save_pkl(obj, pkl_path):
    with open(pkl_path,'wb') as f:
        pickle.dump(obj,f)

def load_pkl(pkl_path):
    content = None
    with open(pkl_path,'rb') as f:
        content = pickle.load(f)
    return content

def save_json(obj, json_path, indent=4):
    with open(json_path,'w') as f:
        json.dump(obj, f,indent=indent)

def load_json(json_path):
    with open(json_path, 'r') as f:
        obj = json.load(f)
    return obj

def load_pyval(py_path) -> dict:
    '''
    read a python file and parse all its variables as a dictionary
    '''
    d = {}
    with open(py_path,'r') as f:
        exec(f.read(), d)
        d.pop('__builtins__')
    return d

def try_load_gif(file_path):
    '''
    test if a GIF file can be successfully loaded.
    '''
    if file_exist(file_path) == False:
        return False
    if file_exist(file_path) and file_empty(file_path):
        return False
    try:
        success = True
        imageio.get_reader(file_path)
    except Exception:
        success = False
    except BaseException:
        raise
    return success

def load_mat(file_path):
    '''
    load a MATLAB *.mat file.
    '''
    mat = loadmat(file_path)
    return mat

def try_load_mat(file_path):
    '''
    test if a MATLAB matrix save file (*.mat) can be successfully loaded.
    '''
    success = True
    try:
        loadmat(file_path)
    except Exception:
        success = False
    except BaseException:
        raise
    return success


#
# NIFTI file operations
#

def _nifti_RAS_fix(image_data, image_affine):
    """
    Check the NIfTI orientation, and flip to 'RAS+' if needed.
    return: array after flipping
    """
    image_data0 = deepcopy(image_data)
    x, y, z = nib.aff2axcodes(image_affine)
    if x != 'R':
        image_data0 = nib.orientations.flip_axis(image_data0, axis=0)
    if y != 'A':
        image_data0 = nib.orientations.flip_axis(image_data0, axis=1)
    if z != 'S':
        image_data0 = nib.orientations.flip_axis(image_data0, axis=2)
    return image_data0

def load_nifti(path, return_type='float32', force_RAS = False, 
    nan = None, posinf = None, neginf = None):
    '''
    Description
    -----------
    returns the loaded nifti data and header.

    Notes
    -----------
    force_RAS: if you want the loaded data is in RAS+ orientation
        (left to Right, posterior to Anterior, inferior to Superior),
        set it to True (default is False).

    nan, posinf, neginf: convert NaNs, +inf, -inf to floating point numbers. 
        If you do not want to convert them, leave them to None (which is the 
        default setting)

    usage
    -----------
    >>> data, header = load_nifti("example.nii.gz")
    >>> data, header = load_nifti("example.nii.gz", return_type='int32')
    '''
    nifti = nib.load(path)
    header = nifti.header.copy()
    data = nifti.get_fdata()
    
    if nan is not None:
        assert isinstance(nan, float), 'param "nan" should be a floating point number.'
        data = np.nan_to_num(data, nan=nan)
    if posinf is not None:
        assert isinstance(posinf, float), 'param "posinf" should be a floating point number.'
        data[data == np.inf] = posinf
    if neginf is not None:
        assert isinstance(neginf, float), 'param "neginf" should be a floating point number.'
        data[data == np.inf] = neginf

    if force_RAS:
        data = _nifti_RAS_fix( data, nifti.affine )
    if return_type is not None:
        data = data.astype(return_type)
    return data, header

def try_load_nifti(path):
    '''
    Sometimes we need to check if the NIFTI file is already exists, but only checking 
    the existense of the file is not enough, we need to guarantee the file is not 
    corrupted and can be successfully read.
    '''
    if file_exist(path) == False:
        return False
    if file_exist(path) and file_empty(path):
        return False
    try:
        success = True
        load_nifti(path)
    except Exception:
        success = False
    except BaseException: 
        # other types of system errors are triggered and we dont know how to handle it
        raise
    return success

def save_nifti(data, header, path):
    nib.save(nib.nifti1.Nifti1Image(data.astype('float32'), None, header=header),path)

def load_nifti_simple(path, return_type='float32'):
    data, _ = load_nifti(path, return_type=return_type)
    return data # only retreive data from file, ignore its header info

def save_nifti_simple(data,path): 
    # save NIFTI using the default header (clear position offset, using identity matrix 
    # and 1x1x1 mm^3 isotropic resolution)
	nib.save(nib.Nifti1Image(data.astype('float32'),affine=np.eye(4)),path)

def get_nifti_header(path):
    _, header = load_nifti(path, return_type=None)
    return header

def get_nifti_data(path, return_type='float32'):
    return load_nifti_simple(path, return_type=return_type)

# synchronize NIFTI file header
def sync_nifti_header(source_path, target_path, output_path):
    target_header = nib.load(target_path).header.copy()
    source_data = nib.load(source_path).get_fdata()
    save_nifti(source_data, target_header, output_path)

# get physical resolution
def get_nifti_pixdim(nii_path):
    nii = nib.load(nii_path)
    nii_dim = list(nii.header['dim'])
    nii_pixdim = list(nii.header['pixdim'])
    actual_dim = list(nii.get_fdata().shape)
    physical_resolution = []
    for v in actual_dim:
        physical_resolution.append(nii_pixdim[nii_dim.index(v)])
    return physical_resolution

def resample_nifti(source_path, new_resolution, output_path):
    '''
    Description
    --------------
    resample NIFTI file to another physical resolution.

    Parameters
    --------------
    source_path: str
        source NIFTI image path. Can be "*.nii" or "*.nii.gz" format
    new_resolution: list
        new physical resolution. Units are "mm". For example, if you
        want to resample image to 1mm isotropic resolution, use [1,1,1].
    output_path: str
        output NIFTI image path with resampled resolution. Can be "*.nii"
        or "*.nii.gz" format.
    '''
    input_img = nib.load(source_path)
    resampled_img = nibproc.resample_to_output(input_img, new_resolution, order=0)
    nib.save(resampled_img, output_path)

def nifti_main_axis(pixdim:list) -> str:
    assert len(pixdim) == 3, 'error, cannot determine main axis for non three dimension data.'
    axis = np.argmax(pixdim)
    if axis == 0: return 'sagittal'
    elif axis == 1: return 'coronal'
    else: return 'axial'

#
# Excel file operations
#

class SimpleExcelWriter(object):
    def __init__(self, file_path, worksheet_names='default'):

        if file_path[-5:]!='.xlsx':
            raise RuntimeError('Invalid file name. File path must ends with ".xlsx".')
        self.file_path = file_path

        if isinstance(worksheet_names, str):
            self.worksheet_names = [worksheet_names]
        elif isinstance(worksheet_names, list):
            self.worksheet_names = worksheet_names
        else:
            raise RuntimeError('Only str or list type are accepted. Got type "%s".' % type(worksheet_names).__name__)

        # create workbook and worksheet(s)
        self.workbook = xlsxwriter.Workbook(self.file_path)
        self.worksheets = {}
        for worksheet_name in self.worksheet_names:
            self.worksheets[worksheet_name] = self.workbook.add_worksheet(worksheet_name)

    def _is_closed(self):
        return self.workbook is None
    
    def _check_closed(self):
        if self._is_closed():
            raise RuntimeError('Excel file is already closed and saved, which cannot be written anymore!')

    def _close(self):
        self.workbook.close()
        self.workbook = None

    def new_format(self,bold=False,italic=False,underline=False,font_color='#000000',bg_color='#FFFFFF'):

        self._check_closed()
        cell_format = self.workbook.add_format()
        cell_format.set_bold(bold)
        cell_format.set_italic(italic)
        cell_format.set_underline(underline)
        cell_format.set_font_color(font_color)
        cell_format.set_bg_color(bg_color)

        return cell_format

    def set_column_width(self, pos, width, worksheet_name='default'):
        self._check_closed()
        if worksheet_name not in self.worksheet_names:
            raise RuntimeError('Cannot find worksheet with name "%s".' % worksheet_name)
        if isinstance(pos, int):
            self.worksheets[worksheet_name].set_column(pos,pos, width)
        elif isinstance(pos,str):
            self.worksheets[worksheet_name].set_column(pos, width)
        elif isinstance(pos,tuple) or isinstance(pos,list):
            assert len(pos)==2, 'Invalid position setting.'
            start,end=pos
            self.worksheets[worksheet_name].set_column(start,end,width)

    def write(self, cell_name_or_pos, content, worksheet_name='default', format=None):
        self._check_closed()        
        if format is not None:
            assert isinstance(format, Format), 'Invalid cell format.'
        if worksheet_name not in self.worksheet_names:
            raise RuntimeError('Cannot find worksheet with name "%s".' % worksheet_name)
        if isinstance(cell_name_or_pos, tuple) or isinstance(cell_name_or_pos, list):
            assert len(cell_name_or_pos) == 2, 'Invalid cell position.'
            row,col = cell_name_or_pos
            if format:
                self.worksheets[worksheet_name].write(row,col,content,format)
            else:
                self.worksheets[worksheet_name].write(row,col,content)
        elif isinstance(cell_name_or_pos, str):
            if format:
                self.worksheets[worksheet_name].write(cell_name_or_pos,content,format)
            else:
                self.worksheets[worksheet_name].write(cell_name_or_pos,content)
        else:
            raise RuntimeError('Invalid cell name or position. Accepted types are: tuple, list or str but got "%s".' % \
                type(cell_name_or_pos).__name__)

    def save_and_close(self):
        self._close()


class SimpleExcelReader(object):
    def __init__(self, file_path):

        if file_path[-5:]!='.xlsx':
            raise RuntimeError('Invalid file name. File path must ends with ".xlsx".')
        self.file_path = file_path

        self.xlsx = openpyxl.load_workbook(file_path)
    
    def max_row(self, worksheet_name = 'default'):
        return self.xlsx[worksheet_name].max_row
    
    def max_column(self, worksheet_name = 'default'):
        return self.xlsx[worksheet_name].max_column
    
    def read(self, pos: Union[list, tuple], worksheet_name='default'):
        if self.xlsx is None:
            raise RuntimeError('file is already closed.')
        assert len(pos) == 2, 'invalid cell position'
        pos0 = pos[0]+1, pos[1]+1 # cell index starts with 1
        return self.xlsx[worksheet_name].cell(pos0[0], pos0[1]).value
    
    def close(self):
        self.xlsx.close()
        self.xlsx = None

