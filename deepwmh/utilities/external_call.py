import os
import subprocess
from time import sleep
from typing import Union
from deepwmh.utilities.misc import ignore_SIGINT, kill_process_tree
import shlex

# emulating shell command
def run_shell(command:str, print_command:bool=True, print_output:bool=True, force_continue:bool=False,
    env_vars: Union[dict, None] = None):
    '''
    Description
    ----------------
    run_shell: send command to shell by simply using a Python function call. 

    Paramaters
    ----------------
    command: str
        command that will be sent to shell for execution
    print_command: bool
        if you want to see what command is being executed 
        you can set print_command=True.
    print_output: bool 
        if you want to see outputs from subprocess you need
        to set this to True.
    force_continue: bool 
        if force_continue=True, then the main process will
        still continue even if error occurs in sub-process.
    env_vars: dict or None
        if set, this param will overload environment variables
        of the child process.

    Returns
    ----------------
    retcode: int
        return value from sub-process.
    '''
    if print_command:
        print(command)
    retcode = None
    stdout, stderr = None, None # default
    if print_output == False:
        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL
    
    # now start the process
    retcode = None
    try:
        p = None
        args = shlex.split(command)
        p = subprocess.Popen(args, shell=False, stdout=stdout, stderr=stderr, env=env_vars)
        retcode = p.wait()
    except BaseException as e:
        if p is not None:
            with ignore_SIGINT():
                kill_process_tree(p.pid, kill_self=True)
                sleep(3.0) # wait till all messes are cleaned up

        if not isinstance(e, Exception):
            # re-raise system error
            raise e

    # handling return values
    if retcode != 0:
        s=''
        if retcode == None: s = 'None'
        else: s = str(retcode)
        print('\n>>> ** Unexpected return value "%s" from command:\n' % s)
        print('>>>',command)
        if force_continue==False:
            print('\n>>> ** Process will exit now.\n')
            exit('Error occurred (code %s) when executing command:\n"%s"' % (s, command))

    return retcode

def try_shell(command: str, stdio = False):
    '''
    Try running a command and get its return value or stdio strings (stdout, stderr).
    '''
    retval = None
    strout, strerr = None, None
    try:
        p = None
        args = shlex.split(command)
        if stdio == False: # return code instead
            p = subprocess.Popen(args, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            retval = p.wait()
        else:
            p = subprocess.Popen(args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            strout, strerr = p.communicate()
    except BaseException as e:
        if p is not None:
            kill_process_tree(p.pid)
        if not isinstance(e, Exception):
            raise e
    if stdio:
        return strout.decode("utf-8"), strerr.decode("utf-8")
    else:
        return retval

def ls_tree(folder,depth=2,stat_size=False,closed=False):
    '''
    Description
    -------------
    ls_tree: similar to "ls" command in unix-based systems but can print file hierarchies and 
    display file sizes.

    Parameters
    -------------
    folder: str
        path to folder that wants to be displayed.
    depth: int
        display depth.
    stat_size: bool
        display file sizes.
    closed: bool
        display directory in open/closed style.
    '''

    def __get_size_in_bytes(start_path):
        '''
        from https://stackoverflow.com/questions/1392413/calculating-a-directorys-size-using-python
        '''
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return total_size
    
    def __size_format(size_in_bytes):
        if size_in_bytes < 1024:
            return '%dB' % size_in_bytes
        elif size_in_bytes >=1024 and size_in_bytes <=1024*1024-1:
            return '%dKB' % (size_in_bytes//1024)
        elif size_in_bytes >=1024*1024 and size_in_bytes <=1024*1024*1024-1:
            return '%.2fMB' % (size_in_bytes/(1024*1024))
        else:
            return '%.2fGB' % (size_in_bytes/(1024*1024*1024))

    def __abbr_filename(filename, length):
        if len(filename)<=length: return filename
        else:
            return '...'+filename[len(filename)-length+3:]

    def __max_item_len(l):
        maxl = 0
        for item in l:
            if len(item)> maxl:
                maxl = len(item)
        return maxl

    def __rreplace_once(s,subs,replace):
        return replace.join(s.rsplit(subs, 1))

    def __ls_ext_from(folder,indent,cur_depth,max_depth,max_len=24):
        # indent: spaces added in each indent level
        # folder: folder path in current search
        # max_len: maximum length of file name

        prespaces =  ('|' + ' '*(indent-1)) * cur_depth
        terminal_cols = os.get_terminal_size()[0]
        
        items_all = os.listdir(folder)
        items_f = sorted([item for item in items_all if os.path.isfile(os.path.join(folder, item)) == True])
        items_d = sorted([item for item in items_all if os.path.isfile(os.path.join(folder, item)) == False])

        abbr_items_f_repr = [ __abbr_filename(item,max_len-1)+' ' for item in items_f ]
        abbr_items_d_repr = [ __abbr_filename(item,max_len-1)+'/' for item in items_d ]

        if cur_depth == max_depth:
            max_repr_len = __max_item_len(abbr_items_f_repr + abbr_items_d_repr)

            disp_cols = (terminal_cols - indent*cur_depth) // (max_repr_len+2)
            if disp_cols < 1: disp_cols = 1
            repr_string = '' + prespaces
            abbr_items_repr = abbr_items_f_repr + abbr_items_d_repr
            for tid in range(len(abbr_items_repr)):
                repr_string += abbr_items_repr[tid]
                if ( (tid+1) % disp_cols == 0 ) and (tid != len(abbr_items_repr)-1): # newline
                    repr_string += '\n' + prespaces
                else:
                    pad = ' ' * (max_repr_len-len(abbr_items_repr[tid]))
                    repr_string += pad + '  '
            if len(abbr_items_repr)>0 and closed == False:
                print(repr_string)
        else:
            max_repr_len = __max_item_len(abbr_items_f_repr) if len(abbr_items_f_repr)>0 else max_len
            # print files
            disp_cols = (terminal_cols - indent*cur_depth) // (max_repr_len+2)
            if disp_cols < 1: disp_cols = 1
            repr_string = '' + prespaces
            abbr_items_repr = abbr_items_f_repr
            for tid in range(len(abbr_items_repr)):
                repr_string += abbr_items_repr[tid]
                if ( (tid+1) % disp_cols == 0 ) and (tid != len(abbr_items_repr)-1): # newline
                    repr_string += '\n' + prespaces
                else:
                    pad = ' ' * (max_repr_len-len(abbr_items_repr[tid]))
                    repr_string += pad + '  '
            if len(abbr_items_repr)>0:
                print(repr_string)

            # recursive print dirs
            for it, it_repr in zip(items_d, abbr_items_d_repr):
                if it_repr[-1] == '/': # is directory
                    fc = len(os.listdir(os.path.join(folder, it))) # number of files in directory
                    subs = '|' + ' '*(indent-1)
                    replace = '+' + '-'*(indent-1)
                    # replace "|  " to "+--"
                    prespaces2 = __rreplace_once(prespaces,subs,replace)
                    if stat_size:
                        size_str = __size_format(__get_size_in_bytes(os.path.join(folder, it)))
                        print( prespaces2 + it + ' +%d item(s)' % fc + ', ' + size_str)
                    else:
                        print( prespaces2 + it + ' +%d item(s)' % fc)
                    __ls_ext_from(os.path.join(folder, it),indent, cur_depth+1,max_depth,max_len=max_len)
                else:
                    print(prespaces + it_repr)

    # check if folder exists
    folder = os.path.abspath(folder)
    if os.path.exists(folder) and os.path.isdir(folder) == False:
        raise FileNotFoundError('folder not exist: "%s".' % os.path.abspath(folder))
    folder = folder + os.path.sep

    foldc = os.listdir(folder)
    if stat_size:
        size_str = __size_format(__get_size_in_bytes(folder))
        print('file(s) in "%s"  +%d item(s), %s' % (folder, len(foldc), size_str))
    else:
        print('file(s) in "%s"  +%d item(s)' % (folder, len(foldc)))
    if len(foldc) == 0: # empty root folder
        print('(folder is empty)')
        return
    else:
        __ls_ext_from(folder, 3, 1, max_depth=depth)
