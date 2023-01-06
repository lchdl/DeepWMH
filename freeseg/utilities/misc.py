from contextlib import contextmanager
from freeseg.utilities.file_ops import file_exist, join_path, mkdir
import os, sys, time, datetime
import signal
import psutil

def kill_process_tree(pid, kill_self=True):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess: # already killed or due to some other reasons
        return
    childs = parent.children(recursive=True)
    for child in childs:
        child.kill()
    if kill_self:
        parent.kill()

def print_ts(*args,**kwargs):
    ts = '['+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'] '
    if 'start' in kwargs:
        print(kwargs['start'],end='')
        kwargs.pop('start')
    print(ts,end='')
    print(*args,**kwargs)

def printi(*args):
    '''
    immediate print without '\\n'
    '''
    print(*args,end='')
    sys.stdout.flush()

def printx(msg):
    '''
    single line erasable output.
    '''
    assert isinstance(msg,str),'msg must be a string object.'
    
    try:
        columns = list(os.get_terminal_size())[0]
    except Exception:
        # get_terminal_size() failed,
        # probably due to invalid output device, then we ignore and return
        return
    except BaseException:
        raise

    outsize = columns-1

    print('\r' +' '*outsize + '\r',end='')
    print(msg[0:outsize],end='')
    sys.stdout.flush()

def printv(*args, **kwargs):
    '''
    print with verbose setting.
    '''
    if 'verbose' in kwargs:
        verbose = kwargs.pop('verbose')
    else:
        verbose = False
    if verbose:
        print(*args,**kwargs)

def minibar(msg=None,a=None,b=None,time=None,fill='=',length=20,last=None):
    if length<5: length=5
    perc = 0.0
    if b != 0:
        perc = a/b

    na=int((length-2)*perc)
    if na<0: na=0
    if na>length-2: na=length-2
    head = ('%s : '%msg) if len(msg)>0 else ''
    bar = '|'+fill*na+' '*(length-2-na)+'|'+' %d%%' % int(100.0*perc)

    def _format_sec(t):
        s = t%60; t = t//60
        m = t%60; t = t//60
        h = t%24; t = t//24
        d = t
        if d>30: 
            # this task is too long... if you run a task that will execute for more than 30 days,
            # it seems more wisely to terminate the program and think about some other workarounds
            # to reduce the execution time :-)
            return '>30d' 
        else:
            if d>0: return '%dd%dh' % (d,h)
            elif h>0: return '%dh%dm' % (h,m)
            elif m>0: return '%dm%ds' % (m,s)
            else: return '%ds' % s

    time_est = ''
    if time is not None:
        if a == 0:
            elapsed = int(time)
            time_est = ' << '+_format_sec(elapsed)
        else:
            elapsed, remaining = int(time), int(time*(b-a)/a)
            time_est = ' << '+_format_sec(elapsed)
            time_est += '|'+_format_sec(remaining)

    if last is None:
        last = ''

    printx(head+bar+time_est+' '+last)

# simple text logging
class SimpleTxtLog(object):
    def __init__(self, location):
        self.location = location
        with open(self.location, 'w') as f: pass
    
    def now(self):
        ts = '['+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+']'
        return ts

    def write(self, msg, timestamp = False, auto_newline=True):
        msg0 = msg if timestamp == False else self.now() + ' ' + msg
        if auto_newline:
            msg0 += '\n'
        with open(self.location, 'a') as f:
            f.write(msg0)
    
class Timer(object):
    def __init__(self, tick_now=True):
        self.time_start = time.time()
        self.time_end = 0
        if tick_now:
            self.tick()
        self.elapse_start = self.time_start
    def tick(self):
        self.time_end = time.time()
        dt = self.time_end - self.time_start
        self.time_start = self.time_end
        return dt
    def elapsed(self):
        self.elapse_end = time.time()
        return self.elapse_end - self.elapse_start
    def now(self):
        string = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return string

class TimeStamps(object):
    def __init__(self):
        self.tstamps = {}

    def _format_now(self): # return a formatted time string
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return ts

    def record(self, name):
        self.tstamps[name] = self._format_now() # save

    def get(self, name):
        if name not in self.tstamps:
            return "unknown"
        else:
            return self.tstamps[name]

@contextmanager
def ignore_SIGINT():
    '''
    Description
    ------------
    Temporarily ignore keyboard interrupt signal (SIGINT).

    Usage
    ------------
    >>> with ignore_SIGINT():
    >>>     # do something here, SIGINT ignored
    >>>     # ...
    >>> # SIGINT no longer ignored here
    '''
    last_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    yield
    signal.signal(signal.SIGINT, last_handler) # restore default handler

@contextmanager
def ignore_print():
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    yield
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout = _original_stdout
    sys.stderr = _original_stderr

class Checkpoints(object):
    def __init__(self, save_folder):
        self._save_folder = mkdir(save_folder)
        self._is_disabled = False

    def disable_all_checkpoints(self):
        self._is_disabled = True
    def enable_all_checkpoints(self):
        self._is_disabled = False

    def is_finished(self, ckpt_name):
        if self._is_disabled: # if all checkpoints are disabled then we just ignore all finished checkpoints
                             # this is useful for debugging
            return False
        if file_exist(join_path(self._save_folder, ckpt_name)):
            return True
        else:
            return False
    def set_finish(self, ckpt_name):
        with open(join_path(self._save_folder, ckpt_name), 'w') as f:
            pass # only create an empty file to indicate that the checkpoint is finished. 
    

def contain_duplicates(l: list):
    '''
    check if a list contains duplicated items.
    '''
    assert isinstance(l, list), 'object should be a list.'
    if len(l) != len(set(l)): return True
    else: return False

def remove_duplicates(l:list):
    return list(dict.fromkeys(l))


def remove_items(l,s):
    newl = []
    for item in l:
        print(item)
        if item not in s:
            newl.append(item)
        else:
            print('delete', item)
    return newl
