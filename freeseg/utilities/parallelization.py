##########################################################
# Implement simple and robust process-based parallelism. # 
##########################################################

from time import sleep
from typing import Callable, List
from freeseg.utilities.misc import kill_process_tree, minibar, Timer
import multiprocessing
import traceback
from multiprocessing import Pool
import os, sys


# mute current process
def _mute_this():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

class ParallelRuntimeError(Exception):
    pass

class ParallelFunctionExceptionWrapper(object):
    '''
    A simple wrapper class that wraps a Python function to have basic exception handling mechanics
    '''
    def __init__(self, callable_object):
        self.__callable_object = callable_object
    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable_object(*args, **kwargs)
        except BaseException as e:
            multiprocessing.get_logger().error( traceback.format_exc() )
            try:
                kill_process_tree(os.getpid(), kill_self=False)
            except: pass
            trace = traceback.format_exc()
            raise ParallelRuntimeError(trace)
        return result

# process-based parallelism
def run_parallel(worker_function:Callable, list_of_tasks_args: List[tuple], num_workers:int, progress_bar_msg:str, 
    print_output:bool=False, show_progress_bar:bool=True):
    '''
    Description
    ------------
    Process-based parallelism. Using multiple CPU cores to execute the same function using different
    parameters. Each worker function is independent and it should not communicate with other workers
    when running.

    Parameters
    ------------
    "worker_function": <function object>
        Function that will be executed concurrently.
    "list_of_tasks_args": list
        A list of arguments for all tasks. Its format should be something like this:  
        [ (arg1, arg2, ...), (arg1, arg2, ...), (arg1, arg2, ...) , ... ]
          ^ worker1 params   ^ worker2 params   ^ worker3 params    ...
    "num_workers": int
        Number of workers.
    "progress_bar_msg": str
        A short string that will be displayed in progress bar.
    "print_output": bool
        If you don't want to see any output or error message during parallel execution, 
        set it to False (default). Otherwise turn it on.
    "show_progress_bar": bool
        If you don't want to show progress bar set this to False (default: True).

    Note
    ------------
    * Currently nested parallelism is not supported.
    * Please DO NOT call this function inside a worker function! The behavior might be unexpected.
    * The worker function can safely invoke a bash call using run_shell(...). You can encapsulate
    the run_parallel(...) in a ignore_SIGINT() context to temporarily disable keyboard interrupt 
    (Ctrl+C) such as:

    >>> from freeseg.utilities.misc import ignore_SIGINT
    >>> with ignore_SIGINT():
    >>>     run_parallel(...)

    * For more detailed usage please see the examples provided below.
    
    Example
    ------------
    >>> tasks = [ (arg1, arg2, ...), (arg1, arg2, ...), ... ]
    >>> # define your worker function here
    >>> def _worker_function(params):
    >>>     arg1, arg2, ... = params
    >>>     # do more work here,
    >>>     # but please DO NOT call "run_parallel" again in this worker function!
    >>>     # ...
    >>> # start running 8 tasks in parallel
    >>> run_parallel(_worker_function, tasks, 8, "working...")
    '''

    pool = Pool(processes=num_workers, initializer=_mute_this if not print_output else None)
    total_tasks = len(list_of_tasks_args)
    if total_tasks == 0: 
        # no task to execute in parallel, just return
        return
    tasks_returned = []
    tasks_error = []

    for i in range(total_tasks):
        pool.apply_async(
            ParallelFunctionExceptionWrapper(worker_function), 
            (list_of_tasks_args[i],) , 
            callback=tasks_returned.append,
            error_callback=tasks_error.append
        )
        
    pool.close() # no more work to append
    # now waiting all tasks to finish
    # and we draw a simple progress bar to track real time progress 
    timer = Timer()
    finished, error_ = 0, 0
    anim_counter, anim_chars = 0, '-\|/' # simple animation effect
    try:
        while True:
            finished = len(tasks_returned)
            error_ = len(tasks_error)

            if error_ > 0:
                trace = tasks_error[0] # only retrieve first error trace
                
                error_msg = '\n\n==========\n'  \
                            'One of the worker process crashed due to unhandled exception.\n' \
                            'Worker function name is: "%s".\n\n' % worker_function.__name__
                error_msg += '** Here is the traceback message:\n\n'
                error_msg += str(trace)
                error_msg += '\n** Main process will exit now.\n'
                error_msg += '==========\n\n'
                print(error_msg)
                raise RuntimeError('One of the worker process crashed due to unhandled exception.')
                
            if show_progress_bar: 
                minibar(msg=progress_bar_msg, a=finished, b=total_tasks, time=timer.elapsed(), last=anim_chars[anim_counter])
                anim_counter += 1
                anim_counter %= len(anim_chars)

            # end condition
            if finished == total_tasks:
                if show_progress_bar: 
                    minibar(msg=progress_bar_msg, a=finished, b=total_tasks, time=timer.elapsed()) # draw bar again
                break
            else:
                sleep(0.2)
        print('')
        pool.join()
    except: 
        # exit the whole program if anything bad happens (for safety)
        try: # try killing all its child, in case worker process
             # spawns child processes 
            for wproc in pool._pool: # for each worker process in pool
                kill_process_tree(wproc.pid, kill_self=False)
                sleep(0.5)
        except: 
            pass # omit any exception when killing child process
        pool.terminate()
        pool.join()
        exit(1)

    return tasks_returned

