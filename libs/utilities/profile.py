#
# Utilities for profiling code.
#

import time

def time_function(f):
    """Given a function, prints the time elapsed after executing function."""

    def profile_time(*args, **kwargs):
        t1 = time.time()
        f(*args, **kwargs)
        t2 = time.time()
        print "{function}() took {duration} seconds.".format(function=f.__name__, duration=t2-t1)

    return profile_time
