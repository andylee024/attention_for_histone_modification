#
# Utilities for profiling code.
#

import time

def time_function(f):
    """Given a function, prints the time elapsed after executing function."""

    def print_timing(function_name, duration):
        print "{function}() took {duration} seconds.".format(function=function_name, duration=duration)

    def profile_time(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()

        print_timing(function_name=f.__name__, duration=end_time-start_time)
        return result

    return profile_time
