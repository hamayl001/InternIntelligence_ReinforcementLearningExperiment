
def patch_gym():
    """
    Patch the Gym library to fix the numpy.bool8 issue.
    This needs to be called before importing gym.
    """
    import sys
    import types
    import numpy as np
    
    # Add bool8 as an alias to bool if it doesn't exist
    if not hasattr(np, 'bool8'):
        np.bool8 = np.bool_
    
    # Alternatively, we could patch the specific checker function
    # but this is a more direct approach