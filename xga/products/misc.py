#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 7/20/26, 1:34 PM. Copyright (c) The Contributors.
"""Deprecated module previously used for miscellaneous product classes, now moved to xga.products.events."""

import warnings

warnings.warn(
    "xga.products.misc is deprecated and will be removed in a future version. "
    "Please import from xga.products.events instead.",
    DeprecationWarning,
    stacklevel=2,
)
