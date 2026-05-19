"""reVRt constants: standard filenames, layer names, etc"""

ALL = "all"

BARRIER_LAYER_NAME = "transmission_barrier"
"""Combined friction and barrier layer name"""

DEFAULT_DTYPE = "float32"
"""Default dtype used across cost layers"""

METERS_IN_MILE = 1609.344
"""Total meters in a mile (1609.344) - useful for conversions"""

SHORT_MULT = 1.5
"""Short-length spur line multiplier"""

MEDIUM_MULT = 1.2
"""Medium-length spur line multiplier"""

SHORT_CUTOFF = 3 * METERS_IN_MILE / 1000
"""Tie line length below which ``SHORT_MULT`` is applied (3 miles)"""

MEDIUM_CUTOFF = 10 * METERS_IN_MILE / 1000
"""Tie line length below which ``MEDIUM_MULT`` is applied (10 miles)"""
