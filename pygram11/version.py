# -*- coding: utf-8 -*-

import re

__version__ = "0.1.dev3"
version = __version__
version_info = tuple(re.split(r"[-\.]", __version__))

del re