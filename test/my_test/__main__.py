# -*- coding: utf-8 -*-
"""``python -m my_test`` 的入口（等价于 ``python test/my_test/run_all.py``）。"""

from __future__ import annotations

import sys

from .run_all import main

raise SystemExit(main(sys.argv[1:]))
