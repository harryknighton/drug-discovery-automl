"""Define custom types for more-easily readable type checking.

Copyright (c) 2023, Harry Knighton
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

from torch import Tensor

Metrics = dict[str, Tensor]
