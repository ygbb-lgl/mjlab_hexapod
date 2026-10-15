"""Tethered Hexapod robot asset."""

from .hexapod_tethered_constants import (
  HEXAPOD_TETHERED_ACTION_SCALE as HEXAPOD_TETHERED_ACTION_SCALE,
)
from .hexapod_tethered_constants import (
  get_hexapod_tethered_robot_cfg as get_hexapod_tethered_robot_cfg,
)
from .hexapod_tethered_constants import (
  get_hexapod_tethered_slope_mixture_robot_cfg as _get_slope_mixture_robot_cfg,
)
from .hexapod_tethered_constants import (
  get_hexapod_tethered_slope_robot_cfg as get_hexapod_tethered_slope_robot_cfg,
)

get_hexapod_tethered_slope_mixture_robot_cfg = _get_slope_mixture_robot_cfg
