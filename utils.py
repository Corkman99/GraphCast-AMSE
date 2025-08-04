import os
import xarray as xr
from typing import Tuple
import dataclasses

from graphcast.graphcast import TaskConfig


def load_stats(path: str) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    files = ["diffs_stddev_by_level.nc", "mean_by_level.nc", "stddev_by_level.nc"]
    return (
        xr.open_dataset(os.path.join(path, files[0])),
        xr.open_dataset(os.path.join(path, files[1])),
        xr.open_dataset(os.path.join(path, files[2])),
    )


def extract_extended_inputs_targets_forcings(
    data: xr.Dataset,
    task_config: TaskConfig,
    leadtime: int,
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    from graphcast.data_utils import extract_inputs_targets_forcings
    from graphcast.rollout import extend_targets_template

    inputs, targets, forcings = extract_inputs_targets_forcings(
        data,
        target_lead_times=("6h",),
        **dataclasses.asdict(task_config),  # type: ignore
    )
    targets = extend_targets_template(
        targets_template=targets, required_num_steps=leadtime
    )
    return inputs, targets, forcings
