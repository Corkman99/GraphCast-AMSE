import xarray as xr

from .config import default_experiment_config
from forecast import generate_model
from graphcast.data_utils import extract_inputs_targets_forcings
import dataclasses

config = default_experiment_config()

(model_config, task_config, params) = generate_model.load_model(
    config.model.checkpoint_path
)

diffs_stddev_by_level = xr.open_dataset(
    config.model.stats_path + "diffs_stddev_by_level.nc"
)
mean_by_level = xr.open_dataset(config.model.stats_path + "mean_by_level.nc")
stddev_by_level = xr.open_dataset(config.model.stats_path + "stddev_by_level.nc")

predictor = generate_model.build_predictor(
    config.compute,
    model_config,
    task_config,
    params,
    diffs_stddev_by_level=diffs_stddev_by_level,
    mean_by_level=mean_by_level,
    stddev_by_level=stddev_by_level,
)

loss, grads = generate_model.build_loss_and_grad(
    compute_config=config.compute,
    optimization_config=config.optimization,
    model_config=model_config,
    task_config=task_config,
    diffs_stddev_by_level=diffs_stddev_by_level,
    mean_by_level=mean_by_level,
    stddev_by_level=stddev_by_level,
)

data = (
    xr.load_dataset(config.data.input_path, engine="netcdf4")
    .isel(time=slice(2, None))
    .compute()
)
inputs, targets, forcings = extract_inputs_targets_forcings(
    data,
    target_lead_times=slice("6h", f"{config.data.leadtime*6}h"),
    **dataclasses.asdict(task_config),  # type: ignore
)
