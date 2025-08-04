import xarray as xr
import dataclasses


from graphcast.data_utils import extract_inputs_targets_forcings
from forecast import generate_model
from config import default_experiment_config
from utils import load_stats
from geg_utils import build_GEG_loss_and_grad


if __name__ == "__main__":

    config = default_experiment_config()

    (model_config, task_config, params) = generate_model.load_model(
        config.model.checkpoint_path
    )

    diffs_stddev_by_level, mean_by_level, stddev_by_level = load_stats(
        config.model.stats_path
    )

    data = (
        xr.load_dataset(config.data.input_path, engine="netcdf4")
        .isel(time=slice(2, 5))
        .compute()
    )
    inputs, targets, forcings = extract_inputs_targets_forcings(
        data,
        target_lead_times=slice("6h", f"{config.data.leadtime * 6}h"),
        **dataclasses.asdict(task_config),  # type: ignore
    )

    # TODO: Preprocess inputs and forcings

    predictor, loss, grads = build_GEG_loss_and_grad(
        compute_config=config.compute,
        optimization_config=config.optimization,
        model_config=model_config,
        task_config=task_config,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
    )
