import xarray as xr
from config import default_experiment_config
from forecast import generate_model
from utils import extract_extended_inputs_targets_forcings, load_stats

if __name__ == "__main__":

    exp = default_experiment_config()

    (model_config, task_config, params) = generate_model.load_model(
        exp.model.checkpoint_path
    )

    diffs_stddev_by_level, mean_by_level, stddev_by_level = load_stats(
        exp.model.stats_path
    )

    data = (
        xr.load_dataset(exp.data.input_path, engine="netcdf4")
        .isel(time=slice(0, 2))
        .compute()
    )
    inputs, targets, forcings = extract_extended_inputs_targets_forcings(
        data,
        task_config,
        exp.data.leadtime
    )

    predictor = generate_model.build_predictor(
        exp.compute,
        model_config,
        task_config,
        params,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
    )

    print(inputs)
    print(targets)
