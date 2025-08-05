import xarray as xr
import optax
import dataclasses

from graphcast.data_utils import extract_inputs_targets_forcings
from graphcast.normalization import normalize, unnormalize
from forecast import generate_model
from config import default_experiment_config
from utils import load_stats, get_optimizer
from geg_utils import build_GEG_loss_and_grad, zero_static_variable_updates


if __name__ == "__main__":

    # 1) Prepare Data and Model
    # -----------------------------------------------------------------
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

    # Preprocess inputs and forcings
    ninputs = normalize(inputs, mean_by_level, stddev_by_level)
    nforcings = normalize(forcings, mean_by_level, stddev_by_level)

    predictor, loss, grads = build_GEG_loss_and_grad(
        compute_config=config.compute,
        optimization_config=config.optimization,
        model_config=model_config,
        task_config=task_config,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
    )

    # Initialize optimizer
    optimiser = get_optimizer(config.optimization)
    opt_state = optimiser.init(ninputs)

    # -----------------------------------------------------------------

    # 2) Optimization Loop
    # -----------------------------------------------------------------

    for ep in range(config.optimization.num_epochs):

        loss, diagnostics, grad = grads(params, ninputs, targets, nforcings)

        updates, opt_state = optimiser.update(grad, opt_state, ninputs)
        updates = zero_static_variable_updates(updates)

        ninputs = optax.apply_updates(ninputs, updates)

        print(f"Epoch {ep + 1}/{config.optimization.num_epochs}, Loss: {loss:.4f}")

    # -----------------------------------------------------------------

    # 3) Save outputs and initial values
    # -----------------------------------------------------------------

    # TODO: figure out how to do conversion from optax.Params back to xarray
    inputs = unnormalize(ninputs, mean_by_level, stddev_by_level)
    inputs.to_netcdf(config.data.output_path + "optimized-inputs.nc")

    outputs = predictor(ninputs, targets, nforcings)
    outputs.to_netcdf(config.data.output_path + "optimized-outputs.nc")

    # -----------------------------------------------------------------
