import logging
from typing import Optional, Tuple

import xarray

from graphcast import predictor_base
from graphcast import xarray_tree
from config import ComputeConfig, OptimizationConfig


def _normalize(
    values: xarray.Dataset,
    scales: xarray.Dataset,
    locations: Optional[xarray.Dataset],
) -> xarray.Dataset:
    """Normalize variables using the given scales and (optionally) locations."""

    def _normalize_array(array):
        if array.name is None:
            raise ValueError(
                "Can't look up normalization constants because array has no name."
            )
        if locations is not None:
            if array.name in locations:
                array = array - locations[array.name].astype(array.dtype)
            else:
                logging.warning("No normalization location found for %s", array.name)
        if array.name in scales:
            array = array / scales[array.name].astype(array.dtype)
        else:
            logging.warning("No normalization scale found for %s", array.name)
        return array

    return xarray_tree.map_structure(_normalize_array, values)


def _unnormalize(
    values: xarray.Dataset,
    scales: xarray.Dataset,
    locations: Optional[xarray.Dataset],
) -> xarray.Dataset:
    """Unnormalize variables using the given scales and (optionally) locations."""

    def _unnormalize_array(array):
        if array.name is None:
            raise ValueError(
                "Can't look up normalization constants because array has no name."
            )
        if array.name in scales:
            array = array * scales[array.name].astype(array.dtype)
        else:
            logging.warning("No normalization scale found for %s", array.name)
        if locations is not None:
            if array.name in locations:
                array = array + locations[array.name].astype(array.dtype)
            else:
                logging.warning("No normalization location found for %s", array.name)
        return array

    return xarray_tree.map_structure(_unnormalize_array, values)


class NormalizedInputsAndResiduals(predictor_base.Predictor):
    """
    Class to replace graphcast.normalization.InputsAndResiduals wrapper
    We assume instead normalized inputs and forcings, such that we are taking
    the gradient of the loss wrt to normalized inputs. The optimization loop can then
    be run in normalized space.
    """

    def __init__(
        self,
        predictor: predictor_base.Predictor,
        stddev_by_level: xarray.Dataset,
        mean_by_level: xarray.Dataset,
        diffs_stddev_by_level: xarray.Dataset,
    ):
        self._predictor = predictor
        self._scales = stddev_by_level
        self._locations = mean_by_level
        self._residual_scales = diffs_stddev_by_level
        self._residual_locations = None

    def _add_input_in_unnormalized_space(self, norm_inputs, norm_prediction):
        if norm_prediction.sizes.get("time") != 1:
            raise ValueError(
                "normalization.InputsAndResiduals only supports predicting a "
                "single timestep."
            )
        if norm_prediction.name in norm_inputs:
            last_input = norm_inputs[norm_prediction.name].isel(time=-1)
            unnormalized_last_input = _unnormalize(
                last_input, self._scales, self._locations
            )
            unnormalized_prediction = _unnormalize(
                norm_prediction, self._scales, self._locations
            )
            unnormalized_prediction += unnormalized_last_input
            return _normalize(unnormalized_prediction, self._scales, self._locations)
        else:
            # A predicted variable which is not an input variable. We are predicting
            # it directly, so unnormalize it directly to the target scale/location:
            return norm_prediction

    def _subtract_input_and_normalize_target(self, inputs, target):
        if target.sizes.get("time") != 1:
            raise ValueError(
                "normalization.InputsAndResiduals only supports wrapping predictors"
                "that predict a single timestep."
            )
        if target.name in inputs:
            target_residual = target
            last_input = inputs[target.name].isel(time=-1)
            target_residual -= last_input
            return _normalize(
                target_residual, self._residual_scales, self._residual_locations
            )
        else:
            return _normalize(target, self._scales, self._locations)

    def __call__(
        self,
        inputs: xarray.Dataset,
        targets_template: xarray.Dataset,
        forcings: xarray.Dataset,
        **kwargs,
    ) -> xarray.Dataset:
        # norm_inputs = normalize(inputs, self._scales, self._locations)
        # norm_forcings = normalize(forcings, self._scales, self._locations)
        norm_predictions = self._predictor(
            inputs, targets_template, forcings=forcings, **kwargs
        )
        return xarray_tree.map_structure(
            lambda pred: self._add_input_in_unnormalized_space(inputs, pred),
            norm_predictions,
        )

    def loss(
        self,
        inputs: xarray.Dataset,
        targets: xarray.Dataset,  # is unnormalized
        forcings: xarray.Dataset,
        **kwargs,
    ) -> predictor_base.LossAndDiagnostics:
        """Returns the loss computed on normalized inputs and targets."""
        # norm_inputs = normalize(inputs, self._scales, self._locations)
        # norm_forcings = normalize(forcings, self._scales, self._locations)
        norm_target_residuals = xarray_tree.map_structure(
            lambda t: self._subtract_input_and_normalize_target(inputs, t), targets
        )
        return self._predictor.loss(
            inputs, norm_target_residuals, forcings=forcings, **kwargs
        )

    def loss_and_predictions(  # pytype: disable=signature-mismatch  # jax-ndarray
        self,
        inputs: xarray.Dataset,
        targets: xarray.Dataset,
        forcings: xarray.Dataset,
        **kwargs,
    ) -> Tuple[predictor_base.LossAndDiagnostics, xarray.Dataset]:
        """The loss computed on normalized data, with normalized predictions."""
        # norm_inputs = normalize(inputs, self._scales, self._locations)
        # norm_forcings = normalize(forcings, self._scales, self._locations)
        norm_target_residuals = xarray_tree.map_structure(
            lambda t: self._subtract_input_and_normalize_target(inputs, t), targets
        )
        (loss, scalars), predictions = self._predictor.loss_and_predictions(
            inputs, norm_target_residuals, forcings=forcings, **kwargs
        )
        predictions = xarray_tree.map_structure(
            lambda pred: self._add_input_in_unnormalized_space(inputs, pred),
            predictions,
        )
        return (loss, scalars), predictions


def build_GEG_loss_and_grad(
    compute_config: ComputeConfig,
    optimization_config: OptimizationConfig,
    model_config,
    task_config,
    diffs_stddev_by_level,
    mean_by_level,
    stddev_by_level,
):
    import jax
    import haiku as hk
    from graphcast import (
        graphcast,
        casting,
        autoregressive,
        xarray_jax,
        xarray_tree,
    )
    import xarray as xr
    import functools

    def construct_wrapped_graphcast(model_config, task_config):
        predictor = graphcast.GraphCast(model_config, task_config)

        # If running on a GPU, operate in BFloat16 mode
        if compute_config.use_bfloat16:
            predictor = casting.Bfloat16Cast(predictor)

        # Wrap in custom class
        predictor = NormalizedInputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level,
        )

        # And wrap in the autoregressive magic to take multi-step predictions.
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)

        return predictor

    @hk.transform_with_state
    def loss_fn(model_config, task_config, inputs, targets, forcings):
        predictor = construct_wrapped_graphcast(model_config, task_config)
        if optimization_config.loss_function is not None:
            loss, diagnostics = optimization_config.loss_function(
                predictor(inputs, targets, forcings), targets
            )
        else:
            loss, diagnostics = predictor.loss(inputs, targets, forcings)
        return xarray_tree.map_structure(
            lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
            (loss, diagnostics),
        )

    def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
        def _aux(params, state, i, t, f):
            (loss, diagnostics), next_state = loss_fn.apply(
                params,
                state,
                jax.random.PRNGKey(compute_config.random_seed),
                model_config,
                task_config,
                i,
                t,
                f,
            )
            return loss, (diagnostics, next_state)

        (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
            _aux, has_aux=True, argnums=4  # wrt inputs
        )(params, state, inputs, targets, forcings)
        return loss, diagnostics, next_state, grads

    def with_configs(fn):
        return functools.partial(fn, model_config=model_config, task_config=task_config)

    # def with_params(fn):
    #     return functools.partial(fn, params=params, state=state)

    # def drop_state(fn)
    #     return lambda **kw: fn(**kw)[0]

    jit_loss = jax.jit(with_configs(loss_fn.apply))
    jit_grad = jax.jit(with_configs(grads_fn))

    def loss_wrapper(params, inputs, targets, forcings):
        ((loss, diagnostics), _) = jit_loss(
            params=params,
            inputs=inputs,
            targets=targets,
            forcings=forcings,
            rng=jax.random.PRNGKey(compute_config.random_seed),
            state={},
        )
        return (loss, diagnostics)

    def grad_wrapper(params, inputs, targets, forcings):
        (loss, diagnostics, _, grad) = jit_grad(
            params=params, inputs=inputs, targets=targets, forcings=forcings, state={}
        )
        return (loss, diagnostics, grad)

    return (
        construct_wrapped_graphcast(model_config, task_config),
        loss_wrapper,
        grad_wrapper,
    )
