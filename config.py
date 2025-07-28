from pydantic import BaseModel
from typing import Literal, Callable, Optional

Rollout = Literal[
    "full",
    "chunked",
]


class ComputeConfig(BaseModel):
    use_gpu: bool = True
    cache_dir: str
    rollout_method: Rollout
    random_seed: int = 0
    use_bfloat16: bool = True


class ModelConfig(BaseModel):
    checkpoint_path: str
    stats_path: str


class DataConfig(BaseModel):
    input_path: str
    leadtime: int  # num steps to predict, cropped from endtime
    reference_path: Optional[str]  # e.g. ERA5
    output_path: str


class OptimizationConfig(BaseModel):
    num_epochs: int = 10
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    loss_function: Callable


class ExperimentConfig(BaseModel):
    """
    A configuration class that combines compute, data, and optimization configurations
    for running experiments.
    """

    compute: ComputeConfig
    model: ModelConfig
    data: DataConfig
    optimization: OptimizationConfig


def default_experiment_config() -> ExperimentConfig:
    import os
    from trainer.loss_utils import make_amse_loss

    home = os.path.expanduser("~")

    compute = ComputeConfig(
        use_gpu=True,
        cache_dir=os.path.join(home, "AMSE_cache"),
        rollout_method="full",
        random_seed=0,
        use_bfloat16=True,
    )
    model = ModelConfig(
        checkpoint_path=os.path.join(home, "scratch/Data/GraphCast_OP/amse.ckpt"),
        stats_path=os.path.join(home, "scratch/Data/GraphCast_small/stats/"),
    )
    data = DataConfig(
        input_path=os.path.join(
            home,
            "scratch/Data/GraphCast_OP/custom-hres_2022-09-26_res-0.25_levels-13_steps-16.nc",
        ),
        leadtime=14,
        reference_path=None,  # TODO: use input_path only to search for two input states, reference path has forecast
        output_path=os.path.join(home, "scratch/GraphCast-OP_TC_5day/AMSE/"),
    )

    PVW = {
        "2m_temperature": 1.0,
        "10m_u_component_of_wind": 0.1,
        "10m_v_component_of_wind": 0.1,
        "mean_sea_level_pressure": 0.1,
        "total_precipitation_6hr": 0.1,
    }

    custom_loss = make_amse_loss(
        model.checkpoint_path,
        model.stats_path,
        per_variable_weights=PVW,
        compute_wind_speed=True,
    )

    optim = OptimizationConfig(
        num_epochs=10, optimizer="adam", learning_rate=1e-3, loss_function=custom_loss
    )

    return ExperimentConfig(compute=compute, model=model, data=data, optimization=optim)
