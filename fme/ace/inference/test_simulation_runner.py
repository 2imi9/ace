import dataclasses
import datetime
import pathlib

import cftime
import numpy as np
import torch
import xarray as xr

from fme.ace.data_loading.inference import ForcingDataLoaderConfig
from fme.ace.inference.data_writer import DataWriterConfig
from fme.ace.inference.data_writer.file_writer import FileWriterConfig
from fme.ace.inference.inference import InferenceConfig, InitialConditionConfig
from fme.ace.inference.simulation_runner import SimulationRunner
from fme.ace.registry import ModuleSelector
from fme.ace.stepper import StepperConfig
from fme.ace.testing import DimSizes, FV3GFSData
from fme.core.coordinates import (
    DimSize,
    HybridSigmaPressureCoordinate,
    LatLonCoordinates,
)
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset_info import DatasetInfo
from fme.core.logging_utils import LoggingConfig
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.ocean import OceanConfig
from fme.core.step.single_module import SingleModuleStepConfig
from fme.core.step.step import StepSelector
from fme.core.testing import mock_wandb


class PlusOne(torch.nn.Module):
    def forward(self, x):
        return x + 1


def save_stepper(
    path: pathlib.Path,
    in_names: list[str],
    out_names: list[str],
    mean: float,
    std: float,
    horizontal_coords: dict[str, xr.DataArray],
    nz_interface: int,
):
    all_names = list(set(in_names).union(out_names))
    config = StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(
                        type="prebuilt", config={"module": PlusOne()}
                    ),
                    in_names=in_names,
                    out_names=out_names,
                    normalization=NetworkAndLossNormalizationConfig(
                        network=NormalizationConfig(
                            means={name: mean for name in all_names},
                            stds={name: std for name in all_names},
                        ),
                    ),
                    ocean=OceanConfig(
                        surface_temperature_name="sst",
                        ocean_fraction_name="ocean_fraction",
                    ),
                )
            ),
        ),
    )
    horizontal_coordinate = LatLonCoordinates(
        lat=torch.tensor(horizontal_coords["lat"].values, dtype=torch.float32),
        lon=torch.tensor(horizontal_coords["lon"].values, dtype=torch.float32),
    )
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(nz_interface), bk=torch.arange(nz_interface)
    )
    dataset_info = DatasetInfo(
        horizontal_coordinates=horizontal_coordinate,
        vertical_coordinate=vertical_coordinate,
        timestep=datetime.timedelta(hours=6),
        variable_metadata={
            "prog": VariableMetadata(
                units="m",
                long_name="a prognostic variable",
            ),
        },
    )
    stepper = config.get_stepper(
        dataset_info=dataset_info,
    )
    torch.save({"stepper": stepper.get_state()}, path)


def test_simulation_runner_caching(tmp_path, monkeypatch):
    forward_steps_in_memory = 2
    in_names = ["prog", "sst", "forcing_var", "DSWRFtoa"]
    out_names = ["prog", "sst", "ULWRFtoa", "USWRFtoa"]
    stepper_path = tmp_path / "stepper.pt"
    horizontal = [DimSize("lat", 8), DimSize("lon", 16)]
    nz_interface = 4

    dim_sizes = DimSizes(
        n_time=9,
        horizontal=horizontal,
        nz_interface=nz_interface,
    )
    data = FV3GFSData(
        path=tmp_path,
        names=["forcing_var", "DSWRFtoa", "sst", "ocean_fraction"],
        dim_sizes=dim_sizes,
        timestep_days=0.25,
        save_vertical_coordinate=False,
    )
    save_stepper(
        stepper_path,
        in_names=in_names,
        out_names=out_names,
        mean=0.0,
        std=1.0,
        horizontal_coords=data.horizontal_coords,
        nz_interface=nz_interface,
    )

    dims = ["sample", "lat", "lon"]
    initial_condition = xr.Dataset(
        {
            "prog": xr.DataArray(
                np.random.rand(2, 8, 16).astype(np.float32), dims=dims
            ),
            "sst": xr.DataArray(
                np.random.rand(2, 8, 16).astype(np.float32), dims=dims
            ),
            "DSWRFtoa": xr.DataArray(
                np.random.rand(2, 8, 16).astype(np.float32), dims=dims
            ),
        }
    )
    initial_condition["time"] = xr.DataArray(
        [
            cftime.DatetimeProlepticGregorian(2000, 1, 1, 6),
            cftime.DatetimeProlepticGregorian(2000, 1, 1, 12),
        ],
        dims=["time"],
    )
    initial_condition_path = tmp_path / "ic" / "ic.nc"
    initial_condition_path.parent.mkdir()
    initial_condition.to_netcdf(initial_condition_path, mode="w")

    forcing_loader = ForcingDataLoaderConfig(
        dataset=data.inference_data_loader_config.dataset,
        num_data_workers=data.inference_data_loader_config.num_data_workers,
    )

    config = InferenceConfig(
        experiment_dir=str(tmp_path / "experiment"),
        n_forward_steps=4,
        forward_steps_in_memory=forward_steps_in_memory,
        checkpoint_path=str(stepper_path),
        logging=LoggingConfig(
            log_to_screen=True,
            log_to_file=False,
            log_to_wandb=True,
        ),
        initial_condition=InitialConditionConfig(path=str(initial_condition_path)),
        forcing_loader=forcing_loader,
        data_writer=DataWriterConfig(
            save_monthly_files=False,
            save_prediction_files=False,
            files=[FileWriterConfig("autoregressive")],
        ),
        allow_incompatible_dataset=False,
    )

    cache_dir = tmp_path / "cache"
    runner = SimulationRunner(config, cache_dir=cache_dir)
    cache_key = "demo-run"

    with mock_wandb() as wandb:
        wandb.configure(log_to_wandb=True)
        result = runner.run(cache_key=cache_key)
        assert not result.cached
        assert result.metadata_path is not None
        assert result.metadata_path.exists()
        assert "prog" in result.summary.variables
        assert result.summary.time_metadata["n_forward_steps"] == config.n_forward_steps
        assert len(result.summary.time_metadata["initial_times"]) == 2
        assert isinstance(result.summary.anomaly_diagnostics, dict)

        def _fail(*args, **kwargs):
            raise AssertionError("Simulation reran despite cache")

        monkeypatch.setattr(
            "fme.ace.inference.simulation_runner.run_inference",
            _fail,
        )
        cached_result = runner.run(cache_key=cache_key)
        assert cached_result.cached
        assert cached_result.summary == result.summary
