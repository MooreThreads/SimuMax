"""Test module mem"""

import os
import pytest

from simumax.testing import ResultCheck
from simumax.core.base_struct import InputOutputInfo, TensorSize
from simumax.core.config import SystemConfig, ModelConfig, StrategyConfig
from simumax.core.perf_llm import PerfLLM
from simumax.core.utils import HumanReadableSize


mem_case_list = [
    {
        "system_file": "configs_for_test/system/a100_bf16.json",
        "startegy_file": "configs_for_test/strategy/tp2_pp1_dp4_mbs2_no_ckpt.json",
        "model_file": "configs_for_test/models/llama2-7b.json",
        "input_size": InputOutputInfo(tensors=[TensorSize(shape=(2, 4096, 4096))]),
        "golden": {
            "fwd_peak_allocated_mem": "47732.6015625 MB",
            "bwd_peak_allocated_mem": "48740.953125 MB",
        },
        "source": "megatron",
    }
]


@pytest.mark.parametrize("case", mem_case_list)
def test_end2end_mem(case):
    current_path = os.path.dirname(__file__)
    system_config = SystemConfig.init_from_config_file(
        os.path.join(current_path, case["system_file"])
    )
    strategy_config = StrategyConfig.init_from_config_file(
        os.path.join(current_path, case["startegy_file"])
    )
    model_config = ModelConfig.init_from_config_file(
        os.path.join(current_path, case["model_file"])
    )

    model = PerfLLM()
    model.configure(
        strategy_config=strategy_config,
        model_config=model_config,
        system_config=system_config,
        debug_points=[],
        debug_points_last_stage=[],
    )
    model.run_estimate()
    mem_result = model.analysis_mem()

    result_check = ResultCheck(rtol=5e-2)

    result = {
        "fwd_peak_allocated_mem": mem_result.get("fwd_peak_allocated_mem"),
        "bwd_peak_allocated_mem": mem_result.get("bwd_peak_allocated_mem"),
    }
    result = {
        k: HumanReadableSize.from_string(
            v, base=1024, units=HumanReadableSize.BYTE_UNITS, target_unit="B"
        ).get_value()
        for k, v in result.items()
    }
    golden = {
        k: HumanReadableSize.from_string(
            v, base=1024, units=HumanReadableSize.BYTE_UNITS, target_unit="B"
        ).get_value()
        for k, v in case["golden"].items()
    }
    assert result_check(result, golden)
