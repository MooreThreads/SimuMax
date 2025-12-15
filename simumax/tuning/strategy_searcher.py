"""grid search for strategy"""

from functools import reduce
import operator
import itertools
from copy import deepcopy
from tqdm import tqdm

from simumax.core.config import StrategyConfig
from simumax.core.perf_llm import PerfLLM

CANDIDATE_STRATEGY = {
    "seq_len": [4096],
    "world_size": [1024],
    "micro_batch_size": [1],
    "micro_batch_num": [1],
    "dtype": ["bf16"],
    "tp_size": [1, 2, 4, 8],
    "enable_sequence_parallel": [True],
    "interleaving_size": [1],
    "zero_state": [1],
    "use_fused_norm": [True],
    "use_math_sdp": [False],
    "use_flash_sdp": [True],
    "use_fp32_accum_grad": [True],
    "use_fused_swiglu": [True],
    "enable_recompute": [False, True],
    "recompute_granularity": ["full_block"],
    "mem_factor": [0.94],
}


class StrategySearcher:
    """
    Given the model config and system config, search the best strategy
    """

    def __init__(self, model_config, system_config):
        self.model_config = model_config
        self.system_config = system_config
        self.strategies = []

    def _comp_cand_parallel_size(self, params: dict):
        """
        acccording to the tp size && world size to compute the candidate parallel size
        """
        tp_size = params["tp_size"]
        world_size = params["world_size"]
        assert world_size % tp_size == 0, "world size should be divisible by tp size"

        layer_num = self.model_config.layer_num
        expert_num = self.model_config.expert_num
        num_per_node = self.system_config.num_per_node

        candidate_list = []
        for pp_size in range(1, world_size // tp_size + 1):
            if layer_num % pp_size != 0:
                continue
            if (world_size // tp_size) % pp_size != 0:
                continue
            if expert_num == 1:
                candidate_list.append({"pp_size": pp_size, "ep_size": 1, "etp_size": 1})
                continue
            etp_size = 1
            while etp_size <= num_per_node:
                for ep_size in range(1, expert_num + 1):
                    if expert_num % ep_size != 0:
                        continue
                    if (world_size // pp_size) % etp_size != 0:
                        continue
                    if (world_size // pp_size) % ep_size != 0:
                        continue
                    candidate_list.append(
                        {"pp_size": pp_size, "ep_size": ep_size, "etp_size": etp_size}
                    )
                etp_size *= 2

        res = []
        for candidate in candidate_list:
            cur_param = deepcopy(params)
            cur_param.update(candidate)
            res.append(cur_param)
        return res

    def generate_grid(self, candidate_dict: dict):
        """
        Given the candidate dict, generate all the possible combinations
        """
        combinations = [
            dict(zip(candidate_dict.keys(), params))
            for params in itertools.product(*candidate_dict.values())
        ]

        final_combinations = []
        for params in combinations:
            for params_with_full_shard in self._comp_cand_parallel_size(params):
                layers = (
                    self.model_config.layer_num // params_with_full_shard["pp_size"]
                )
                stride = -1
                # The number of recomputed layers can be approximated by bucketing
                bucket_num = 4
                if layers // bucket_num > 1:
                    stride = -layers // bucket_num
                if params["enable_recompute"]:
                    recompute_strategy_list = [
                        {
                            **deepcopy(params_with_full_shard),
                            "recompute_layer_num": recompute_layer_num,
                        }
                        for recompute_layer_num in range(layers, 0, stride)
                    ]
                    final_combinations.extend(recompute_strategy_list)

                else:
                    final_combinations.append(params_with_full_shard)

        return final_combinations

    def _select_net(self, strategy: StrategyConfig):
        # default, we follow tp-dp-pp/etp-ep-edp-pp order to build the network
        num_per_node = self.system_config.num_per_node
        dense_mesh_order = ["tp", "dp", "pp"]
        dense_mesh_size = strategy.get_mesh_size("-".join(dense_mesh_order))
        moe_mesh_order = ["etp", "ep", "edp", "pp"]
        moe_mesh_size = strategy.get_mesh_size("-".join(moe_mesh_order))

        def select_impl(name: str, mesh_order: list, mesh_size: list):
            index = mesh_order.index(name)
            if reduce(operator.mul, mesh_size[: index + 1]) <= num_per_node:
                return "high_intra_node"
            return "inter_node"

        strategy.tp_net = select_impl("tp", dense_mesh_order, dense_mesh_size)
        strategy.dp_net = select_impl("dp", dense_mesh_order, dense_mesh_size)
        strategy.pp_net = select_impl("pp", dense_mesh_order, dense_mesh_size)
        strategy.etp_net = select_impl("etp", moe_mesh_order, moe_mesh_size)
        strategy.ep_net = select_impl("ep", moe_mesh_order, moe_mesh_size)

    def _prune_strategy(self):
        raise NotImplementedError

    def search(self, topk=5, candidate_strategy=None):
        """
        Search the best strategy
        """
        if candidate_strategy is None:
            candidate_strategy = CANDIDATE_STRATEGY
        self.strategies = []
        # grid search
        combinations = self.generate_grid(candidate_strategy)

        for params in tqdm(combinations):

            try:
                # convert to StrategyConfig
                cur_candidate_strategy = StrategyConfig(**params)
                self._select_net(cur_candidate_strategy)
                perf_model = PerfLLM()
                perf_model.configure(
                    strategy_config=cur_candidate_strategy,
                    model_config=self.model_config,
                    system_config=self.system_config,
                )
                # choose max batch size for now
                max_micro_batch_size = perf_model.search_max_micro_batch_size()
                if max_micro_batch_size < 1:
                    print(
                        f"Error: memory is not enough, skip this strategy {cur_candidate_strategy.to_dict()}"  # pylint: disable=line-too-long
                    )
                    # TODO: prune some invalid strategies
                    continue

                cur_candidate_strategy.micro_batch_size = max_micro_batch_size
                # when we don't consider batch size constraint,
                # we set micro_batch_num to a large number to avoid the overhead of bubble
                # but it is not reasonable, need to consider the global batch size constraint here
                if cur_candidate_strategy.pp_size > 1:
                    cur_candidate_strategy.micro_batch_num = 128
                perf_model._set_strategy_config(cur_candidate_strategy)
                perf_model._cross_sanity_check()

                perf_model.run_estimate()

                cost_result = perf_model.analysis_cost()

                memory_result = perf_model.analysis_mem()

                first_stage_mem_result = (
                    memory_result
                    if memory_result.get("first_stage") is None
                    else memory_result.get("first_stage")
                )

                self.strategies.append(
                    (
                        cur_candidate_strategy,
                        cost_result.get("mfu"),
                        cost_result.get("throughput_per_accelerator"),
                        cost_result.get("breakdown_result"),
                        first_stage_mem_result.get("peak_cached_mem"),
                    )
                )
            except AssertionError as err:
                print(
                    f"Error: {err}, skip this strategy {cur_candidate_strategy.to_dict()}"
                )

                continue

        self.strategies.sort(key=lambda x: x[1], reverse=True)
        top_k_strategies = self._get_top_k_strategies(k=topk)
        return top_k_strategies

    def _get_top_k_strategies(self, k):
        return self.strategies[: min(k, len(self.strategies))]
