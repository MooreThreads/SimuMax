"""Pipeline schedule and optimizer simulator helpers."""

from copy import deepcopy

from simumax.core.base_struct import (
    MetaModule,
    all_gather,
    reduce_scatter,
    all_reduce,
    AtomModel,
    FwdQue,
    send_next,
    send_prev,
    recv_next,
    recv_prev,
    async_send_next,
    async_send_prev,
    async_recv_next,
    async_recv_prev,
    async_wait_recv_next,
    async_wait_recv_prev,
)
from simumax.core.config import StrategyConfig
from simumax.core.utils import (
    format_scope_microbatch_tag,
    get_pp_p2p_comm_size,
    get_rank_group,
)

class OptimizerSimulator(MetaModule):

    def __init__(self, perf_model=None, model_name=None) -> None:
        model_base = perf_model.model_chunk_dict[model_name]
        strategy, system, model_info = model_base.strategy, model_base.system, model_base.get_model_info()
        super().__init__(strategy, system)
        self.model_info = model_info
        self.perf_model = perf_model
        self.model_name = model_name

    def prefill(self, args, call_stk='', com_buff=None):
        self.call_stk = f"rank{args.rank}-{format_scope_microbatch_tag(args)}{call_stk}{self.call_stk}"
        state = args.thread_state
        rank_info = get_rank_group(args.rank, self.strategy)
        comm_info = self.perf_model._compute_dp_time(self.model_name)
        opt_info = self.perf_model._compute_optim_time(self.model_name)
        
        if self.strategy.zero_state >= 1:
            # optimizer_group_size = self.strategy.dp_size * self.strategy.cp_size, set comm_num accordingly
            cost_dense, cost_moe = comm_info['dense'], comm_info['moe']
            
            cost_dense_rs =  cost_dense['details']['reduce_scatter_time']
            self.layers.append(reduce_scatter(f"{state.comm_order}-dp_cp_group:{rank_info['dp_cp_group_id']}", 
                                rank_info['dp_cp_rank'], self.strategy.dp_size * self.strategy.cp_size,  com_buff=com_buff,
                                fwd_cost=cost_dense_rs, global_rank=args.rank))
            state.comm_order += 1
            
            cost_moe_rs =  cost_moe['details']['reduce_scatter_time']
            self.layers.append(reduce_scatter(f"{state.comm_order}-edp_group:{rank_info['edp_group_id']}", 
                                rank_info['edp_rank'], self.strategy.edp_size,  com_buff=com_buff,
                                fwd_cost=cost_moe_rs, global_rank=args.rank))
            state.comm_order += 1            
            
            
            #sync whole wolrd in rerun_state_machine
            self.layers.append(all_reduce(f"default_group-pp_size:{self.strategy.pp_size}", 
                                args.rank, self.strategy.world_size,  com_buff=com_buff,
                                fwd_cost=1, global_rank=args.rank))                  

            self.layers.append(AtomModel(fwd_cost=opt_info['optim_time'], bwd_cost=0, specific_name='optimizer_step'))

            cost_dense_ag =  cost_dense['details']['all_gather_time']
            self.layers.append(all_gather(f"{state.comm_order}-dp_cp_group:{rank_info['dp_cp_group_id']}", 
                                rank_info['dp_cp_rank'], self.strategy.dp_size * self.strategy.cp_size,  com_buff=com_buff,
                                fwd_cost=cost_dense_ag, global_rank=args.rank))
            state.comm_order += 1

            cost_moe_ag =  cost_moe['details']['all_gather_time']
            self.layers.append(all_gather(f"{state.comm_order}-edp_group:{rank_info['edp_group_id']}", 
                                rank_info['edp_rank'], self.strategy.edp_size,  com_buff=com_buff,
                                fwd_cost=cost_moe_ag, global_rank=args.rank))
            state.comm_order += 1

        else:
            raise "not support now"

        for layer in self.layers:
            layer.prefill(args, self.call_stk, com_buff=com_buff)

class PpSchedule(MetaModule):
    """normal mlp layers"""
    def __init__(self, strategy:StrategyConfig, system, model) -> None:
        super().__init__(strategy, system)
        self.models = model if isinstance(model, list) else [model]
        self.model = self.models[0]
        self.vp_size = max(1, len(self.models))

    def _prefill_batch_interleaved(self, args, com_buff=None):
        job = []
        rank_info = get_rank_group(args.rank, self.strategy)
        pp_size = self.strategy.pp_size
        pp_rank = rank_info['pp_rank']
        pp_group = rank_info['pp_group_id']
        if pp_size <= 1:
            raise NotImplementedError(
                "Strict interleaved simu schedule requires pp_size > 1."
            )
        pp_comm_size = get_pp_p2p_comm_size(
            self.strategy,
            self.model.model_config.hidden_size,
            self.dtype_to_element_size[self.strategy.dtype],
        )
        pp_cost = self.system.compute_net_op_time(
            "p2p", pp_comm_size, 2, net=self.strategy.pp_net
        )

        def _make_model(chunk_idx: int, real_mb: int, mb_virtual: int):
            model = deepcopy(self.models[chunk_idx])
            args.microbatch = real_mb
            args.chunk_idx = chunk_idx
            args.virtual_microbatch = mb_virtual
            model.prefill(args, call_stk=f"-chunk{chunk_idx}-", com_buff=com_buff)
            return model

        total_virtual_stages = self.vp_size * pp_size
        total_virtual_microbatches = self.strategy.micro_batch_num * self.vp_size
        group_size_per_vp_stage = getattr(
            self.strategy, "microbatch_group_size_per_vp_stage", None
        )
        if group_size_per_vp_stage is None:
            group_size_per_vp_stage = pp_size
        # Megatron interleaved warmup formula.
        num_warmup_microbatches = (pp_size - pp_rank - 1) * 2 + (
            self.vp_size - 1
        ) * group_size_per_vp_stage
        num_warmup_microbatches = min(num_warmup_microbatches, total_virtual_microbatches)
        num_microbatches_remaining = total_virtual_microbatches - num_warmup_microbatches

        schedule_table = []
        num_microbatches = self.strategy.micro_batch_num
        for min_mb in range(0, num_microbatches, group_size_per_vp_stage):
            max_mb = min(num_microbatches, min_mb + group_size_per_vp_stage)
            for chunk_idx in range(self.vp_size):
                for mb in range(min_mb, max_mb):
                    schedule_table.append((mb, chunk_idx))

        def _schedule_table_entry(virtual_k: int):
            return schedule_table[virtual_k]

        use_async_pp_comm = getattr(self.strategy, "pp_comm_async", True)
        if use_async_pp_comm and self.strategy.micro_batch_num < (pp_size * self.vp_size):
            raise RuntimeError(
                "async VPP does not support micro_batch_num < pp_size * vp_size yet"
            )
        prefetched_fwd_recv_keys = set()
        prefetched_bwd_recv_keys = set()

        def _get_model_chunk_id(virtual_k: int, forward: bool) -> int:
            chunk = schedule_table[virtual_k % total_virtual_microbatches][1]
            if not forward:
                chunk = self.vp_size - chunk - 1
            return chunk

        def _recv_tensor_from_previous_stage(virtual_k: int, forward: bool):
            recv = True
            is_leading = (pp_rank == 0) if forward else (pp_rank == pp_size - 1)
            last_chunk = (self.vp_size - 1) if forward else 0
            if is_leading:
                if virtual_k < (pp_size - 1):
                    recv = False
                    next_chunk = _get_model_chunk_id(virtual_k + 1, forward)
                else:
                    next_chunk = _get_model_chunk_id(virtual_k - (pp_size - 1), forward)
                if next_chunk == last_chunk:
                    recv = False
                next_chunk = next_chunk + 1 if forward else next_chunk - 1
            else:
                next_chunk = _get_model_chunk_id(virtual_k + 1, forward)
            return recv, next_chunk

        def _make_async_recv_prev(real_mb: int, virtual_idx: int):
            return async_recv_prev(
                id=f'forward-v{virtual_idx}-mb{real_mb}-pp_group:{pp_group}-',
                rank=pp_rank,
                pp_size=pp_size,
                fwd_cost=pp_cost,
                global_rank=args.rank,
                call_stk=f"rank{args.rank}",
            )

        def _make_async_recv_next(real_mb: int, virtual_idx: int):
            return async_recv_next(
                id=f'backward-v{virtual_idx}-mb{real_mb}-pp_group:{pp_group}-',
                rank=pp_rank,
                pp_size=pp_size,
                fwd_cost=pp_cost,
                global_rank=args.rank,
                call_stk=f"rank{args.rank}",
            )

        def _append_async_bundle(
            *,
            send_next_spec=None,
            send_prev_spec=None,
            recv_prev_spec=None,
            recv_next_spec=None,
        ):
            ops = []

            def _mk_send_next(spec):
                if spec is None:
                    return None
                real_mb, virtual_idx = spec
                return async_send_next(
                    id=f'forward-v{virtual_idx + 1}-mb{real_mb}-pp_group:{pp_group}-',
                    rank=pp_rank,
                    pp_size=pp_size,
                    fwd_cost=pp_cost,
                    global_rank=args.rank,
                    call_stk=f"rank{args.rank}",
                )

            def _mk_send_prev(spec):
                if spec is None:
                    return None
                real_mb, virtual_idx = spec
                return async_send_prev(
                    id=f'backward-v{virtual_idx - 1}-mb{real_mb}-pp_group:{pp_group}-',
                    rank=pp_rank,
                    pp_size=pp_size,
                    fwd_cost=pp_cost,
                    global_rank=args.rank,
                    call_stk=f"rank{args.rank}",
                )

            def _mk_recv_prev(spec):
                if spec is None:
                    return None
                real_mb, virtual_idx = spec
                key = ("fwd", real_mb, virtual_idx)
                if key in prefetched_fwd_recv_keys:
                    return None
                prefetched_fwd_recv_keys.add(key)
                return _make_async_recv_prev(real_mb, virtual_idx)

            def _mk_recv_next(spec):
                if spec is None:
                    return None
                real_mb, virtual_idx = spec
                key = ("bwd", real_mb, virtual_idx)
                if key in prefetched_bwd_recv_keys:
                    return None
                prefetched_bwd_recv_keys.add(key)
                return _make_async_recv_next(real_mb, virtual_idx)

            recv_prev_op = _mk_recv_prev(recv_prev_spec)
            send_next_op = _mk_send_next(send_next_spec)
            recv_next_op = _mk_recv_next(recv_next_spec)
            send_prev_op = _mk_send_prev(send_prev_spec)

            if pp_rank % 2 == 0:
                ordered = [send_next_op, recv_prev_op, send_prev_op, recv_next_op]
            else:
                ordered = [recv_prev_op, send_next_op, recv_next_op, send_prev_op]
            ops = [op for op in ordered if op is not None]
            if ops:
                job.append(FwdQue(que=ops))

        def _append_fwd_compute(virtual_k: int, need_recv_prev: bool):
            real_mb, chunk_idx = _schedule_table_entry(virtual_k)
            virtual_idx = chunk_idx * pp_size + pp_rank
            mb_virtual = real_mb * self.vp_size + chunk_idx

            if virtual_idx > 0 and need_recv_prev:
                recv_op = async_wait_recv_prev(
                    id=f'forward-v{virtual_idx}-mb{real_mb}-pp_group:{pp_group}-',
                    rank=pp_rank,
                    pp_size=pp_size,
                    fwd_cost=pp_cost,
                    global_rank=args.rank,
                    call_stk=f"rank{args.rank}",
                )
                job.append(FwdQue(que=[recv_op]))

            model = _make_model(chunk_idx, real_mb, mb_virtual)
            job.append(model.prefill_fwd())

        def _append_bwd_compute(virtual_k: int, need_recv_next: bool):
            real_mb, fwd_chunk_idx = _schedule_table_entry(virtual_k)
            chunk_idx = self.vp_size - 1 - fwd_chunk_idx
            virtual_idx = chunk_idx * pp_size + pp_rank
            mb_virtual = real_mb * self.vp_size + chunk_idx

            if virtual_idx < total_virtual_stages - 1 and need_recv_next:
                recv_op = async_wait_recv_next(
                    id=f'backward-v{virtual_idx}-mb{real_mb}-pp_group:{pp_group}-',
                    rank=pp_rank,
                    pp_size=pp_size,
                    fwd_cost=pp_cost,
                    global_rank=args.rank,
                    call_stk=f"rank{args.rank}",
                )
                job.append(FwdQue(que=[recv_op]))

            model = _make_model(chunk_idx, real_mb, mb_virtual)
            job.append(model.prefill_bwd())

        def _fwd_ref(virtual_k: int):
            real_mb, chunk_idx = _schedule_table_entry(virtual_k)
            virtual_idx = chunk_idx * pp_size + pp_rank
            mb_virtual = real_mb * self.vp_size + chunk_idx
            return real_mb, chunk_idx, virtual_idx, mb_virtual

        def _bwd_ref(virtual_k: int):
            real_mb, fwd_chunk_idx = _schedule_table_entry(virtual_k)
            chunk_idx = self.vp_size - 1 - fwd_chunk_idx
            virtual_idx = chunk_idx * pp_size + pp_rank
            mb_virtual = real_mb * self.vp_size + chunk_idx
            return real_mb, chunk_idx, virtual_idx, mb_virtual

        def _append_blocking_comms(
            *,
            send_prev_op=None,
            recv_prev_op=None,
            send_next_op=None,
            recv_next_op=None,
        ):
            # Mirror Megatron blocking VPP with batch_isend_irecv: only non-None
            # ops are issued, and the local submission order is
            # [send_prev, recv_prev, send_next, recv_next].
            ordered = [
                op
                for op in (send_prev_op, recv_prev_op, send_next_op, recv_next_op)
                if op is not None
            ]
            if ordered:
                job.append(
                    FwdQue(
                        call_stk=f"rank{args.rank}-batch_pp_comm",
                        que=ordered,
                        batch_blocking_comm=True,
                    )
                )

        if not use_async_pp_comm:
            # Blocking interleaved path. Keep communications in paired queues
            # to avoid cyclic waits under multi-rank PP.
            if pp_rank != 0:
                real_mb0, _, virtual_idx0, _ = _fwd_ref(0)
                if virtual_idx0 > 0:
                    job.append(
                        FwdQue(
                            que=[
                                recv_prev(
                                    id=f'forward-v{virtual_idx0}-mb{real_mb0}-pp_group:{pp_group}-',
                                    rank=pp_rank,
                                    pp_size=pp_size,
                                    com_buff=com_buff,
                                    fwd_cost=pp_cost,
                                    global_rank=args.rank,
                                    call_stk=f"rank{args.rank}",
                                )
                            ]
                        )
                    )

            need_recv_fwd = pp_rank != 0
            need_recv_bwd = False

            # warmup forward
            for k in range(num_warmup_microbatches):
                real_mb, chunk_idx, virtual_idx, mb_virtual = _fwd_ref(k)
                model = _make_model(chunk_idx, real_mb, mb_virtual)
                job.append(model.prefill_fwd())

                need_recv_fwd_next, _ = _recv_tensor_from_previous_stage(k, True)
                if k == total_virtual_microbatches - 1:
                    need_recv_fwd_next = False
                if k == (num_warmup_microbatches - 1) and num_microbatches_remaining > 0:
                    need_recv_bwd = (pp_rank != (pp_size - 1))

                send_next_op = None
                recv_prev_op = None
                recv_next_op = None
                if virtual_idx < total_virtual_stages - 1:
                    send_next_op = send_next(
                        id=f'forward-v{virtual_idx + 1}-mb{real_mb}-pp_group:{pp_group}-',
                        rank=pp_rank,
                        pp_size=pp_size,
                        com_buff=com_buff,
                        fwd_cost=pp_cost,
                        global_rank=args.rank,
                        call_stk=f"rank{args.rank}",
                    )
                if (k + 1) < total_virtual_microbatches and need_recv_fwd_next:
                    next_real_mb, _, next_virtual_idx, _ = _fwd_ref(k + 1)
                    if next_virtual_idx > 0:
                        recv_prev_op = recv_prev(
                            id=f'forward-v{next_virtual_idx}-mb{next_real_mb}-pp_group:{pp_group}-',
                            rank=pp_rank,
                            pp_size=pp_size,
                            com_buff=com_buff,
                            fwd_cost=pp_cost,
                            global_rank=args.rank,
                            call_stk=f"rank{args.rank}",
                        )
                elif (
                    num_microbatches_remaining == 0
                    and pp_rank == 0
                    and (k + 1) < total_virtual_microbatches
                ):
                    next_real_mb, _, next_virtual_idx, _ = _fwd_ref(k + 1)
                    if next_virtual_idx > 0:
                        recv_prev_op = recv_prev(
                            id=f'forward-v{next_virtual_idx}-mb{next_real_mb}-pp_group:{pp_group}-',
                            rank=pp_rank,
                            pp_size=pp_size,
                            com_buff=com_buff,
                            fwd_cost=pp_cost,
                            global_rank=args.rank,
                            call_stk=f"rank{args.rank}",
                        )
                if k == (num_warmup_microbatches - 1) and num_microbatches_remaining > 0 and need_recv_bwd:
                    b_real_mb0, _, b_virtual_idx0, _ = _bwd_ref(0)
                    if b_virtual_idx0 < total_virtual_stages - 1:
                        recv_next_op = recv_next(
                            id=f'backward-v{b_virtual_idx0}-mb{b_real_mb0}-pp_group:{pp_group}-',
                            rank=pp_rank,
                            pp_size=pp_size,
                            com_buff=com_buff,
                            fwd_cost=pp_cost,
                            global_rank=args.rank,
                            call_stk=f"rank{args.rank}",
                        )
                _append_blocking_comms(
                    recv_prev_op=recv_prev_op,
                    send_next_op=send_next_op,
                    recv_next_op=recv_next_op,
                )
                need_recv_fwd = need_recv_fwd_next

            # If warmup consumed all virtual microbatches, there is no steady
            # 1F1B iteration to inject the first backward recv. Prime it here
            # before cooldown so the first backward compute consumes mb0
            # instead of incorrectly skipping to the next backward token.
            if num_microbatches_remaining == 0 and pp_rank != (pp_size - 1):
                b_real_mb0, _, b_virtual_idx0, _ = _bwd_ref(0)
                if b_virtual_idx0 < total_virtual_stages - 1:
                    job.append(
                        FwdQue(
                            que=[
                                recv_next(
                                    id=f'backward-v{b_virtual_idx0}-mb{b_real_mb0}-pp_group:{pp_group}-',
                                    rank=pp_rank,
                                    pp_size=pp_size,
                                    com_buff=com_buff,
                                    fwd_cost=pp_cost,
                                    global_rank=args.rank,
                                    call_stk=f"rank{args.rank}",
                                )
                            ]
                        )
                    )

            for k in range(num_microbatches_remaining):
                forward_k = k + num_warmup_microbatches
                backward_k = k

                f_real_mb, f_chunk_idx, f_virtual_idx, f_mb_virtual = _fwd_ref(forward_k)
                f_model = _make_model(f_chunk_idx, f_real_mb, f_mb_virtual)
                job.append(f_model.prefill_fwd())

                b_real_mb, b_chunk_idx, b_virtual_idx, b_mb_virtual = _bwd_ref(backward_k)
                b_model = _make_model(b_chunk_idx, b_real_mb, b_mb_virtual)
                job.append(b_model.prefill_bwd())

                need_recv_fwd_next, _ = _recv_tensor_from_previous_stage(forward_k, True)
                need_recv_bwd_next, _ = _recv_tensor_from_previous_stage(backward_k, False)
                if k == (num_microbatches_remaining - 1):
                    need_recv_fwd_next = False

                send_next_op = None
                send_prev_op = None
                recv_prev_op = None
                recv_next_op = None
                if f_virtual_idx < total_virtual_stages - 1:
                    send_next_op = send_next(
                        id=f'forward-v{f_virtual_idx + 1}-mb{f_real_mb}-pp_group:{pp_group}-',
                        rank=pp_rank,
                        pp_size=pp_size,
                        com_buff=com_buff,
                        fwd_cost=pp_cost,
                        global_rank=args.rank,
                        call_stk=f"rank{args.rank}",
                    )
                if b_virtual_idx > 0:
                    send_prev_op = send_prev(
                        id=f'backward-v{b_virtual_idx - 1}-mb{b_real_mb}-pp_group:{pp_group}-',
                        rank=pp_rank,
                        pp_size=pp_size,
                        com_buff=com_buff,
                        fwd_cost=pp_cost,
                        global_rank=args.rank,
                        call_stk=f"rank{args.rank}",
                    )
                if (forward_k + 1) < total_virtual_microbatches and need_recv_fwd_next:
                    nf_real_mb, _, nf_virtual_idx, _ = _fwd_ref(forward_k + 1)
                    if nf_virtual_idx > 0:
                        recv_prev_op = recv_prev(
                            id=f'forward-v{nf_virtual_idx}-mb{nf_real_mb}-pp_group:{pp_group}-',
                            rank=pp_rank,
                            pp_size=pp_size,
                            com_buff=com_buff,
                            fwd_cost=pp_cost,
                            global_rank=args.rank,
                            call_stk=f"rank{args.rank}",
                        )
                if (backward_k + 1) < total_virtual_microbatches and need_recv_bwd_next:
                    nb_real_mb, _, nb_virtual_idx, _ = _bwd_ref(backward_k + 1)
                    if nb_virtual_idx < total_virtual_stages - 1:
                        recv_next_op = recv_next(
                            id=f'backward-v{nb_virtual_idx}-mb{nb_real_mb}-pp_group:{pp_group}-',
                            rank=pp_rank,
                            pp_size=pp_size,
                            com_buff=com_buff,
                            fwd_cost=pp_cost,
                            global_rank=args.rank,
                            call_stk=f"rank{args.rank}",
                        )
                _append_blocking_comms(
                    send_prev_op=send_prev_op,
                    recv_prev_op=recv_prev_op,
                    send_next_op=send_next_op,
                    recv_next_op=recv_next_op,
                )
                need_recv_fwd = need_recv_fwd_next
                need_recv_bwd = need_recv_bwd_next

            for k in range(num_microbatches_remaining, total_virtual_microbatches):
                b_real_mb, b_chunk_idx, b_virtual_idx, b_mb_virtual = _bwd_ref(k)
                b_model = _make_model(b_chunk_idx, b_real_mb, b_mb_virtual)
                job.append(b_model.prefill_bwd())

                need_recv_bwd_next, _ = _recv_tensor_from_previous_stage(k, False)
                if k == (total_virtual_microbatches - 1):
                    need_recv_bwd_next = False

                send_prev_op = None
                recv_next_op = None
                if b_virtual_idx > 0:
                    send_prev_op = send_prev(
                        id=f'backward-v{b_virtual_idx - 1}-mb{b_real_mb}-pp_group:{pp_group}-',
                        rank=pp_rank,
                        pp_size=pp_size,
                        com_buff=com_buff,
                        fwd_cost=pp_cost,
                        global_rank=args.rank,
                        call_stk=f"rank{args.rank}",
                    )
                if (k + 1) < total_virtual_microbatches and need_recv_bwd_next:
                    nb_real_mb, _, nb_virtual_idx, _ = _bwd_ref(k + 1)
                    if nb_virtual_idx < total_virtual_stages - 1:
                        recv_next_op = recv_next(
                            id=f'backward-v{nb_virtual_idx}-mb{nb_real_mb}-pp_group:{pp_group}-',
                            rank=pp_rank,
                            pp_size=pp_size,
                            com_buff=com_buff,
                            fwd_cost=pp_cost,
                            global_rank=args.rank,
                            call_stk=f"rank{args.rank}",
                        )
                elif (
                    num_microbatches_remaining == 0
                    and pp_rank == (pp_size - 1)
                    and (k + 1) < total_virtual_microbatches
                ):
                    nb_real_mb, _, nb_virtual_idx, _ = _bwd_ref(k + 1)
                    if nb_virtual_idx < total_virtual_stages - 1:
                        recv_next_op = recv_next(
                            id=f'backward-v{nb_virtual_idx}-mb{nb_real_mb}-pp_group:{pp_group}-',
                            rank=pp_rank,
                            pp_size=pp_size,
                            com_buff=com_buff,
                            fwd_cost=pp_cost,
                            global_rank=args.rank,
                            call_stk=f"rank{args.rank}",
                        )
                _append_blocking_comms(
                    send_prev_op=send_prev_op,
                    recv_next_op=recv_next_op,
                )
                need_recv_bwd = need_recv_bwd_next
            return job

        # Async interleaved path: mirror Megatron batched P2P schedule semantics.
        if use_async_pp_comm:
            if pp_rank != 0:
                real_mb0, _, virtual_idx0, _ = _fwd_ref(0)
                if virtual_idx0 > 0:
                    job.append(
                        FwdQue(
                            que=[
                                async_wait_recv_prev(
                                    id=f'forward-v{virtual_idx0}-mb{real_mb0}-pp_group:{pp_group}-',
                                    rank=pp_rank,
                                    pp_size=pp_size,
                                    fwd_cost=pp_cost,
                                    global_rank=args.rank,
                                    call_stk=f"rank{args.rank}",
                                )
                            ]
                        )
                    )

            need_recv_fwd = pp_rank != 0
            need_recv_bwd = False

            for k in range(num_warmup_microbatches):
                real_mb, _, virtual_idx, _ = _fwd_ref(k)
                _append_fwd_compute(k, need_recv_prev=need_recv_fwd)

                need_recv_fwd_next, _ = _recv_tensor_from_previous_stage(k, True)
                if k == total_virtual_microbatches - 1:
                    need_recv_fwd_next = False
                last_warmup_special = (
                    k == (num_warmup_microbatches - 1) and num_microbatches_remaining > 0
                )
                recv_next_spec = None
                if last_warmup_special:
                    need_recv_bwd = (pp_rank != (pp_size - 1))
                    if need_recv_bwd:
                        b_real_mb0, _, b_virtual_idx0, _ = _bwd_ref(0)
                        if b_virtual_idx0 < total_virtual_stages - 1:
                            recv_next_spec = (b_real_mb0, b_virtual_idx0)
                recv_prev_spec = None
                if (k + 1) < total_virtual_microbatches and need_recv_fwd_next:
                    nf_real_mb, _, nf_virtual_idx, _ = _fwd_ref(k + 1)
                    if nf_virtual_idx > 0:
                        recv_prev_spec = (nf_real_mb, nf_virtual_idx)

                send_next_spec = None
                if virtual_idx < total_virtual_stages - 1:
                    send_next_spec = (real_mb, virtual_idx)
                _append_async_bundle(
                    send_next_spec=send_next_spec,
                    recv_prev_spec=recv_prev_spec,
                    recv_next_spec=recv_next_spec,
                )
                need_recv_fwd = need_recv_fwd_next

            for k in range(num_microbatches_remaining):
                forward_k = k + num_warmup_microbatches
                backward_k = k

                f_real_mb, _, f_virtual_idx, _ = _fwd_ref(forward_k)
                b_real_mb, _, b_virtual_idx, _ = _bwd_ref(backward_k)

                _append_fwd_compute(forward_k, need_recv_prev=need_recv_fwd)
                _append_bwd_compute(backward_k, need_recv_next=need_recv_bwd)

                need_recv_fwd_next, _ = _recv_tensor_from_previous_stage(forward_k, True)
                need_recv_bwd_next, _ = _recv_tensor_from_previous_stage(backward_k, False)
                if k == (num_microbatches_remaining - 1):
                    need_recv_fwd_next = False

                send_next_spec = None
                if f_virtual_idx < total_virtual_stages - 1:
                    send_next_spec = (f_real_mb, f_virtual_idx)
                send_prev_spec = None
                if b_virtual_idx > 0:
                    send_prev_spec = (b_real_mb, b_virtual_idx)

                recv_prev_spec = None
                if (forward_k + 1) < total_virtual_microbatches and need_recv_fwd_next:
                    nf_real_mb, _, nf_virtual_idx, _ = _fwd_ref(forward_k + 1)
                    if nf_virtual_idx > 0:
                        recv_prev_spec = (nf_real_mb, nf_virtual_idx)

                recv_next_spec = None
                if (backward_k + 1) < total_virtual_microbatches and need_recv_bwd_next:
                    nb_real_mb, _, nb_virtual_idx, _ = _bwd_ref(backward_k + 1)
                    if nb_virtual_idx < total_virtual_stages - 1:
                        recv_next_spec = (nb_real_mb, nb_virtual_idx)

                _append_async_bundle(
                    send_next_spec=send_next_spec,
                    send_prev_spec=send_prev_spec,
                    recv_prev_spec=recv_prev_spec,
                    recv_next_spec=recv_next_spec,
                )
                need_recv_fwd = need_recv_fwd_next
                need_recv_bwd = need_recv_bwd_next

            for k in range(num_microbatches_remaining, total_virtual_microbatches):
                b_real_mb, _, b_virtual_idx, _ = _bwd_ref(k)
                _append_bwd_compute(k, need_recv_next=need_recv_bwd)

                need_recv_bwd_next, _ = _recv_tensor_from_previous_stage(k, False)
                if k == (total_virtual_microbatches - 1):
                    need_recv_bwd_next = False

                send_prev_spec = None
                if b_virtual_idx > 0:
                    send_prev_spec = (b_real_mb, b_virtual_idx)

                recv_next_spec = None
                if (k + 1) < total_virtual_microbatches and need_recv_bwd_next:
                    nb_real_mb, _, nb_virtual_idx, _ = _bwd_ref(k + 1)
                    if nb_virtual_idx < total_virtual_stages - 1:
                        recv_next_spec = (nb_real_mb, nb_virtual_idx)
                _append_async_bundle(
                    send_prev_spec=send_prev_spec,
                    recv_next_spec=recv_next_spec,
                )
                need_recv_bwd = need_recv_bwd_next
            return job

    def prefill_batch(self, args, com_buff=None):
        if self.vp_size > 1:
            return self._prefill_batch_interleaved(args, com_buff=com_buff)

        job = []
        rank_info = get_rank_group(args.rank, self.strategy)
        pp_size = self.strategy.pp_size
        pp_rank = rank_info['pp_rank']
        pp_group = rank_info['pp_group_id']
        pp_comm_size = get_pp_p2p_comm_size(
            self.strategy,
            self.model.model_config.hidden_size,
            self.dtype_to_element_size[self.strategy.dtype],
        )
        pp_cost = self.system.compute_net_op_time(
            "p2p", pp_comm_size, 2, net=self.strategy.pp_net
        )  # p2p
        use_async_pp_comm = getattr(self.strategy, "pp_comm_async", True)

        def _append_wait_recv_prev_forward(fwd_idx):
            if pp_rank == 0:
                return
            if use_async_pp_comm:
                comm = async_wait_recv_prev(
                    id=f'forward-{fwd_idx}-pp_group:{pp_group}-',
                    rank=pp_rank,
                    pp_size=pp_size,
                    fwd_cost=pp_cost,
                    global_rank=args.rank,
                    call_stk=f"rank{args.rank}",
                )
            else:
                comm = recv_prev(
                    id=f'forward-{fwd_idx}-pp_group:{pp_group}-',
                    rank=pp_rank,
                    com_buff=com_buff,
                    fwd_cost=pp_cost,
                    pp_size=pp_size,
                    global_rank=args.rank,
                    call_stk=f"rank{args.rank}",
                )
            job.append(FwdQue(que=[comm]))

        def _append_post_recv_prev_forward(fwd_idx):
            if pp_rank == 0 or not use_async_pp_comm:
                return
            comm = async_recv_prev(
                id=f'forward-{fwd_idx}-pp_group:{pp_group}-',
                rank=pp_rank,
                pp_size=pp_size,
                fwd_cost=pp_cost,
                global_rank=args.rank,
                call_stk=f"rank{args.rank}",
            )
            job.append(FwdQue(que=[comm]))

        def _append_send_next_forward(fwd_idx):
            if pp_rank == pp_size - 1:
                return
            if use_async_pp_comm:
                comm = async_send_next(
                    id=f'forward-{fwd_idx}-pp_group:{pp_group}-',
                    rank=pp_rank,
                    pp_size=pp_size,
                    fwd_cost=pp_cost,
                    global_rank=args.rank,
                    call_stk=f"rank{args.rank}",
                )
            else:
                comm = send_next(
                    id=f'forward-{fwd_idx}-pp_group:{pp_group}-',
                    rank=pp_rank,
                    com_buff=com_buff,
                    fwd_cost=pp_cost,
                    pp_size=pp_size,
                    global_rank=args.rank,
                    call_stk=f"rank{args.rank}",
                )
            job.append(FwdQue(que=[comm]))

        def _append_wait_recv_next_backward(bwd_idx):
            if pp_rank == pp_size - 1:
                return
            if use_async_pp_comm:
                comm = async_wait_recv_next(
                    id=f'backward-{bwd_idx}-pp_group:{pp_group}-',
                    rank=pp_rank,
                    pp_size=pp_size,
                    fwd_cost=pp_cost,
                    global_rank=args.rank,
                    call_stk=f"rank{args.rank}",
                )
            else:
                comm = recv_next(
                    id=f'backward-{bwd_idx}-pp_group:{pp_group}-',
                    rank=pp_rank,
                    com_buff=com_buff,
                    fwd_cost=pp_cost,
                    pp_size=pp_size,
                    global_rank=args.rank,
                    call_stk=f"rank{args.rank}",
                )
            job.append(FwdQue(que=[comm]))

        def _append_post_recv_next_backward(bwd_idx):
            if pp_rank == pp_size - 1 or not use_async_pp_comm:
                return
            comm = async_recv_next(
                id=f'backward-{bwd_idx}-pp_group:{pp_group}-',
                rank=pp_rank,
                pp_size=pp_size,
                fwd_cost=pp_cost,
                global_rank=args.rank,
                call_stk=f"rank{args.rank}",
            )
            job.append(FwdQue(que=[comm]))

        def _append_send_prev_backward(bwd_idx):
            if pp_rank == 0:
                return
            if use_async_pp_comm:
                comm = async_send_prev(
                    id=f'backward-{bwd_idx}-pp_group:{pp_group}-',
                    rank=pp_rank,
                    pp_size=pp_size,
                    fwd_cost=pp_cost,
                    global_rank=args.rank,
                    call_stk=f"rank{args.rank}",
                )
            else:
                comm = send_prev(
                    id=f'backward-{bwd_idx}-pp_group:{pp_group}-',
                    rank=pp_rank,
                    com_buff=com_buff,
                    fwd_cost=pp_cost,
                    pp_size=pp_size,
                    global_rank=args.rank,
                    call_stk=f"rank{args.rank}",
                )
            job.append(FwdQue(que=[comm]))
        
        num_warmup_microbatches = (
            self.strategy.pp_size
            - rank_info['pp_rank']
            - 1
        )
        num_warmup_microbatches = min(num_warmup_microbatches, self.strategy.micro_batch_num)
        num_microbatches_remaining = self.strategy.micro_batch_num - num_warmup_microbatches
        fwd_queue = []
        fwd_idx = 0  #increase happened immediately after send forward, even no need to send in rank pp-1
        bwd_idx = 0  #increase happened immediately after send bwd, even no need to send in rank 0
        args.microbatch = 0
        for i in range(num_warmup_microbatches):
            if not pp_rank == 0:
                _append_wait_recv_prev_forward(fwd_idx)
            model = deepcopy(self.model)
            model.prefill(args, com_buff=com_buff)
            args.microbatch += 1
            job.append(model.prefill_fwd())
            fwd_queue.append(model)

            if not pp_rank == pp_size - 1:
                _append_send_next_forward(fwd_idx)
            if (
                use_async_pp_comm
                and i == (num_warmup_microbatches - 1)
                and num_microbatches_remaining > 0
                and pp_rank != pp_size - 1
            ):
                _append_post_recv_next_backward(bwd_idx)
            fwd_idx += 1
        for i in range(num_microbatches_remaining):
            last_iteration = i == (num_microbatches_remaining - 1)
            # In sync 1F1B, steady-state recv_prev for fwd_idx>0 is already bundled
            # with the previous iteration's send_prev/recv_prev pair.
            if not pp_rank == 0 and (use_async_pp_comm or i == 0):
                _append_wait_recv_prev_forward(fwd_idx)
            model = deepcopy(self.model)
            model.prefill(args, com_buff=com_buff)
            args.microbatch += 1
            job.append(model.prefill_fwd())
            fwd_queue.append(model)

            if not pp_rank == pp_size - 1:
                if use_async_pp_comm:
                    _append_send_next_forward(fwd_idx)
                    if not last_iteration:
                        _append_post_recv_next_backward(bwd_idx + 1)
                else:
                    que = []
                    comm1 = send_next(id=f'forward-{fwd_idx}-pp_group:{pp_group}-', rank=pp_rank, com_buff=com_buff, fwd_cost=pp_cost, pp_size=pp_size,global_rank=args.rank, call_stk=f"rank{args.rank}")
                    que.append(comm1)
                    comm2 = recv_next(id=f'backward-{bwd_idx}-pp_group:{pp_group}-', rank=pp_rank, com_buff=com_buff, fwd_cost=pp_cost, pp_size=pp_size, global_rank=args.rank, call_stk=f"rank{args.rank}")
                    que.append(comm2)
                    if pp_rank % 2:
                        comm_que = FwdQue(que=que)
                    else:
                        comm_que = FwdQue(que=que[::-1])
                    job.append(comm_que)
            fwd_idx += 1

            # In sync 1F1B, recv_next for the current backward step is already bundled
            # together with send_next in the steady-state communication pair above.
            if not pp_rank == pp_size - 1 and use_async_pp_comm:
                _append_wait_recv_next_backward(bwd_idx)
            model = fwd_queue.pop(0)
            job.append(model.prefill_bwd())


            if last_iteration:
                if not pp_rank == 0:
                    _append_send_prev_backward(bwd_idx)
                bwd_idx += 1
            else:
                if not pp_rank == 0:
                    if use_async_pp_comm:
                        _append_send_prev_backward(bwd_idx)
                        _append_post_recv_prev_forward(fwd_idx)
                    else:
                        que = []
                        comm1 = send_prev(id=f'backward-{bwd_idx}-pp_group:{pp_group}-', rank=pp_rank, com_buff=com_buff, fwd_cost=pp_cost, pp_size=pp_size, global_rank=args.rank, call_stk=f"rank{args.rank}")
                        que.append(comm1)
                        comm2 = recv_prev(id=f'forward-{fwd_idx}-pp_group:{pp_group}-', rank=pp_rank, com_buff=com_buff, fwd_cost=pp_cost, pp_size=pp_size, global_rank=args.rank, call_stk=f"rank{args.rank}")
                        que.append(comm2)
                        if pp_rank % 2:
                            comm_que = FwdQue(que=que)
                        else:
                            comm_que = FwdQue(que=que[::-1])
                        job.append(comm_que)
                bwd_idx += 1
                
        for i in range(num_warmup_microbatches):
            if not pp_rank == pp_size - 1:
                _append_wait_recv_next_backward(bwd_idx)

            model = fwd_queue.pop(0)
            job.append(model.prefill_bwd())

            if not pp_rank == 0:
                _append_send_prev_backward(bwd_idx)

            bwd_idx += 1
        return job
