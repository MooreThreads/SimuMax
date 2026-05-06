"""Simulator-side memory timeline helpers."""

from collections import defaultdict
from dataclasses import dataclass
import re


@dataclass
class OpMemoryProfile:
    """Runtime memory profile for one logical compute op in simulator replay."""

    op_name: str = ""
    fwd_peak_mem_no_cache: int = 0
    bwd_peak_mem_no_cache: int = 0
    recompute_peak_mem_no_cache: int = 0
    cache_size_bytes: int = 0
    cache_alloc_phase: str | None = None
    cache_release_phase: str | None = "bwd"
    cache_token_scope: str = ""

    def phase_peak_no_cache(self, phase: str) -> int:
        if phase == "fwd":
            return int(self.fwd_peak_mem_no_cache)
        if phase == "recompute_fwd":
            return int(self.recompute_peak_mem_no_cache)
        if phase == "bwd":
            return int(self.bwd_peak_mem_no_cache)
        raise ValueError(f"Unsupported phase: {phase}")

    def phase_allocates_cache(self, phase: str) -> bool:
        return bool(self.cache_size_bytes) and phase == self.cache_alloc_phase

    def phase_releases_cache(self, phase: str) -> bool:
        return bool(self.cache_size_bytes) and phase == self.cache_release_phase


class SimuMemoryTracker:
    """Rank-local allocated-memory tracker for simulator replay."""

    def __init__(self):
        self.static_bytes = defaultdict(int)
        self.cached_bytes = defaultdict(int)
        self.peak_bytes = defaultdict(int)
        self.events = []
        self.snapshots = []
        self._cache_token_id = 0
        self._live_cached_tokens = defaultdict(dict)
        self._cached_tokens_by_key = defaultdict(lambda: defaultdict(list))
        self.cache_token_events = []

    def _parse_scope_tags(self, scope: str):
        scope = scope or ""
        mb_match = re.search(r"microbatch(\d+)", scope)
        chunk_match = re.search(r"chunk(\d+)", scope)
        return {
            "scope": scope,
            "microbatch": int(mb_match.group(1)) if mb_match else None,
            "chunk": int(chunk_match.group(1)) if chunk_match else None,
        }

    def _make_cache_token_key(self, profile: OpMemoryProfile) -> str:
        scope = profile.cache_token_scope or profile.op_name
        return f"{scope}|{profile.op_name}"

    def _allocate_cached_token(self, rank, ts, profile: OpMemoryProfile, phase: str, size: int):
        size = int(size)
        if size <= 0:
            return
        token_id = self._cache_token_id
        self._cache_token_id += 1
        scope_tags = self._parse_scope_tags(profile.cache_token_scope or profile.op_name)
        token = {
            "token_id": token_id,
            "rank": f"rank{rank}",
            "token_key": self._make_cache_token_key(profile),
            "token_scope": profile.cache_token_scope or profile.op_name,
            "op_name": profile.op_name,
            "microbatch": scope_tags["microbatch"],
            "chunk": scope_tags["chunk"],
            "alloc_phase": phase,
            "alloc_ts_us": float(ts) * 1e3,
            "free_phase": None,
            "free_ts_us": None,
            "size_bytes": size,
        }
        self._live_cached_tokens[rank][token_id] = token
        self._cached_tokens_by_key[rank][token["token_key"]].append(token_id)
        self.cache_token_events.append(
            {
                "action": "alloc",
                **token,
            }
        )
        self.cached_bytes[rank] += size

    def _free_cached_token(self, rank, ts, profile: OpMemoryProfile, phase: str):
        if int(profile.cache_size_bytes) <= 0:
            return
        token_key = self._make_cache_token_key(profile)
        token_queue = self._cached_tokens_by_key[rank].get(token_key, [])
        if not token_queue:
            raise RuntimeError(
                f"missing cached token for rank{rank} key={token_key} release={profile.cache_size_bytes}"
            )

        token_id = token_queue.pop(0)
        token = self._live_cached_tokens[rank].pop(token_id)
        if not token_queue:
            self._cached_tokens_by_key[rank].pop(token_key, None)
        expected_size = int(profile.cache_size_bytes)
        if token["size_bytes"] != expected_size:
            raise RuntimeError(
                f"cached token size mismatch for rank{rank} key={token_key}: "
                f"live={token['size_bytes']} release={expected_size}"
            )

        token["free_phase"] = phase
        token["free_ts_us"] = float(ts) * 1e3
        self.cache_token_events.append(
            {
                "action": "free",
                **token,
            }
        )
        self.cached_bytes[rank] -= token["size_bytes"]
        if self.cached_bytes[rank] < 0:
            raise RuntimeError(f"cached_bytes underflow for rank{rank}")

    def _append_counter(self, rank, ts, allocated_bytes, phase, op_name, kind, scope=""):
        self.peak_bytes[rank] = max(self.peak_bytes[rank], int(allocated_bytes))
        temp_bytes = max(0, int(allocated_bytes) - int(self.static_bytes[rank]) - int(self.cached_bytes[rank]))
        scope_tags = self._parse_scope_tags(scope)
        self.events.append(
            {
                "name": "mem",
                "cat": "memory",
                "ph": "C",
                "ts": float(ts) * 1e3,
                "pid": f"rank{rank}",
                "args": {
                    "allocated_bytes": int(allocated_bytes),
                    "static_bytes": int(self.static_bytes[rank]),
                    "cached_bytes": int(self.cached_bytes[rank]),
                    "temp_bytes": int(temp_bytes),
                    "cached_token_count": len(self._live_cached_tokens[rank]),
                    "phase": phase,
                    "op_name": op_name,
                    "kind": kind,
                },
            }
        )
        self.snapshots.append(
            {
                "rank": f"rank{rank}",
                "ts_us": float(ts) * 1e3,
                "allocated_bytes": int(allocated_bytes),
                "static_bytes": int(self.static_bytes[rank]),
                "cached_bytes": int(self.cached_bytes[rank]),
                "temp_bytes": int(temp_bytes),
                "cached_token_count": len(self._live_cached_tokens[rank]),
                "phase": phase,
                "op_name": op_name,
                "kind": kind,
                "scope": scope_tags["scope"],
                "microbatch": scope_tags["microbatch"],
                "chunk": scope_tags["chunk"],
            }
        )

    def init_rank(self, rank, static_bytes):
        self.static_bytes[rank] = int(static_bytes)
        self.cached_bytes[rank] = 0
        self._append_counter(rank, 0.0, self.static_bytes[rank], "init", "static", "init")

    def phase_start(self, rank, ts, profile: OpMemoryProfile, phase):
        base = self.static_bytes[rank] + self.cached_bytes[rank]
        peak = base + profile.phase_peak_no_cache(phase)
        self._append_counter(rank, ts, base, phase, profile.op_name, "start", profile.cache_token_scope)
        self._append_counter(rank, ts + 1e-9, peak, phase, profile.op_name, "peak", profile.cache_token_scope)

    def phase_end(self, rank, ts, profile: OpMemoryProfile, phase):
        if profile.phase_allocates_cache(phase):
            self._allocate_cached_token(rank, ts, profile, phase, int(profile.cache_size_bytes))
        elif profile.phase_releases_cache(phase):
            self._free_cached_token(rank, ts, profile, phase)
        total = self.static_bytes[rank] + self.cached_bytes[rank]
        self._append_counter(rank, ts, total, phase, profile.op_name, "end", profile.cache_token_scope)

    def summary(self):
        return {
            "static_allocated_bytes_by_rank": {
                f"rank{rank}": int(v) for rank, v in sorted(self.static_bytes.items())
            },
            "peak_allocated_bytes_by_rank": {
                f"rank{rank}": int(v) for rank, v in sorted(self.peak_bytes.items())
            },
        }

    def snapshot(self):
        return {
            "schema": "simumax_memory_snapshot_v1",
            "notes": [
                "allocated_bytes includes static + cached + temporary op-local peak bytes",
                "temp_bytes is derived as allocated_bytes - static_bytes - cached_bytes",
                "cached_bytes represents currently live activation cache retained for backward",
                "cache_tokens records first-hand cached activation lifetime tracked by the simulator",
            ],
            "events": self.snapshots,
            "cache_tokens": self.cache_token_events,
        }

    def memory_viz_snapshot(self):
        """Build a minimal torch.cuda.memory._snapshot()-compatible payload."""

        def _rank_to_device(rank_name: str) -> int:
            if isinstance(rank_name, int):
                return rank_name
            if isinstance(rank_name, str) and rank_name.startswith("rank"):
                return int(rank_name[4:])
            return int(rank_name)

        def _frame(op_name: str, phase: str, kind: str, component: str, microbatch=None, chunk=None):
            tags = []
            if microbatch is not None:
                tags.append(f"mb{microbatch}")
            if chunk is not None:
                tags.append(f"chunk{chunk}")
            tag_suffix = f"[{','.join(tags)}]" if tags else ""
            return [
                {
                    "filename": "simumax",
                    "line": 0,
                    "name": f"{component}:{phase}:{kind}:{op_name}{tag_suffix}",
                }
            ]

        def _alloc_from_stack(active_stack, size, component):
            remain = int(size)
            while remain > 0 and active_stack[component]:
                pop_idx = 0 if component == "cached" else -1
                block = active_stack[component][pop_idx]
                if block["size"] <= remain:
                    remain -= block["size"]
                    yield {"freed": active_stack[component].pop(pop_idx), "realloc": None}
                    continue
                freed = {
                    "addr": block["addr"],
                    "size": remain,
                    "frames": block["frames"],
                }
                remainder = {
                    "addr": block["addr"] + remain,
                    "size": block["size"] - remain,
                    "frames": block["frames"],
                }
                active_stack[component][pop_idx] = remainder
                remain = 0
                yield {"freed": freed, "realloc": remainder}
            if remain != 0:
                raise RuntimeError(f"free underflow in memory_viz export for {component}: {remain}")

        max_device = -1
        ranks = sorted({snapshot["rank"] for snapshot in self.snapshots}, key=_rank_to_device)
        if ranks:
            max_device = max(_rank_to_device(rank) for rank in ranks)

        snapshot = {
            "device_traces": [[] for _ in range(max_device + 1)],
            "segments": [],
        }

        for rank_name in ranks:
            device = _rank_to_device(rank_name)
            rank_events = [event for event in self.snapshots if event["rank"] == rank_name]
            rank_token_events = [event for event in self.cache_token_events if event["rank"] == rank_name]
            rank_events.sort(
                key=lambda event: (
                    0 if event["phase"] == "init" and event["kind"] == "init" else 1,
                    event["ts_us"],
                    {"init": 0, "start": 1, "peak": 2, "end": 3}.get(event["kind"], 9),
                )
            )
            rank_token_events.sort(
                key=lambda event: (
                    event["alloc_ts_us"] if event["action"] == "alloc" else event["free_ts_us"],
                    0 if event["action"] == "alloc" else 1,
                    event["token_id"],
                )
            )
            dynamic_required = 0
            probe_state = {"static": 0, "cached": 0, "temp": 0}
            for event in rank_events:
                new_state = {
                    "static": int(event["static_bytes"]),
                    "cached": int(event["cached_bytes"]),
                    "temp": int(event["temp_bytes"]),
                }
                for component in ("cached", "temp"):
                    delta = new_state[component] - probe_state[component]
                    if delta > 0:
                        dynamic_required += int(delta)
                    probe_state[component] = new_state[component]
            dynamic_required = max(
                dynamic_required,
                sum(int(event["size_bytes"]) for event in rank_token_events if event["action"] == "alloc"),
            )
            static_size = int(self.static_bytes[device])
            base_addr = (device + 1) << 40
            static_addr = base_addr
            dynamic_next_addr = base_addr + static_size
            active_stack = {"static": [], "cached": [], "temp": []}
            active_cached_tokens = {}
            final_blocks = []
            current = {"static": 0, "cached": 0, "temp": 0}
            emitted_init_snapshot = False
            peak_bytes = int(max(self.peak_bytes[device], dynamic_required + static_size))

            if peak_bytes > 0:
                snapshot["device_traces"][device].append(
                    {
                        "action": "segment_alloc",
                        "addr": base_addr,
                        "size": peak_bytes,
                        "stream": 0,
                        "frames": [],
                    }
                )

            token_idx = 0

            def _drain_token_events(up_to_ts_us, include_equal):
                nonlocal token_idx, dynamic_next_addr
                while token_idx < len(rank_token_events):
                    token_event = rank_token_events[token_idx]
                    token_ts = (
                        token_event["alloc_ts_us"]
                        if token_event["action"] == "alloc"
                        else token_event["free_ts_us"]
                    )
                    if token_ts > up_to_ts_us or (token_ts == up_to_ts_us and not include_equal):
                        break
                    if token_event["action"] == "alloc":
                        block = {
                            "addr": dynamic_next_addr,
                            "size": int(token_event["size_bytes"]),
                            "frames": _frame(
                                token_event["op_name"],
                                token_event["alloc_phase"],
                                "alloc",
                                "cached",
                                token_event.get("microbatch"),
                                token_event.get("chunk"),
                            ),
                        }
                        dynamic_next_addr += block["size"]
                        active_cached_tokens[token_event["token_id"]] = block
                        snapshot["device_traces"][device].append(
                            {
                                "action": "alloc",
                                "addr": block["addr"],
                                "size": block["size"],
                                "stream": 0,
                                "frames": block["frames"],
                            }
                        )
                    else:
                        block = active_cached_tokens.pop(token_event["token_id"])
                        snapshot["device_traces"][device].append(
                            {
                                "action": "free_requested",
                                "addr": block["addr"],
                                "size": block["size"],
                                "stream": 0,
                                "frames": block["frames"],
                            }
                        )
                        snapshot["device_traces"][device].append(
                            {
                                "action": "free_completed",
                                "addr": block["addr"],
                                "size": block["size"],
                                "stream": 0,
                                "frames": block["frames"],
                            }
                        )
                        final_blocks.append(
                            {
                                "address": block["addr"],
                                "size": block["size"],
                                "requested_size": block["size"],
                                "state": "inactive",
                                "frames": block["frames"],
                            }
                        )
                    token_idx += 1

            for event in rank_events:
                _drain_token_events(event["ts_us"], include_equal=False)

                if (
                    not emitted_init_snapshot
                    and not (event["phase"] == "init" and event["kind"] == "init")
                ):
                    snapshot["device_traces"][device].append(
                        {
                            "action": "snapshot",
                            "stream": 0,
                    "frames": _frame("static", "init", "snapshot", "static"),
                }
            )
                    emitted_init_snapshot = True
                new_state = {
                    "static": int(event["static_bytes"]),
                    "cached": int(event["cached_bytes"]),
                    "temp": int(event["temp_bytes"]),
                }
                for component in ("static", "cached", "temp"):
                    prev = current[component]
                    nxt = new_state[component]
                    delta = nxt - prev
                    if component == "cached":
                        current[component] = nxt
                        continue
                    if delta > 0:
                        if component == "static":
                            addr = static_addr + prev
                        else:
                            addr = dynamic_next_addr
                            dynamic_next_addr += int(delta)
                        block = {
                            "addr": addr,
                            "size": int(delta),
                            "frames": _frame(
                                event["op_name"],
                                event["phase"],
                                event["kind"],
                                component,
                                event.get("microbatch"),
                                event.get("chunk"),
                            ),
                        }
                        active_stack[component].append(block.copy())
                        snapshot["device_traces"][device].append(
                            {
                                "action": "alloc",
                                "addr": block["addr"],
                                "size": block["size"],
                                "stream": 0,
                                "frames": block["frames"],
                            }
                        )
                    elif delta < 0:
                        for item in _alloc_from_stack(active_stack, -delta, component):
                            block = item["freed"]
                            if component != "static":
                                snapshot["device_traces"][device].append(
                                    {
                                        "action": "free_requested",
                                        "addr": block["addr"],
                                        "size": block["size"],
                                        "stream": 0,
                                        "frames": block["frames"],
                                    }
                                )
                                snapshot["device_traces"][device].append(
                                    {
                                        "action": "free_completed",
                                        "addr": block["addr"],
                                        "size": block["size"],
                                        "stream": 0,
                                        "frames": block["frames"],
                                    }
                                )
                                if item["realloc"] is not None:
                                    snapshot["device_traces"][device].append(
                                        {
                                            "action": "alloc",
                                            "addr": item["realloc"]["addr"],
                                            "size": item["realloc"]["size"],
                                            "stream": 0,
                                            "frames": item["realloc"]["frames"],
                                        }
                                    )
                            final_blocks.append(
                                {
                                    "address": block["addr"],
                                    "size": block["size"],
                                    "requested_size": block["size"],
                                    "state": "inactive",
                                    "frames": block["frames"],
                                }
                            )
                    current[component] = nxt

                _drain_token_events(event["ts_us"], include_equal=True)

            if not emitted_init_snapshot:
                snapshot["device_traces"][device].append(
                    {
                        "action": "snapshot",
                        "stream": 0,
                        "frames": _frame("static", "init", "snapshot", "static"),
                    }
                )

            active_blocks = []
            for component in ("static", "cached", "temp"):
                blocks = (
                    active_cached_tokens.values()
                    if component == "cached"
                    else active_stack[component]
                )
                for block in blocks:
                    active_blocks.append(
                        {
                            "address": block["addr"],
                            "size": block["size"],
                            "requested_size": block["size"],
                            "state": "active_allocated",
                            "frames": block["frames"],
                        }
                    )

            all_blocks = sorted(final_blocks + active_blocks, key=lambda block: block["address"])
            blocks = []
            cursor = base_addr
            segment_end = base_addr + peak_bytes
            for block in all_blocks:
                if cursor < block["address"]:
                    blocks.append({"size": block["address"] - cursor, "state": "inactive"})
                blocks.append(
                    {
                        "size": block["size"],
                        "requested_size": block.get("requested_size", block["size"]),
                        "frames": block.get("frames", []),
                        "state": block["state"],
                    }
                )
                cursor = block["address"] + block["size"]
            if cursor < segment_end:
                blocks.append({"size": segment_end - cursor, "state": "inactive"})

            snapshot["segments"].append(
                {
                    "device": device,
                    "address": base_addr,
                    "total_size": segment_end - base_addr,
                    "stream": 0,
                    "segment_type": "large",
                    "allocated_size": int(self.static_bytes[device] + self.cached_bytes[device]),
                    "active_size": int(self.static_bytes[device] + self.cached_bytes[device]),
                    "blocks": blocks if blocks else [{"size": max(peak_bytes, 1), "state": "inactive"}],
                }
            )

        return snapshot
