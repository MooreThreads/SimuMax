"""Various utilities"""

import json
import os, subprocess

class HumanReadableSize:
    """Convert a size in bytes to a human-readable format."""

    BYTE_UNITS = ["B", "KB", "MB", "GB", "TB"]
    NUM_UNITS = ["", "K", "M", "B", "T"]
    TIME_UNITS = ["ms", "s"]

    def __init__(
        self, value, base=1024, units=None, source_unit=None, target_unit=None
    ):
        """
        :param value: original value
        :param base: base: 1024 for byte conversion, 1000 for FLOPS or parameter conversion
        :param units:  ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        :param target_unit:  target unit, if specified, force conversion to this unit
        """
        self.original_value = float(value)
        self.base = base
        self.units = units or ["B", "KB", "MB", "GB", "TB", "PB"]
        self.source_unit = source_unit or self.units[0]
        self.target_unit = target_unit
        assert self.source_unit in self.units
        assert self.target_unit is None or self.target_unit in self.units
        self.converted_value, self.unit = self._convert()

    def _convert(self):
        size = self.original_value
        source_index = self.units.index(self.source_unit)
        size_in_base_unit = size * (self.base**source_index)

        # If the target unit is provided, convert to the specified unit
        if self.target_unit and self.target_unit in self.units:
            target_index = self.units.index(self.target_unit)
            size_in_target_unit = size_in_base_unit / (self.base**target_index)
            return size_in_target_unit, self.target_unit

        # Automatically select units
        unit_index = 0
        size_in_target_unit = size_in_base_unit
        while size_in_target_unit >= self.base and unit_index < len(self.units) - 1:
            size_in_target_unit /= self.base
            unit_index += 1

        return size_in_target_unit, self.units[unit_index]

    @staticmethod
    def from_string(size_str, units, base, target_unit=None):
        """Parse a size string like '500 MB' or '1 GB'."""
        value, source_unit = size_str.split(" ")
        if source_unit not in units:
            raise ValueError(f"Unknown unit: '{source_unit}'")
        return HumanReadableSize(
            float(value),
            base=base,
            units=units,
            source_unit=source_unit,
            target_unit=target_unit,
        )

    def __str__(self):
        return f"{self.converted_value:.4f} {self.unit}"

    def get_value(self):
        return self.converted_value

    def get_unit(self):
        return self.unit


def human_readable_bytes(value, target_unit=None):
    return str(
        HumanReadableSize(
            value,
            base=1024,
            units=HumanReadableSize.BYTE_UNITS,
            target_unit=target_unit,
        )
    )


def human_readable_nums(value, target_unit=None):
    return str(
        HumanReadableSize(
            value, base=1000, units=HumanReadableSize.NUM_UNITS, target_unit=target_unit
        )
    )


def human_readable_times(value, target_unit=None):
    return str(
        HumanReadableSize(
            value,
            base=1000,
            units=HumanReadableSize.TIME_UNITS,
            target_unit=target_unit,
        )
    )


def convert_final_result_to_human_format(result: dict):
    """：
    Based on the regularity of the key value of result,
    convert the value to a human-readable format.
    """
    if result is None:
        return

    for k, v in result.items():
        if not isinstance(v, (int, float, dict)):
            continue
        if isinstance(v, dict):
            convert_final_result_to_human_format(v)
            continue
        convert_func = None
        if "time" in k:
            convert_func = human_readable_times
        elif "mem" in k or "bytes" in k:
            convert_func = human_readable_bytes
        elif "flops" in k:
            convert_func = human_readable_nums
        if convert_func is None:
            continue
        result[k] = convert_func(v)
    return


def to_json_string(obj: dict):
    return json.dumps(obj, indent=2, sort_keys=False, ensure_ascii=False)


def get_point_name(parent, current, sep=" -> ") -> str:
    if parent and current:
        res = parent + sep + current
    else:
        res = parent if parent else current
    return res


def path_convert_to_str(path: list) -> str:
    path_name = ""
    if len(path) == 1:
        path_name = path[0]
    elif len(path) > 1:
        path_name = " -> ".join(path)
    return path_name

def get_rank_group(global_rank, strategy):
    ## world_size = tp*pp*dp

    tp_rank = global_rank % strategy.tp_size
    pp_rank = (global_rank // strategy.tp_size) % strategy.pp_size
    dp_rank = (global_rank // (strategy.tp_size * strategy.pp_size))
    ep_rank = dp_rank % strategy.ep_size
    edp_rank = dp_rank // strategy.ep_size
    tp_group_id = f"pp:{pp_rank}-dp:{dp_rank}"
    pp_group_id = f"tp:{tp_rank}-dp:{dp_rank}"
    dp_group_id = f"tp:{tp_rank}-pp:{pp_rank}"
    ep_group_id = f"tp:{tp_rank}-pp:{pp_rank}-edp:{edp_rank}"
    edp_group_id = f"tp:{tp_rank}-pp:{pp_rank}-ep:{ep_rank}"
    dic = {
        "tp_group_id": tp_group_id,
        "tp_rank": tp_rank,
        "pp_group_id": pp_group_id,
        "pp_rank": pp_rank,
        "dp_group_id": dp_group_id,
        "dp_rank": dp_rank,
        "ep_group_id": ep_group_id,
        "ep_rank": ep_rank,
        "edp_group_id": edp_group_id,
        "edp_rank": edp_rank,
    }
    return dic

def merge_dict(cur_data, merges_data):
    if len(merges_data) == 0:
        for key, value in cur_data.items():
            merges_data[key] = [value]
    else:
        for key, value in cur_data.items():
            merges_data[key].append(value)  
    return merges_data

def rm_tmp():
    if os.path.exists("./tmp"):
        subprocess.run(["rm", "-rf", "./tmp"])