from __future__ import annotations

import inspect
import textwrap
from datetime import datetime, timezone

from .graph import Graph
from .packet import ColumnRole, DataFramePacket, resolve_role_column
from .util import ROLE_PARAM_NAMES, accepts_param, find_role_param


class CompileError(Exception):
    pass


def _sanitize(name: str) -> str:
    out = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    return out if out and not out[0].isdigit() else f"_{out}"


def _resolve_role_value(graph: Graph, runner, block_id: str, role: ColumnRole) -> str | None:
    """Same dynamic default as Runner.run_block (see packet.resolve_role_column),
    computed here from the already-cached, already-green upstream packets
    `runner` holds (compiling requires every reachable block to be green
    first -- see the not_ready check above) rather than re-running anything.
    Retagging a column's role changes the block's cache key (see
    Runner.compute_key), so a block whose target/predicted column just moved
    is grey again and (in strict mode) blocks compilation until it's re-run
    -- by the time this runs, the cached packet always reflects the current
    tag."""
    if runner is None:
        return None
    schema_metas = []
    for wire in graph.input_wires(block_id).values():
        st = runner.state.get(wire.from_block)
        if st is None or st.last_successful_key is None:
            continue
        entry = runner.cache.get(st.last_successful_key)
        if entry is None:
            continue
        packet = entry.outputs.get(wire.from_port)
        if isinstance(packet, DataFramePacket):
            schema_metas.append(packet.schema_meta)
    return resolve_role_column(schema_metas, role)


def compile_graph(
    graph: Graph,
    runner=None,
    output_blocks: list[str] | None = None,
    strict: bool = True,
) -> str:
    """Compile the graph to a single, self-contained Python script (plan
    section 7, default/clean mode — DataFramePacket/ColumnMeta scaffolding
    is absent by construction, since block bodies are plain-dataframe
    functions to begin with). `--with-metadata` verbose mode is not yet
    implemented.
    """
    targets = output_blocks or list(graph.blocks)
    reachable = graph.ancestors(targets)

    if runner is not None:
        not_ready = sorted(b for b in reachable if runner.status(b) in ("grey", "red"))
        if not_ready and strict:
            raise CompileError(f"blocks not ready to compile (grey/red): {not_ready}")

    order = [b for b in graph.topo_order() if b in reachable]

    fn_names: dict[str, str] = {}
    wants_output_dir: dict[str, bool] = {}
    wants_block_id: dict[str, bool] = {}
    wants_role_params: dict[str, dict[ColumnRole, str]] = {}
    def_lines: list[str] = []
    # Two blocks that run the same code (same category, for registry blocks;
    # same category *and* code text, for custom ones) get exactly one
    # function definition, shared across every instance's call site below --
    # e.g. two WoE blocks compile to one `woe_transform` function called
    # twice with each instance's own params, not two near-identical copies.
    seen_fns: dict[tuple[str, str], str] = {}
    used_fn_names: set[str] = set()
    for bid in order:
        block = graph.blocks[bid]
        fn = block.resolved_fn()
        wants_output_dir[bid] = accepts_param(fn, "output_dir")
        wants_block_id[bid] = accepts_param(fn, "block_id")
        wants_role_params[bid] = {
            role: role_param for role in ROLE_PARAM_NAMES if (role_param := find_role_param(fn, role)) is not None
        }
        src = block.code if block.is_custom else inspect.getsource(fn)
        src = textwrap.dedent(src).strip("\n")
        fn_sig = (block.category, src)
        existing_fn_name = seen_fns.get(fn_sig)
        if existing_fn_name is not None:
            fn_names[bid] = existing_fn_name
            continue

        # A custom (AI-authored) block's function is named after the block
        # itself -- the same human-chosen name already used for its call-site
        # variable below -- rather than its auto-generated category (e.g.
        # "ai_block_lz3k9f"), so the compiled def reads like the block does
        # in the UI. Registry blocks keep the plain category name (a stable,
        # well-known function name shared by every instance); a different
        # *bodied* block already claiming it only bites two custom blocks
        # that happen to share a category but were drafted with different
        # bodies, where the block id disambiguates them.
        base_name = _sanitize(block.name) if block.is_custom else _sanitize(block.category)
        fn_name = base_name if base_name not in used_fn_names else f"{base_name}_{bid}"
        used_fn_names.add(fn_name)
        seen_fns[fn_sig] = fn_name
        fn_names[bid] = fn_name
        src = src.replace(f"def {fn.__name__}(", f"def {fn_name}(", 1)
        def_lines.append(f"# === Function: {fn_name} | type={block.block_type} | category={block.category} ===")
        def_lines.append(src)
        def_lines.append("")
        def_lines.append("")

    call_lines: list[str] = []
    var_names: dict[tuple[str, str], str] = {}
    used_var_names: set[str] = set()
    current_lane: str | None = "__unset__"
    uses_group_by = any(graph.blocks[bid].group_by for bid in order)
    for bid in order:
        block = graph.blocks[bid]
        # Lanes group the pipeline into modeling phases (Data Prep -> Feature
        # Engineering -> ...); a banner marks each phase's calls in the
        # compiled script, purely for readability -- lane membership plays no
        # role in execution order (see graph.topo_order).
        lane_name = graph.lanes[block.lane].name if block.lane in graph.lanes else None
        if lane_name != current_lane:
            current_lane = lane_name
            if lane_name:
                call_lines.append(f"# ===== Lane: {lane_name} =====")

        # (kwarg name, source-code expression for its value) pairs -- kept
        # apart instead of pre-joined into "name=value" text so grouped
        # blocks (below) can split dataframe-typed wired inputs (candidates
        # for partitioning) from everything else (params, role defaults,
        # output_dir/block_id -- unchanged across groups).
        dataframe_ports = {p.name for p in block.inputs if p.type == "dataframe"}
        kwarg_pairs: list[tuple[str, str]] = []
        for port, wire in sorted(graph.input_wires(bid).items()):
            kwarg_pairs.append((port, var_names[(wire.from_block, wire.from_port)]))
        for pname, pval in block.params.items():
            kwarg_pairs.append((pname, repr(pval)))
        for role, role_param in wants_role_params[bid].items():
            if role_param not in block.params:
                resolved = _resolve_role_value(graph, runner, bid, role)
                if resolved is not None:
                    kwarg_pairs.append((role_param, repr(resolved)))
        if block.block_type == "output" and wants_output_dir[bid]:
            kwarg_pairs.append(("output_dir", "OUTPUT_DIR"))
        if block.block_type == "output" and wants_block_id[bid]:
            kwarg_pairs.append(("block_id", repr(bid)))

        out_ports = [p.name for p in block.outputs]
        call_lines.append(f'# --- Call: {bid} | name="{block.name}" ---')

        varlist = []
        for p in out_ports:
            # A port the user's named (see BlockInstance.port_names,
            # settable by clicking that output's data in the UI) becomes
            # the variable holding it here, so the compiled script reads
            # with the same names the user gave the data -- falling back
            # to the block-name/id/port scheme, which is always unique by
            # construction, for any port left unnamed or whose chosen
            # name collides with another one already used in this script.
            custom = block.port_names.get(p)
            var = _sanitize(custom) if custom else None
            if var is None or var in used_var_names:
                var = f"{_sanitize(block.name)}_{bid}" if len(out_ports) == 1 else f"{_sanitize(block.name)}_{bid}_{p}"
            used_var_names.add(var)
            var_names[(bid, p)] = var
            varlist.append(var)

        if block.group_by:
            call_lines.extend(_grouped_call_lines(bid, block, fn_names[bid], dataframe_ports, kwarg_pairs, varlist))
        else:
            call = f"{fn_names[bid]}({', '.join(f'{k}={v}' for k, v in kwarg_pairs)})"
            if not out_ports:
                call_lines.append(call)
            else:
                call_lines.append(f"{', '.join(varlist)} = {call}" if len(varlist) > 1 else f"{varlist[0]} = {call}")
        call_lines.append("")

    header = [
        "# Generated by Model-Maker. Do not edit function bodies here directly --",
        "# edit the source blocks in the project and recompile.",
        f"# Compiled: {datetime.now(timezone.utc).isoformat()}",
        "",
        "import os",
        "import polars as pl",
    ]
    if uses_group_by:
        header.append("from concurrent.futures import ThreadPoolExecutor")
    header += [
        "",
        'OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")',
        "os.makedirs(OUTPUT_DIR, exist_ok=True)",
        "",
        "",
    ]
    if uses_group_by:
        header.append(_COMBINE_GROUP_RESULTS_SRC)
        header.append("")

    return "\n".join(header + def_lines + call_lines) + "\n"


# A grouped block's per-group calls each return whatever the block normally
# returns (a single value, or a tuple for a multi-output block) -- this
# reshapes {group_value: that return value} into the single/tuple result
# the rest of the script expects, exactly mirroring Runner._combine_group_results
# so a compiled script's grouped output matches the live engine's. Emitted
# once, only when at least one block in the graph uses group_by.
_COMBINE_GROUP_RESULTS_SRC = '''def _combine_group_results(group_col, group_values, raw_by_group, out_names):
    if not out_names:
        return None
    per_group = {}
    for gval in group_values:
        raw = raw_by_group[gval]
        per_group[gval] = (raw,) if len(out_names) == 1 else tuple(raw)
    combined = []
    for i in range(len(out_names)):
        values = [per_group[g][i] for g in group_values]
        if all(isinstance(v, pl.DataFrame) for v in values):
            parts = [
                v if group_col in v.columns else v.with_columns(pl.lit(gval).alias(group_col))
                for gval, v in zip(group_values, values)
            ]
            combined.append(pl.concat(parts, how="diagonal_relaxed"))
        elif all(isinstance(v, dict) for v in values):
            combined.append(pl.DataFrame([{group_col: gval, **v} for gval, v in zip(group_values, values)]))
        else:
            combined.append(dict(zip(group_values, values)))
    return combined[0] if len(combined) == 1 else tuple(combined)
'''


def _grouped_call_lines(
    bid: str,
    block,
    fn_name: str,
    dataframe_ports: set[str],
    kwarg_pairs: list[tuple[str, str]],
    varlist: list[str],
) -> list[str]:
    """Emit a group_by block's call as a partition/dispatch/recombine
    sequence instead of a single call -- mirrors Runner._run_grouped, but
    threaded (ThreadPoolExecutor) rather than process-isolated: this is a
    one-shot batch script the user runs themselves, not a long-lived server
    process other work needs protecting from, so the extra isolation the
    live app pays for isn't worth the portability cost here (a compiled
    script using real subprocesses would need every grouped block wrapped
    in `if __name__ == "__main__":` to be safe under Windows' default
    "spawn" start method)."""
    df_pairs = [(k, v) for k, v in kwarg_pairs if k in dataframe_ports]
    other_pairs = [(k, v) for k, v in kwarg_pairs if k not in dataframe_ports]
    group_col = block.group_by
    max_workers = block.max_workers or 4
    suffix = bid

    lines = [
        f"_df_kwargs_{suffix} = {{{', '.join(f'{k!r}: {v}' for k, v in df_pairs)}}}",
        f"_fixed_kwargs_{suffix} = {{{', '.join(f'{k!r}: {v}' for k, v in other_pairs)}}}",
        f"_groupable_{suffix} = {{k: v for k, v in _df_kwargs_{suffix}.items() if {group_col!r} in v.columns}}",
        f"if not _groupable_{suffix}:",
        f"    raise RuntimeError({f'group_by column {group_col!r} not found in any dataframe input of block {bid!r}'!r})",
        f"_primary_port_{suffix} = next(iter(_groupable_{suffix}))",
        f"_n_groups_{suffix} = _groupable_{suffix}[_primary_port_{suffix}].select(pl.col({group_col!r}).n_unique()).item() or 0",
        f"if _n_groups_{suffix} == 0:",
        f"    raise RuntimeError({f'no groups found for group_by column {group_col!r}'!r})",
        f"if _n_groups_{suffix} > 500:",
        f"    raise RuntimeError(f\"grouping by {group_col!r} would produce {{_n_groups_{suffix}}} groups, over the 500-group safety limit\")",
        f"_partitions_{suffix} = {{"
        f"port: {{k[0]: v for k, v in frame.partition_by({group_col!r}, as_dict=True).items()}} "
        f"for port, frame in _groupable_{suffix}.items()}}",
        f"_group_values_{suffix} = list(_partitions_{suffix}[_primary_port_{suffix}].keys())",
        "",
        f"def _task_{suffix}(_gval):",
        f"    _kwargs = dict(_df_kwargs_{suffix})",
        f"    _kwargs.update(_fixed_kwargs_{suffix})",
        f"    for _port, _parts in _partitions_{suffix}.items():",
        "        if _gval in _parts:",
        "            _kwargs[_port] = _parts[_gval]",
        f"    return {fn_name}(**_kwargs)",
        "",
        f"with ThreadPoolExecutor(max_workers=min({max_workers}, _n_groups_{suffix})) as _ex:",
        f"    _raw_{suffix} = dict(zip(_group_values_{suffix}, _ex.map(_task_{suffix}, _group_values_{suffix})))",
        "",
        f"_combined_{suffix} = _combine_group_results({group_col!r}, _group_values_{suffix}, _raw_{suffix}, {varlist!r})",
    ]
    if varlist:
        lines.append(f"{', '.join(varlist)} = _combined_{suffix}" if len(varlist) > 1 else f"{varlist[0]} = _combined_{suffix}")
    return lines
