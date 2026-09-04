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

        kwargs = []
        for port, wire in sorted(graph.input_wires(bid).items()):
            kwargs.append(f"{port}={var_names[(wire.from_block, wire.from_port)]}")
        for pname, pval in block.params.items():
            kwargs.append(f"{pname}={pval!r}")
        for role, role_param in wants_role_params[bid].items():
            if role_param not in block.params:
                resolved = _resolve_role_value(graph, runner, bid, role)
                if resolved is not None:
                    kwargs.append(f"{role_param}={resolved!r}")
        if block.block_type == "output" and wants_output_dir[bid]:
            kwargs.append("output_dir=OUTPUT_DIR")
        if block.block_type == "output" and wants_block_id[bid]:
            kwargs.append(f"block_id={bid!r}")

        out_ports = [p.name for p in block.outputs]
        call = f"{fn_names[bid]}({', '.join(kwargs)})"
        call_lines.append(f'# --- Call: {bid} | name="{block.name}" ---')
        if not out_ports:
            call_lines.append(call)
        else:
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
            call_lines.append(f"{', '.join(varlist)} = {call}" if len(varlist) > 1 else f"{varlist[0]} = {call}")
        call_lines.append("")

    header = [
        "# Generated by Model-Maker. Do not edit function bodies here directly --",
        "# edit the source blocks in the project and recompile.",
        f"# Compiled: {datetime.now(timezone.utc).isoformat()}",
        "",
        "import os",
        "import polars as pl",
        "",
        'OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")',
        "os.makedirs(OUTPUT_DIR, exist_ok=True)",
        "",
        "",
    ]

    return "\n".join(header + def_lines + call_lines) + "\n"
