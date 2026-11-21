"""Data: read sources, filter, join, split and write.

Each block is a plain function over pl.DataFrame and literal params -- no
DataFramePacket, no ColumnMeta. A compiled script emits these verbatim; the
engine wraps packets around calls at run time and applies each block's
declared metadata rules separately.
"""

from __future__ import annotations

import polars as pl

BLOCKS = {
    "read_csv": {
        "capabilities": ["filesystem"],
        "use_instead": "read_dataset",
        "display_name": "Read CSV",
        "version": 1,
        "block_type": "input",
        "inputs": [],
        "outputs": [{"name": "out", "lazy": True}],
        "form": [
            {"key": "path", "label": "File path", "kind": "path", "options": [".csv", ".txt"]},
            {"key": "separator", "label": "Separator (e.g. , or | or \\t)", "kind": "text", "placeholder": ","},
            {"key": "has_header", "label": "First row is a header", "kind": "checkbox"},
            {
                "key": "column_names",
                "label": "Column names (comma-separated; optional, replaces the header or column_1, column_2, ...)",
                "kind": "text",
            },
            {
                "key": "infer_schema_rows",
                "label": "Rows to infer dtypes from (0 = read every column as text)",
                "kind": "number",
            },
        ],
        "meta": {"inherit": "none"},
    },
    "stage_csv_parquet": {
        "capabilities": ["filesystem"],
        "use_instead": "read_dataset",
        "display_name": "Stage CSV as Parquet",
        "version": 1,
        "block_type": "input",
        "inputs": [],
        "outputs": [{"name": "out", "lazy": True}],
        "form": [
            {
                "key": "path",
                "label": "File, folder or glob (.csv/.txt, or zipped/gzipped; a folder takes every such file in it)",
                "kind": "path",
                "options": [".csv", ".txt", ".zip", ".gz"],
            },
            {"key": "separator", "label": "Separator (e.g. , or | or \\t)", "kind": "text", "placeholder": ","},
            {"key": "has_header", "label": "First row is a header", "kind": "checkbox"},
            {
                "key": "column_names",
                "label": "Column names (comma-separated; optional, replaces the header or column_1, column_2, ...)",
                "kind": "text",
            },
            {
                "key": "infer_schema_rows",
                "label": "Rows to infer dtypes from (0 = read every column as text)",
                "kind": "number",
            },
            {
                "key": "columns",
                "label": "Columns to keep (comma-separated; optional, default all -- a slim stage of a wide file)",
                "kind": "text",
            },
            {
                "key": "compression",
                "label": "Compression",
                "kind": "select",
                "options": ["zstd", "snappy", "lz4", "gzip", "uncompressed"],
            },
            {
                "key": "parquet_dir",
                "label": "Parquet folder (optional; default: <file or folder name>_parquet next to it)",
                "kind": "text",
            },
        ],
        "meta": {"inherit": "none"},
    },
    "demo_credit_data": {
        "display_name": "Demo credit data (PD/LGD/CCF)",
        "version": 1,
        "block_type": "input",
        "inputs": [],
        "outputs": [{"name": "out"}],
        "meta": {"inherit": "none"},
    },
    "read_parquet": {
        "capabilities": ["filesystem"],
        "use_instead": "read_dataset",
        "display_name": "Read Parquet",
        "version": 1,
        "block_type": "input",
        "inputs": [],
        "outputs": [{"name": "out", "lazy": True}],
        "form": [{"key": "path", "label": "File path", "kind": "path", "options": [".parquet"]}],
        "meta": {"inherit": "none"},
    },
    "read_json": {
        "capabilities": ["filesystem"],
        "use_instead": "read_dataset",
        "display_name": "Read JSON",
        "version": 1,
        "block_type": "input",
        "inputs": [],
        "outputs": [{"name": "out"}],
        "form": [{"key": "path", "label": "File path", "kind": "path", "options": [".json"]}],
        "meta": {"inherit": "none"},
    },
    "read_excel": {
        "capabilities": ["filesystem"],
        "use_instead": "read_dataset",
        "display_name": "Read Excel",
        "version": 1,
        "block_type": "input",
        "inputs": [],
        "outputs": [{"name": "out"}],
        "form": [
            {"key": "path", "label": "File path", "kind": "path", "options": [".xlsx"]},
            {
                "key": "sheet",
                "label": "Sheet name (optional -- defaults to the first sheet)",
                "kind": "text",
            },
        ],
        "meta": {"inherit": "none"},
    },
    "read_sql": {
        "capabilities": ["filesystem"],
        "use_instead": "read_source",
        "display_name": "Read SQL",
        "version": 1,
        "block_type": "input",
        "inputs": [],
        "outputs": [{"name": "out"}],
        "form": [
            {
                "key": "connection_env",
                "label": "Connection variable",
                "kind": "env",
                "placeholder": "WAREHOUSE_URI",
            },
            {
                "key": "query",
                "label": "SQL query",
                "kind": "sql",
                "placeholder": "SELECT * FROM loans",
            },
        ],
        "meta": {"inherit": "none"},
    },
    "read_dataset": {
        "display_name": "Read dataset",
        "version": 1,
        "block_type": "input",
        "capabilities": ["datasets"],
        "inputs": [],
        "outputs": [{"name": "out", "lazy": True}],
        "form": [
            {"key": "dataset", "label": "Dataset", "kind": "dataset"},
            {"key": "sheet", "label": "Sheet name (Excel only; optional)", "kind": "text"},
        ],
        "meta": {"inherit": "none"},
    },
    "read_source": {
        "display_name": "Read from source",
        "version": 1,
        "block_type": "input",
        "capabilities": ["sources"],
        "inputs": [],
        "outputs": [{"name": "out", "lazy": True}],
        "form": [
            {"key": "source", "label": "Source", "kind": "source"},
            {"key": "query", "label": "SQL query", "kind": "sql", "placeholder": "SELECT * FROM loans"},
        ],
        "meta": {"inherit": "none"},
    },
    "filter": {
        "display_name": "Filter",
        "version": 1,
        "block_type": "standard",
        "tags": ["data_prep"],
        "inputs": [{"name": "df", "lazy": True}],
        "outputs": [{"name": "out"}],
        "form": [
            {
                "key": "expr",
                "label": "Filter expression (SQL)",
                "kind": "text",
                "placeholder": "age > 30 and region = 'West'",
            },
        ],
    },
    "select": {
        "display_name": "Select columns",
        "version": 1,
        "block_type": "standard",
        "tags": ["data_prep"],
        "inputs": [{"name": "df", "lazy": True}],
        "outputs": [{"name": "out"}],
        "form": [{"key": "cols", "label": "Columns to keep", "kind": "columns"}],
    },
    "groupby_agg": {
        "display_name": "Group by / aggregate",
        "version": 1,
        "block_type": "standard",
        "tags": ["data_prep"],
        "inputs": [{"name": "df", "lazy": True}],
        "outputs": [{"name": "out"}],
        "meta": {"inherit": {"columns_param": "by"}},
    },
    "join": {
        "display_name": "Join",
        "version": 1,
        "block_type": "standard",
        "tags": ["data_prep"],
        "inputs": [{"name": "left", "lazy": True}, {"name": "right", "lazy": True}],
        "outputs": [{"name": "out"}],
        "form": [
            {"key": "on", "label": "Join on", "kind": "columns"},
            {
                "key": "how",
                "label": "How",
                "kind": "select",
                "options": ["inner", "left", "right", "full", "semi", "anti"],
            },
        ],
    },
    "train_test_split": {
        "display_name": "Train/test split",
        "version": 2,
        "block_type": "standard",
        "tags": ["sampling"],
        "inputs": [{"name": "df"}],
        "outputs": [{"name": "train"}, {"name": "test"}],
        "form": [
            {"key": "test_size", "label": "Test size (fraction)", "kind": "number", "step": 0.05},
            {"key": "seed", "label": "Random seed", "kind": "number"},
            {
                "key": "stratify_col",
                "label": "Stratify by (optional, e.g. the default flag)",
                "kind": "column",
            },
            {
                "key": "group_col",
                "label": "Keep each ID on one side (optional, e.g. loan_id for panel data)",
                "kind": "column",
            },
        ],
        "hints": "split",
    },
    "time_split": {
        "display_name": "Out-of-time split",
        "version": 1,
        "block_type": "standard",
        "tags": ["sampling"],
        "inputs": [{"name": "df"}],
        "outputs": [{"name": "development"}, {"name": "out_of_time"}],
        "form": [
            {"key": "date_col", "label": "Date column", "kind": "column"},
            {
                "key": "cutoff",
                "label": "Cut-off date (first out-of-time day)",
                "kind": "text",
                "placeholder": "2025-01-01",
            },
            {
                "key": "oot_end",
                "label": "Out-of-time end date (optional, exclusive)",
                "kind": "text",
                "placeholder": "2026-01-01",
            },
        ],
        "hints": "split",
    },
    "derive_columns": {
        "display_name": "Derive columns",
        "version": 1,
        "block_type": "standard",
        "tags": ["data_prep"],
        "inputs": [{"name": "df", "lazy": True}],
        "outputs": [{"name": "out"}],
        "meta": {"new_role": "feature"},
    },
    "one_hot_encode": {
        "display_name": "One-hot encode",
        "version": 1,
        "block_type": "standard",
        "tags": ["data_prep"],
        "inputs": [{"name": "df"}],
        "outputs": [{"name": "out"}],
        "meta": {"new_role": "feature"},
    },
    "write_csv": {
        "capabilities": ["filesystem"],
        "use_instead": "export",
        "display_name": "Write CSV",
        "version": 1,
        "block_type": "output",
        "inputs": [{"name": "df", "lazy": True}],
        "outputs": [],
        "form": [
            {
                "key": "filename",
                "label": "Output filename",
                "kind": "text",
                "placeholder": "output.csv",
            },
        ],
        "meta": {"inherit": "none"},
    },
    "write_parquet": {
        "capabilities": ["filesystem"],
        "use_instead": "export",
        "display_name": "Write Parquet",
        "version": 1,
        "block_type": "output",
        "inputs": [{"name": "df", "lazy": True}],
        "outputs": [],
        "form": [
            {
                "key": "filename",
                "label": "Output filename",
                "kind": "text",
                "placeholder": "output.parquet",
            },
            {
                "key": "compression",
                "label": "Compression",
                "kind": "select",
                "options": ["zstd", "snappy", "lz4", "gzip", "uncompressed"],
            },
        ],
        "meta": {"inherit": "none"},
    },
    "export": {
        "display_name": "Export",
        "version": 1,
        "block_type": "output",
        "tags": ["output"],
        "capabilities": ["exports"],
        "inputs": [{"name": "df"}],
        "outputs": [{"name": "out"}],
        "form": [
            {"key": "format", "label": "Format", "kind": "select", "options": ["csv", "parquet"]},
            {"key": "name", "label": "Export name", "kind": "text", "placeholder": "<block name>"},
        ],
        "meta": {"inherit": "all"},
    },
    "display_table": {
        "display_name": "Display table",
        "version": 2,
        "block_type": "output",
        "tags": ["output"],
        "display": "table",
        "inputs": [{"name": "df"}],
        "outputs": [{"name": "out"}],
        "form": [
            {"key": "save_csv", "label": "Also save as CSV", "kind": "checkbox"},
            {"key": "filename", "label": "CSV file name", "kind": "text", "placeholder": "<block id>.csv"},
        ],
    },
    "display_value": {
        "display_name": "View value",
        "version": 2,
        "block_type": "output",
        "tags": ["output"],
        "display": "value",
        "inputs": [{"name": "value", "type": "any"}],
        "outputs": [{"name": "value", "type": "any"}],
        "form": [
            {"key": "save_csv", "label": "Also save as CSV", "kind": "checkbox"},
            {"key": "filename", "label": "CSV file name", "kind": "text", "placeholder": "<block id>.csv"},
        ],
    },
    "display_kpi": {
        "display_name": "Headline number",
        "version": 1,
        "block_type": "output",
        "tags": ["output"],
        "display": "kpi",
        "inputs": [{"name": "value", "type": "any"}],
        "outputs": [{"name": "kpi", "type": "scalar_metric"}],
        "form": [
            {"key": "column", "label": "Column (or metric key)", "kind": "column"},
            {
                "key": "agg",
                "label": "Aggregate rows",
                "kind": "select",
                "options": ["first", "last", "sum", "mean", "min", "max", "count"],
            },
            {"key": "label", "label": "Label", "kind": "text", "placeholder": "e.g. Total RWA"},
            {
                "key": "number_format",
                "label": "Format",
                "kind": "select",
                "options": ["number", "compact", "percent"],
            },
            {"key": "decimals", "label": "Decimals", "kind": "number", "step": 1},
            {"key": "unit", "label": "Unit", "kind": "text", "placeholder": "e.g. EUR"},
        ],
    },
    "generate_image": {
        "display_name": "Generate image",
        "version": 1,
        "block_type": "output",
        "tags": ["output"],
        "inputs": [{"name": "df"}],
        "outputs": [{"name": "image", "type": "image"}],
        "form": [
            {
                "key": "kind",
                "label": "Chart type",
                "kind": "select",
                "options": ["hist", "bar", "scatter", "line"],
            },
            {"key": "x", "label": "X column", "kind": "column"},
            {"key": "y", "label": "Y column (bar / scatter / line)", "kind": "column"},
            {"key": "bins", "label": "Bins (hist)", "kind": "number"},
            {"key": "title", "label": "Title", "kind": "text"},
        ],
        "meta": {"inherit": "none"},
    },
}


def read_csv(
    path: str,
    separator: str = ",",
    has_header: bool = True,
    column_names: str = "",
    infer_schema_rows: int = 100,
) -> pl.LazyFrame:
    """Reads a delimited text file into a dataframe on the `out` port; no
    inputs. `path` is the file path (relative to the working directory the
    engine runs in). `separator` is one character ("," by default; "|" for
    pipe-delimited files, "\\t" or "tab" for tab). `has_header` says whether
    the first row holds column names; without one the columns are named
    column_1, column_2, ... unless `column_names` (comma-separated) names
    them -- given with a header, it replaces the header's names.
    `infer_schema_rows` is how many rows polars reads to infer dtypes
    (default 100); 0 reads every column as text, the safe choice for a
    large file whose first rows aren't representative (cast afterwards).
    Column roles start untagged."""
    if separator in ("\\t", "tab", "TAB"):
        separator = "\t"
    if len(separator) != 1:
        raise ValueError(f"separator must be a single character, got {separator!r}")
    names = [n.strip() for n in column_names.split(",") if n.strip()] if column_names else None
    # A lazy scan: the engine collects it, pushing sample mode's row cap
    # down into the scan first, and a streaming run fuses it with whatever
    # follows (see the output port's "lazy" flag).
    return pl.scan_csv(
        path,
        separator=separator,
        has_header=has_header,
        new_columns=names,
        infer_schema_length=infer_schema_rows,
    )


# Bumped when the staged files' layout changes, so older staged folders are
# rebuilt rather than misread.
_STAGE_FORMAT = 1
_STAGE_MANIFEST = "_manifest.json"
_STAGE_TEXT_EXTS = (".csv", ".txt", ".dat", ".psv", ".tsv")


def stage_csv_parquet(
    path: str,
    separator: str = ",",
    has_header: bool = True,
    column_names: str = "",
    infer_schema_rows: int = 10000,
    compression: str = "zstd",
    parquet_dir: str = "",
    columns: str = "",
) -> pl.LazyFrame:
    """Converts large delimited text files to Parquet once, then reads the
    Parquet on the `out` port; no inputs. The way to bring a big CSV source
    in -- never a staging script of your own. `path` is a .csv/.txt file,
    a .gz of one, a .zip holding one or more of them, a folder (every such
    file in it, e.g. a set of quarterly zips) or a glob ("data/*.zip");
    each file, and each member of a zip, becomes its own Parquet part.
    `columns` (comma-separated, after `column_names` naming) keeps only
    those columns -- stage a slim table of a wide file. A zip member or gzipped file is decompressed
    straight from the archive in blocks of about 8 MB, each parsed and
    streamed into the Parquet: nothing is unpacked to disk and neither the
    file nor the archive is ever held in memory whole. `separator`,
    `has_header`, `column_names` and `infer_schema_rows` work as in Read
    CSV; dtypes are inferred from the first part converted and every other
    part is read with that schema, so all parts line up -- use 0 (all
    text, cast afterwards) when early rows aren't representative, or when
    codes with leading zeros (e.g. MMYYYY dates) must survive.
    `compression` is the Parquet codec. The Parquet goes into
    `parquet_dir` (default: a "<file name>_parquet" folder next to the
    file) with a _manifest.json recording each part's source hash (the
    zip's CRC-32 per member; SHA-256 for a plain or gzipped file, skipped
    while its size and modification time are unchanged) and the read
    options: a part is rebuilt only when its source or the options
    changed, so later runs go straight to the Parquet."""
    import os

    if separator in ("\\t", "tab", "TAB"):
        separator = "\t"
    if len(separator) != 1:
        raise ValueError(f"separator must be a single character, got {separator!r}")
    if compression not in PARQUET_COMPRESSIONS:
        raise ValueError(f"compression must be one of {', '.join(PARQUET_COMPRESSIONS)}, got {compression!r}")
    files = _stage_files(path)
    names = [n.strip() for n in column_names.split(",") if n.strip()] if column_names else None
    keep = [c.strip() for c in columns.split(",") if c.strip()] if columns else None
    options = {
        "separator": separator,
        "has_header": bool(has_header),
        "column_names": names,
        "infer_schema_rows": int(infer_schema_rows),
        "compression": compression,
    }
    if keep:
        options["columns"] = keep  # only when set, so stages made before it existed stay current
    target = parquet_dir or _stage_default_dir(path)
    os.makedirs(target, exist_ok=True)
    manifest_path = os.path.join(target, _STAGE_MANIFEST)
    manifest = _stage_read_manifest(manifest_path)
    previous = set(manifest.get("parts", {}))  # every part on disk, for the clean-up at the end
    if manifest.get("format") != _STAGE_FORMAT or manifest.get("options") != options:
        manifest = {"format": _STAGE_FORMAT, "options": options, "parts": {}}
    old_parts = manifest["parts"]
    units = [u for file in files for u in _stage_units(file, old_parts, prefix=len(files) > 1)]
    if not units:
        raise ValueError(f"{path!r} holds no delimited text file to stage")
    if len({u["part"] for u in units}) != len(units):
        raise ValueError(f"two of the files in {path!r} would stage to the same part name -- stage them separately")

    current = [
        u["part"]
        for u in units
        if old_parts.get(u["part"], {}).get("hash") == u["hash"] and os.path.isfile(os.path.join(target, u["part"]))
    ]
    # Every part shares one schema: an up-to-date part's, else the first
    # part built infers it.
    schema = pl.read_parquet_schema(os.path.join(target, current[0])) if current else None
    parts: dict[str, dict] = {}
    for u in units:
        entry = {k: u[k] for k in ("source", "hash", "size", "mtime_ns") if k in u}
        if u["part"] in current:
            parts[u["part"]] = entry
            continue
        out = os.path.join(target, u["part"])
        _stage_build(u["file"], u, out, options, schema)
        if schema is None:
            schema = pl.read_parquet_schema(out)
        parts[u["part"]] = entry
        # Saved after every part, so an interrupted run keeps what it did.
        manifest["parts"] = {**{p: e for p, e in old_parts.items() if p in current}, **parts}
        _stage_write_manifest(manifest_path, manifest)
    for stale in previous - set(parts):
        try:
            os.remove(os.path.join(target, stale))
        except OSError:
            pass
    manifest["parts"] = parts
    _stage_write_manifest(manifest_path, manifest)
    return pl.scan_parquet([os.path.join(target, u["part"]) for u in units])


def _stage_files(path: str) -> list[str]:
    """The files `path` names: itself, every stageable file in a folder,
    or a glob's matches -- sorted, so parts keep a stable order."""
    import glob
    import os

    exts = (".zip", ".gz", *_STAGE_TEXT_EXTS)
    if os.path.isdir(path):
        found = [os.path.join(path, n) for n in os.listdir(path) if n.lower().endswith(exts)]
        files = sorted(f for f in found if os.path.isfile(f))
        if not files:
            raise FileNotFoundError(f"no .csv/.txt/.zip/.gz files in the folder {path!r}")
        return files
    if any(c in path for c in "*?["):
        files = sorted(f for f in glob.glob(path) if os.path.isfile(f))
        if not files:
            raise FileNotFoundError(f"nothing matches {path!r}")
        return files
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no file at {path!r}")
    return [path]


def _stage_default_dir(path: str) -> str:
    import os

    if any(c in path for c in "*?["):
        return os.path.join(os.path.dirname(os.path.abspath(path)), "staged_parquet")
    path = path.rstrip("/\\")
    base = os.path.basename(path)
    for ext in (".gz", ".zip", *_STAGE_TEXT_EXTS):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
    return os.path.join(os.path.dirname(os.path.abspath(path)), f"{base}_parquet")


def _stage_part_name(name: str) -> str:
    import re

    stem = name
    for ext in (".gz", *_STAGE_TEXT_EXTS):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem.replace("/", "__")).strip("_") + ".parquet"


def _stage_units(path: str, old_parts: dict, prefix: bool = False) -> list[dict]:
    """The text files `path` holds, one per Parquet part, each with a hash
    of its content: a zip member's CRC-32 and size from the zip's own
    directory (no decompressing), else the file's SHA-256 -- reused from
    the manifest while the file's size and mtime are unchanged. With
    `prefix` (one of several files), a zip member's part is named after
    the zip too, unless the member is named like it (2016Q1.zip holding
    2016Q1.csv stages to 2016Q1.parquet)."""
    import hashlib
    import os
    import zipfile

    if path.lower().endswith(".zip"):
        zip_stem = _stage_part_name(os.path.basename(path)[:-4])[: -len(".parquet")]

        def _part(member: str) -> str:
            name = _stage_part_name(member)
            return name if not prefix or name == f"{zip_stem}.parquet" else f"{zip_stem}__{name}"

        with zipfile.ZipFile(path) as zf:
            members = [i for i in zf.infolist() if not i.is_dir() and not i.filename.startswith("__MACOSX/")]
            text = [i for i in members if i.filename.lower().endswith(_STAGE_TEXT_EXTS)]
            return [
                {
                    "part": _part(i.filename),
                    "file": path,
                    "member": i.filename,
                    "source": f"{os.path.basename(path)}:{i.filename}",
                    "hash": f"crc32:{i.CRC:08x}:{i.file_size}",
                }
                for i in (text or members)
            ]
    st = os.stat(path)
    part = _stage_part_name(os.path.basename(path))
    old = old_parts.get(part, {})
    if old.get("size") == st.st_size and old.get("mtime_ns") == st.st_mtime_ns and old.get("hash"):
        digest = old["hash"]
    else:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
                h.update(chunk)
        digest = f"sha256:{h.hexdigest()}"
    return [
        {
            "part": part,
            "file": path,
            "source": os.path.basename(path),
            "hash": digest,
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
        }
    ]


# How much decompressed text a zipped/gzipped source is parsed in at a
# time. Memory follows the block size, not the file's: a block's parsed
# rows take several times its text (a string value costs ~16 bytes, even an
# empty one) and the streaming engine keeps a few blocks in flight. Measured
# on a 31-column all-text file: 64 MB blocks peaked at ~1.35 GB, 8 MB at
# ~355 MB (for 1.2 and 2.4 GB of text alike), at ~10% more run time.
_STAGE_CHUNK_BYTES = 8 * 1024 * 1024


def _stage_build(path: str, unit: dict, out: str, options: dict, schema) -> None:
    """Converts one unit to `out` with the streaming engine, via `out`.tmp
    so a half-written part is never taken for a finished one. A plain file
    is scanned directly. A zip member or gzipped file is decompressed
    straight from the archive, a block of whole lines at a time (see
    _stage_text_blocks), each block parsed and streamed into the Parquet
    -- nothing is unpacked to disk and the text is never held whole."""
    import os

    compressed = "member" in unit or path.lower().endswith(".gz")
    try:
        if compressed:
            lf = _stage_compressed_frame(path, unit, options, schema)
        else:
            lf = pl.scan_csv(path, **_stage_csv_kwargs(options, schema))
        if options.get("columns"):
            lf = lf.select(options["columns"])
        lf.sink_parquet(out + ".tmp", compression=options["compression"])
        os.replace(out + ".tmp", out)
    except pl.exceptions.PolarsError as e:
        hint = (
            " -- it is read with the schema of the first part; set infer_schema_rows to 0 to stage every column as text"
            if schema is not None
            else ""
        )
        raise ValueError(f"staging {unit['source']} failed: {e}{hint}") from e
    finally:
        if os.path.exists(out + ".tmp"):
            os.remove(out + ".tmp")


def _stage_csv_kwargs(options: dict, schema) -> dict:
    """How a part's text is parsed. The first part infers dtypes; every
    later one reads every column as text and casts the staged ones to the
    first part's Parquet schema, so the parts line up -- and columns not
    staged (see `columns`) never fail a parse."""
    kwargs: dict = {
        "separator": options["separator"],
        "has_header": options["has_header"],
        "new_columns": options["column_names"],
    }
    if schema is None:
        kwargs["infer_schema_length"] = options["infer_schema_rows"]
    else:
        kwargs["infer_schema_length"] = 0
        kwargs["schema_overrides"] = dict(schema)
    return kwargs


def _stage_open(path: str, unit: dict):
    """A binary stream of the unit's decompressed text, read straight from
    the archive."""
    import gzip
    import zipfile

    if "member" in unit:
        zf = zipfile.ZipFile(path)
        stream = zf.open(unit["member"])
        close = stream.close

        def _close_both() -> None:
            close()
            zf.close()

        stream.close = _close_both
        return stream
    return gzip.open(path, "rb")


def _stage_text_blocks(stream, size: int):
    """Yields the stream's bytes in blocks of about `size`, each ending at a
    record boundary: on a newline outside double quotes (a quoted field may
    hold line breaks; with "" as the escape, an even quote count before the
    newline means it is outside one). Each block starts at a boundary, so
    the count starts afresh."""
    rest = b""
    while True:
        data = stream.read(size)
        if not data:
            if rest.strip():
                yield rest
            return
        block = rest + data
        cut = block.rfind(b"\n")
        while cut >= 0 and block.count(b'"', 0, cut) % 2:
            cut = block.rfind(b"\n", 0, cut)
        if cut < 0:
            rest = block  # no boundary yet: read on
            continue
        yield block[: cut + 1]
        rest = block[cut + 1 :]


def _stage_compressed_frame(path: str, unit: dict, options: dict, schema) -> pl.LazyFrame:
    """A LazyFrame over a zip member or gzipped file that parses it block
    by block as the streaming engine pulls (polars' IO-source plugin), so
    sink_parquet writes it without it ever being unpacked or held whole.
    Without a `schema`, the first block's inferred one is used for all.

    register_io_source is marked unstable by polars (it may change in any
    release); test_large_data covers this path, so an upgrade that breaks
    it shows up there."""
    try:
        from polars.io.plugins import register_io_source
    except ImportError as e:
        raise RuntimeError(
            f"staging a zipped or gzipped file needs polars' IO-source plugin (polars.io.plugins.register_io_source), "
            f"which this polars ({pl.__version__}) doesn't have -- upgrade polars, or unzip the file and stage the CSV"
        ) from e

    def _parse(block: bytes, first: bool, schema) -> pl.DataFrame:
        has_header = first and options["has_header"]
        if schema is not None:
            # This part's own full schema (see below): every block alike.
            return pl.read_csv(block, separator=options["separator"], has_header=has_header, schema=schema)
        return pl.read_csv(block, **{**_stage_csv_kwargs(options, part_schema), "has_header": has_header})

    # The schema this part is parsed with: its first block's, which follows
    # the shared one (`part_schema`, the first part's) for staged columns.
    part_schema = schema
    with _stage_open(path, unit) as stream:
        first = next(_stage_text_blocks(stream, _STAGE_CHUNK_BYTES), None)
    if first is None:
        raise ValueError(f"{unit['source']} is empty")
    schema = _parse(first, True, None).schema

    def _source(with_columns, predicate, n_rows, batch_size):
        left = n_rows
        with _stage_open(path, unit) as stream:
            for i, block in enumerate(_stage_text_blocks(stream, _STAGE_CHUNK_BYTES)):
                df = _parse(block, i == 0, schema)
                if with_columns is not None:
                    df = df.select(with_columns)
                if predicate is not None:
                    df = df.filter(predicate)
                if left is not None:
                    df = df.head(left)
                    left -= df.height
                yield df
                if left is not None and left <= 0:
                    return

    return register_io_source(_source, schema=schema)


def _stage_read_manifest(manifest_path: str) -> dict:
    import json

    try:
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) and isinstance(data.get("parts"), dict) else {}


def _stage_write_manifest(manifest_path: str, manifest: dict) -> None:
    import json
    import os

    with open(manifest_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(manifest_path + ".tmp", manifest_path)


def demo_credit_data(n_rows: int = 5000, seed: int = 42) -> pl.DataFrame:
    """Generates the synthetic retail credit-risk demo dataset (term loans
    and credit cards, one row per facility) on the `out` port; no inputs,
    no file. Supports PD (`default_flag`, a 12-month default), LGD
    (`balance_at_default`, `recovery_amount`, `workout_cost` on defaults --
    see compute_lgd) and CCF (`credit_limit`, `balance_at_reference`,
    `balance_at_default` on defaulted credit cards -- see compute_ccf).
    `n_rows` is the number of records; `seed` fixes the data, so the same
    seed always gives the same rows, and a smaller `n_rows` gives the first
    rows of the larger dataset."""
    from modelmaker import demo_data

    if n_rows < 1:
        raise ValueError(f"n_rows must be at least 1, got {n_rows}")
    return demo_data.to_frame(n_rows, seed)


def read_parquet(path: str) -> pl.LazyFrame:
    """Reads a Parquet file into a dataframe on the `out` port; no inputs.
    `path` is the file path. Dtypes come from the Parquet schema."""
    return pl.scan_parquet(path)


def read_json(path: str) -> pl.DataFrame:
    """Reads a JSON file into a dataframe on the `out` port; no inputs.
    `path` ending in .jsonl/.ndjson (case-insensitive) is read as
    newline-delimited JSON (one object per line); anything else must be a
    plain JSON array of records."""
    # A plain JSON array has no lazy reader (polars must load it whole to
    # find its structure), so this block returns an eager frame and never
    # joins a streaming run's fusion.
    if path.lower().endswith((".jsonl", ".ndjson")):
        return pl.read_ndjson(path)
    return pl.read_json(path)


def read_excel(path: str, sheet: str | None = None) -> pl.DataFrame:
    """Reads one worksheet of an Excel workbook into a dataframe on the
    `out` port; no inputs. `path` is the .xlsx file path; `sheet` is the
    worksheet name -- unset/empty reads the first sheet. The first row is
    taken as the header."""
    return pl.read_excel(path, sheet_name=sheet) if sheet else pl.read_excel(path)


def read_sql(connection_env: str, query: str) -> pl.DataFrame:
    """Runs a SQL query against a database and returns the result set as a
    dataframe on the `out` port; no inputs. `connection_env` is the *name*
    of an environment variable (e.g. "WAREHOUSE_URI") holding the
    connection URI -- never the URI itself; raises KeyError if that
    variable isn't set. `query` is the SQL text, in the target database's
    own dialect. Set the block's optional `probe_query` (a cheap query such
    as SELECT MAX(updated_at) ...) to let "check for changes" detect new
    data."""
    import os

    # The connection string itself never lives in params -- it's baked as a
    # literal into saved project files and compiled scripts, so a raw DB
    # password there would leak into both. The secret is read fresh from
    # the environment at run time, live or compiled.
    uri = os.environ[connection_env]
    return pl.read_database_uri(query, uri)


def read_dataset(dataset: str, sheet: str | None = None) -> pl.LazyFrame:
    """Reads one of the workspace's datasets into a dataframe on the `out`
    port; no inputs. `dataset` is the dataset's name, picked from the
    workspace's datasets; before the block runs, the engine resolves it to
    the dataset's current file (read-only) and passes that path in its
    place -- so replacing the dataset re-runs the block. The file's format
    follows its extension: CSV (dtypes inferred), Parquet, Arrow IPC, JSON
    (a plain array of records), NDJSON or Excel (`sheet` names the
    worksheet; unset reads the first)."""
    lower = dataset.lower()
    if lower.endswith(".csv"):
        return pl.scan_csv(dataset)
    if lower.endswith(".parquet"):
        return pl.scan_parquet(dataset)
    if lower.endswith((".arrow", ".ipc", ".feather")):
        return pl.scan_ipc(dataset)
    if lower.endswith((".ndjson", ".jsonl")):
        return pl.scan_ndjson(dataset)
    if lower.endswith(".json"):
        return pl.read_json(dataset).lazy()
    if lower.endswith(".xlsx"):
        return (pl.read_excel(dataset, sheet_name=sheet) if sheet else pl.read_excel(dataset)).lazy()
    raise ValueError(f"can't read {dataset!r}: unknown dataset format")


def read_source(source: str, query: str) -> pl.LazyFrame:
    """Runs a SQL query against one of the data sources the administrator
    set up and returns the result set as a dataframe on the `out` port; no
    inputs. `source` is the source's name; `query` is the SQL text, in that
    database's own dialect. The query runs in the application, with the
    administrator's credentials, never in the block's own process: the
    engine stores its result as a dataset and passes that file in place of
    `source`. Refresh the block to run the query again."""
    # `query` has already run (see Runner._resolve_data); it's a parameter
    # so that it's part of the cache key and shown in the form.
    del query
    return pl.scan_parquet(source)


def filter(df: pl.DataFrame, expr: str) -> pl.DataFrame:
    """Keeps only the rows of `df` for which `expr` is true, emitted on
    `out` with the same columns and roles. `expr` is a Polars SQL boolean
    expression over column names, e.g. "age >= 18 AND region = 'EU'" or
    "balance IS NOT NULL" (string literals in single quotes)."""
    return df.filter(pl.sql_expr(expr))


def select(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Keeps only the listed columns of `df`, in the order given by `cols`,
    emitted on `out`; kept columns retain their roles. Every name in `cols`
    must exist in the input."""
    return df.select(cols)


def groupby_agg(df: pl.DataFrame, by: list[str], aggs: dict[str, str | list[str]]) -> pl.DataFrame:
    """Groups `df` by the `by` key columns and aggregates, one row per
    group on `out`. `aggs` maps a column name to the name of a polars
    expression method applied to it, e.g. {"balance": "sum", "pd":
    "mean"}; valid names include sum, mean, median, min, max, std, var,
    count, n_unique, first, last. A single aggregation keeps the column's
    own name; a list of them (e.g. {"lgd": ["mean", "count", "std"]})
    gives one column per aggregation, named `<column>_<agg>`. Only `by`
    plus the aggregated columns appear in the output. Key columns keep
    their roles; aggregated columns lose theirs. Output row order is not
    guaranteed."""
    agg_exprs = []
    for c, fn in aggs.items():
        if isinstance(fn, (list, tuple)):
            agg_exprs += [getattr(pl.col(c), f)().alias(f"{c}_{f}") for f in fn]
        else:
            agg_exprs.append(getattr(pl.col(c), fn)().alias(c))
    return df.group_by(by).agg(agg_exprs)


def join(left: pl.DataFrame, right: pl.DataFrame, on: list[str], how: str = "inner") -> pl.DataFrame:
    """Joins the `left` and `right` input dataframes on the key columns in
    `on` (same names on both sides), emitted on `out`. `how` is a polars
    join strategy: "inner" (default), "left", "right", "full", "semi", or
    "anti" ("cross" is not usable: polars rejects join keys with it).
    Non-key columns present on both sides get a "_right" suffix on the
    right-hand copy; a "full" join also keeps the right-hand keys as
    "<key>_right". Where a column exists on both sides, the left side's
    role wins. "outer" (polars' old name for "full") still works, for
    projects saved before the rename."""
    if how == "outer":
        how = "full"
    if how == "cross":
        raise ValueError('how="cross" is not supported: polars rejects join keys with it')
    return left.join(right, on=on, how=how)


def train_test_split(
    df: pl.DataFrame,
    test_size: float = 0.2,
    seed: int = 0,
    stratify_col: str | None = None,
    group_col: str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Randomly splits `df`'s rows into two disjoint dataframes on the
    `train` and `test` ports; both keep every column and role. `test_size`
    is the fraction of rows sent to `test` (0-1, default 0.2; the test row
    count is rounded down), the rest go to `train`. `seed` fixes the
    shuffle, so the same seed gives the same split.

    `stratify_col` (e.g. the default flag, or a segment) splits each of its
    values separately, so `train` and `test` get the same share of each --
    the usual choice for a low default rate, where a plain random split can
    leave the test sample with a noticeably different event rate. Left
    unset, it's a simple random split.

    `group_col` (e.g. loan_id) keeps all of an ID's rows on one side: the
    IDs are split, not the rows, so panel data (one row per loan per
    snapshot) never has the same loan in train and test. `test_size` is
    then the fraction of IDs. With `stratify_col` too, each ID is
    stratified by its highest value of that column (for a 0/1 default
    flag: whether the loan ever defaulted), or its first for text."""
    if stratify_col:
        _check_split_column(df, stratify_col, "stratify_col")
    if group_col:
        return _split_by_group(df, test_size, seed, stratify_col, group_col)
    if stratify_col is None:
        shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)
        n_test = int(len(shuffled) * test_size)
        test = shuffled.head(n_test)
        train = shuffled.tail(len(shuffled) - n_test)
        return train, test
    trains, tests = [], []
    for _, part in sorted(df.group_by(stratify_col, maintain_order=True), key=lambda kv: str(kv[0])):
        shuffled = part.sample(fraction=1.0, shuffle=True, seed=seed)
        n_test = int(len(shuffled) * test_size)
        tests.append(shuffled.head(n_test))
        trains.append(shuffled.tail(len(shuffled) - n_test))
    return pl.concat(trains), pl.concat(tests)


def _split_by_group(
    df: pl.DataFrame, test_size: float, seed: int, stratify_col: str | None, group_col: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """train_test_split with `group_col`: the IDs are shuffled and split
    (within each stratum), then every row follows its ID. Rows with a null
    ID count as one ID. Both sides keep the input's row order."""
    _check_split_column(df, group_col, "group_col")
    # Each ID as an integer (null -> 0), so the split needs no null-aware
    # join -- which older polars versions spell differently.
    gid = pl.col(group_col).rank("dense").fill_null(0).alias("_gid")
    work = df.select(gid, *([stratify_col] if stratify_col else []))
    if stratify_col:
        dtype = df.schema[stratify_col]
        by_max = dtype.is_numeric() or dtype == pl.Boolean
        stratum = (pl.col(stratify_col).max() if by_max else pl.col(stratify_col).first()).alias("_stratum")
        ids = work.group_by("_gid").agg(stratum)
    else:
        ids = work.select(pl.col("_gid").unique()).with_columns(_stratum=pl.lit(0))
    ids = ids.sort("_gid")  # a stable order for the seed
    test_ids: list[int] = []
    for _, part in sorted(ids.group_by("_stratum", maintain_order=True), key=lambda kv: str(kv[0])):
        shuffled = part.sample(fraction=1.0, shuffle=True, seed=seed)
        test_ids.extend(shuffled.head(int(len(shuffled) * test_size))["_gid"].to_list())
    in_test = work["_gid"].is_in(pl.Series(test_ids, dtype=work["_gid"].dtype))
    return df.filter(~in_test), df.filter(in_test)


def _check_split_column(df: pl.DataFrame, col: str, param: str) -> None:
    if col not in df.columns:
        raise ValueError(f"{param} {col!r} is not a column of the input")


def time_split(
    df: pl.DataFrame, date_col: str, cutoff: str, oot_end: str | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Out-of-time split on a date column: rows dated before `cutoff`
    (an ISO date, "YYYY-MM-DD") go to `development`, rows on or after it to
    `out_of_time` -- the standard way to hold back the most recent period
    for validation, instead of two hand-written filters. `oot_end`, when
    set, also drops rows on/after that date from `out_of_time` (e.g. a
    period whose outcome window isn't complete yet). `date_col` may be a
    Date/Datetime column or ISO date strings. Rows with a null date go to
    neither output. Both outputs keep every column and role."""
    from datetime import date

    dtype = df.schema[date_col]
    if dtype == pl.Utf8:
        d = pl.col(date_col).str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False)
    elif isinstance(dtype, pl.Datetime):
        d = pl.col(date_col).dt.date()
    else:
        d = pl.col(date_col).cast(pl.Date)
    cut = date.fromisoformat(cutoff)
    development = df.filter(d < cut)
    oot_mask = d >= cut
    if oot_end is not None:
        oot_mask = oot_mask & (d < date.fromisoformat(oot_end))
    out_of_time = df.filter(oot_mask)
    return development, out_of_time


def derive_columns(df: pl.DataFrame, expressions: dict[str, str]) -> pl.DataFrame:
    """Adds (or replaces) columns computed from SQL expressions over the
    existing columns -- ratios, flags, differences, bands -- the everyday
    feature engineering that otherwise needs a custom code block.
    `expressions` maps each new column name to a Polars SQL expression,
    e.g. {"utilisation": "balance / NULLIF(credit_limit, 0)",
    "headroom": "credit_limit - balance", "is_secured": "collateral_value > 0",
    "age_band": "CASE WHEN age < 30 THEN 'young' ELSE 'other' END"}.
    Expressions are evaluated against the input columns (not each other),
    in one step. New columns are tagged role=feature; a replaced column
    keeps its role."""
    return df.with_columns([pl.sql_expr(expr).alias(name) for name, expr in expressions.items()])


def one_hot_encode(
    df: pl.DataFrame, columns: list[str], drop_first: bool = True, max_categories: int = 30, keep_original: bool = False
) -> pl.DataFrame:
    """Dummy-encodes categorical `columns` into 0/1 indicator columns named
    `<column>_<value>` -- what a regression (logistic, GLM, LGD/CCF
    fractional logit) needs for a categorical driver like product type or
    seniority. `drop_first` (default) leaves out the alphabetically first
    category of each column as the reference level (always the same one,
    whatever the row order), so the dummies aren't collinear with the
    intercept. A null gets its own `<column>_null` indicator. Refuses a
    column with more than `max_categories` distinct values (almost
    certainly an id -- use fit_binning for a high-cardinality driver).
    `keep_original` keeps the source columns; by default they're dropped.

    The category set comes from the data it's run on, so encode before any
    train/test split to get the same dummies on both sides."""
    for c in columns:
        n = df[c].n_unique()
        if n > max_categories:
            raise ValueError(f"'{c}' has {n} distinct values (> max_categories={max_categories}) -- not a categorical to one-hot encode")
    dummies = []
    for c in columns:
        values = df[c].cast(pl.Utf8)
        levels = sorted(values.drop_nulls().unique().to_list())
        if drop_first and levels:
            levels = levels[1:]
        dummies += [(values == level).fill_null(False).cast(pl.Int8).alias(f"{c}_{level}") for level in levels]
        if values.null_count():
            dummies.append(values.is_null().cast(pl.Int8).alias(f"{c}_null"))
    base = df if keep_original else df.drop(columns)
    return base.hstack(dummies)


def write_csv(df: pl.DataFrame, filename: str, output_dir: str = ".") -> None:
    """Writes the `df` input to a CSV file; no output ports. `filename` is
    the file name (e.g. "scored.csv"), written inside `output_dir`, which
    the engine injects as the run's output directory -- leave it unset.
    Overwrites any existing file of the same name."""
    import os

    # .lazy() is a no-op on a LazyFrame: in a streaming run this block is
    # handed its upstream's uncollected plan and writes it straight to disk.
    df.lazy().sink_csv(os.path.join(output_dir, filename))


PARQUET_COMPRESSIONS = ("zstd", "snappy", "lz4", "gzip", "uncompressed")


def write_parquet(df: pl.DataFrame, filename: str, compression: str = "zstd", output_dir: str = ".") -> None:
    """Writes the `df` input to a Parquet file; no output ports. `filename`
    is the file name (e.g. "loans.parquet"), written inside `output_dir`,
    which the engine injects as the run's output directory -- leave it
    unset. `compression` is "zstd" (default), "snappy", "lz4", "gzip" or
    "uncompressed". Overwrites any existing file of the same name. In a
    streaming run (Run all (streaming)) the data goes straight from its
    source to disk without ever being held in memory whole -- the way to
    convert a large CSV to Parquet once, so later runs read only the
    columns they use."""
    import os

    if compression not in PARQUET_COMPRESSIONS:
        raise ValueError(f"compression must be one of {', '.join(PARQUET_COMPRESSIONS)}, got {compression!r}")
    # As write_csv: .lazy() is a no-op on a LazyFrame, so a streaming run
    # hands this its upstream's uncollected plan and it sinks to disk.
    df.lazy().sink_parquet(os.path.join(output_dir, filename), compression=compression)


EXPORT_FORMATS = ("csv", "parquet")


def export(df: pl.DataFrame, format: str = "csv", name: str = "") -> pl.DataFrame:
    """Exports the `df` input as a file the user downloads -- `format` is
    "csv" or "parquet", `name` the export's name (default: the block's
    name) -- and passes `df` on unchanged on `out`. The export is made by
    the application from the block's result after each successful run (see
    datasets.ExportSink); in sample mode it holds only the sampled rows."""
    if format not in EXPORT_FORMATS:
        raise ValueError(f"format must be one of {', '.join(EXPORT_FORMATS)}, got {format!r}")
    return df


def display_table(
    df: pl.DataFrame, save_csv: bool = True, filename: str = "", output_dir: str = ".", block_id: str = "table"
) -> pl.DataFrame:
    """Shows a results table right on its canvas node -- the first rows and
    the row count, updated whenever its upstream runs -- and passes `df` on
    unchanged on `out`. With `save_csv` on (the default) it also writes the
    whole table to `filename` (default `<block id>.csv`) in `output_dir`,
    the run's output directory, which the engine injects -- leave it unset.
    Use it for the tables worth reporting, e.g. a one-row RWA total."""
    if save_csv:
        _save_display_csv(df, filename, output_dir, block_id)
    return df


def display_value(value, save_csv: bool = True, filename: str = "", output_dir: str = ".", block_id: str = "value"):
    """Shows whatever comes in right on its canvas node and passes it on
    unchanged: a dataframe (first rows and row count), a metric or model
    dict (its keys and values) or an image. Its ports are typed "any" so it
    wires up to whatever you point it at. A dataframe is also saved as CSV
    like display_table does (`save_csv`, `filename`, `output_dir`)."""
    if save_csv and isinstance(value, pl.DataFrame):
        _save_display_csv(value, filename, output_dir, block_id)
    return value


def _save_display_csv(df: pl.DataFrame, filename: str, output_dir: str, block_id: str) -> None:
    import os

    os.makedirs(output_dir, exist_ok=True)
    df.write_csv(os.path.join(output_dir, filename or f"{block_id}.csv"))


_KPI_AGGS = ("first", "last", "sum", "mean", "min", "max", "count")


def display_kpi(
    value,
    column: str = "",
    agg: str = "first",
    label: str = "",
    number_format: str = "number",
    decimals: int = 2,
    unit: str = "",
) -> dict:
    """Shows one headline number, with a label, right on the canvas -- e.g.
    the total RWA at the end of a capital pipeline, or a model's Gini.
    `value` is a dataframe (its `column`, reduced over the rows by `agg`:
    first, last, sum, mean, min, max or count), a metric dict such as
    auc_gini's (its `column` key) or a plain number. Leave `column` empty
    when there is only one numeric column or key. `label` defaults to the
    column name; `number_format` (number, compact like 1.2M, percent),
    `decimals` and `unit` only change how the number is shown. Returns
    {"label", "value", "column", "agg", "number_format", "decimals", "unit"}
    on `kpi`."""
    import math
    import numbers

    if agg not in _KPI_AGGS:
        raise ValueError(f"agg must be one of {', '.join(_KPI_AGGS)}, got {agg!r}")
    if isinstance(value, pl.DataFrame):
        if not column:
            numeric = [c for c, t in value.schema.items() if t.is_numeric()]
            if len(numeric) != 1:
                raise ValueError(f"set `column`: the table has {len(numeric)} numeric columns ({', '.join(numeric)})")
            column = numeric[0]
        if column not in value.columns:
            raise ValueError(f"no column {column!r}; columns are {', '.join(value.columns)}")
        s = value[column]
        if agg == "count":
            number = s.drop_nulls().len()
        elif s.is_empty():
            number = None
        else:
            number = s[0] if agg == "first" else s[-1] if agg == "last" else getattr(s, agg)()
    elif isinstance(value, dict):
        if not column:
            numeric = [k for k, v in value.items() if isinstance(v, numbers.Real) and not isinstance(v, bool)]
            if len(numeric) != 1:
                raise ValueError(f"set `column` to one of the metric's keys: {', '.join(map(str, value))}")
            column = numeric[0]
        if column not in value:
            raise ValueError(f"no key {column!r}; keys are {', '.join(map(str, value))}")
        number = value[column]
    elif isinstance(value, numbers.Real):
        number = value
    else:
        raise TypeError(f"display_kpi needs a dataframe, a metric dict or a number, got {type(value).__name__}")
    if number is not None and not isinstance(number, numbers.Real):
        raise TypeError(f"{column!r} isn't a number: {number!r}")
    if isinstance(number, float) and not math.isfinite(number):
        number = None
    return {
        "label": label or column or "value",
        "value": number if number is None or isinstance(number, int) else float(number),
        "column": column,
        "agg": agg,
        "number_format": number_format,
        "decimals": int(decimals),
        "unit": unit,
    }


def _example_display_kpi():
    df = pl.DataFrame({"segment": ["retail", "sme"], "rwa": [120.0, 80.0]})
    assert display_kpi(df, column="rwa", agg="sum", label="Total RWA")["value"] == 200.0
    assert display_kpi({"gini": 0.42, "n": 1000}, column="gini")["label"] == "gini"


def generate_image(
    df: pl.DataFrame,
    kind: str = "hist",
    x: str = "",
    y: str = "",
    bins: int = 30,
    title: str = "",
    output_dir: str = ".",
    block_id: str = "",
) -> bytes:
    """Draws a matplotlib chart from the `df` input and emits it as PNG
    bytes on the `image` port, also saving it as
    generate_image_<block_id>.png in `output_dir`. `kind` is one of
    "hist" (default; histogram of column `x` with `bins` bins, `y`
    unused), "bar" (x = category labels, y = bar heights -- one bar per
    row, so aggregate first), "scatter", or "line" (both plot `y` against
    `x` in row order). `x` is always required; `y` is required for every
    kind but hist. `title` is optional. `output_dir` and `block_id` are
    engine-injected; leave them unset."""
    # Self-contained imports (rather than relying on this module's own
    # top-level imports) so this function stays a valid, independent unit
    # both when run live by the engine and when inlined verbatim into a
    # compiled script.
    import io
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    if kind == "hist":
        ax.hist(df[x].to_list(), bins=bins)
    elif kind == "bar":
        ax.bar([str(v) for v in df[x].to_list()], df[y].to_list())
    elif kind == "scatter":
        ax.scatter(df[x].to_list(), df[y].to_list(), s=10, alpha=0.6)
    elif kind == "line":
        ax.plot(df[x].to_list(), df[y].to_list())
    else:
        raise ValueError(f"unknown chart kind: {kind!r}")
    ax.set_xlabel(x)
    if y:
        ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)

    os.makedirs(output_dir, exist_ok=True)
    filename = f"generate_image_{block_id}.png" if block_id else "generate_image.png"
    with open(os.path.join(output_dir, filename), "wb") as f:
        f.write(buf.getvalue())

    return buf.getvalue()
