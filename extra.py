import argparse
import ast
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
except Exception as e:
    print("This script requires pandas. Please install it: pip install pandas", file=sys.stderr)
    raise

try:
    import phonenumbers  # type: ignore
    HAS_PHONES = True
except Exception:
    HAS_PHONES = False

try:
    import requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

try:
    from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    HAS_KAGGLE = True
except Exception:
    HAS_KAGGLE = False

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
PHONE_RE = re.compile(r"^\+?\d[\d \-\(\)]{7,}\d$")
COORD_PAIR_RE = re.compile(r"""^\s*\[?\s*
    -?\d{1,3}(?:\.\d+)?\s*[,; ]\s*
    -?\d{1,3}(?:\.\d+)?\s*
    \]?\s*$""", re.X)
ADDRESS_HINT_RE = re.compile(r"(street|st\.|st,|avenue|ave\.|road|rd\.|ул\.|просп\.|пр-кт|дом|д\.)", re.I)
PRODUCT_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-_\.]{3,29}$")

TRUE_SET = {"true","t","yes","y","1","да","oui","sí","evet","نعم"}
FALSE_SET = {"false","f","no","n","0","нет","non","não","hayır","لا"}

DATE_FORMAT_HINTS = [
    "%Y-%m-%d",
    "%d.%m.%Y",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%Y/%m/%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d.%m.%Y %H:%م:%S",
    "%d.%m.%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
]

KNOWN_SENSE_OVERRIDES: Dict[str, str] = {
    "amount": "amount",
    "currency": "currency",
    "country": "country",
    "ip_address": "ip_address",
    "card_number": "card_number",
    "transaction_hour": "numeric_hour",
}


def try_parse_date(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    try:
        datetime.fromisoformat(s.replace("Z","").replace("z",""))
        return True
    except Exception:
        pass
    for fmt in DATE_FORMAT_HINTS:
        try:
            datetime.strptime(s, fmt)
            return True
        except Exception:
            continue
    if re.search(r"\d{4}", s) and re.search(r"[\-/.: ]", s) and len(s) <= 32:
        return True
    return False

def is_boolish(x: Any) -> Optional[bool]:
    if isinstance(x, bool):
        return True
    if isinstance(x, (int, float)) and (x in (0,1) or (isinstance(x, float) and x in (0.0,1.0))):
        return True
    if isinstance(x, str):
        s = x.strip().lower()
        if s in TRUE_SET or s in FALSE_SET:
            return True
    return False

def parse_list_like(s: str) -> Optional[List[Any]]:
    s = s.strip()
    if not s:
        return None
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple)):
                return list(val)
        except Exception:
            return None
    if "," in s and not s.lower().startswith("http"):
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 2 and all(parts):
            return parts
    return None

def inner_type_of_list(values: List[Any]) -> str:
    counters = Counter(type(v).__name__ for v in values if v is not None)
    if not counters:
        return "unknown"
    def map_name(n: str) -> str:
        return {
            "int": "integer",
            "float": "float",
            "str": "string",
            "bool": "boolean",
            "dict": "object",
            "list": "list",
        }.get(n, n)
    top = counters.most_common(1)[0][0]
    return map_name(top)

def is_numeric_series(sample: List[Any]) -> bool:
    seen = 0
    ok = 0
    for x in sample:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            continue
        seen += 1
        if isinstance(x, (int, float)):
            ok += 1
            continue
        if isinstance(x, str):
            s = x.strip().replace(",", ".")
            try:
                float(s)
                ok += 1
            except Exception:
                pass
    return seen > 0 and ok / seen >= 0.9  # 90%+ numeric-like

def detect_semantic(value: str) -> Optional[str]:
    v = value.strip()
    if not v:
        return None
    if EMAIL_RE.match(v):
        return "email"
    if HAS_PHONES:
        try:
            pn = phonenumbers.parse(v, None)
            if phonenumbers.is_possible_number(pn) and phonenumbers.is_valid_number(pn):
                return "phone"
        except Exception:
            pass
    else:
        if PHONE_RE.match(v):
            return "phone"
    if COORD_PAIR_RE.match(v):
        try:
            parts = re.split(r"[,; ]+", v.strip("[]() "))
            lat, lon = float(parts[0]), float(parts[-1])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return "geo_coordinates"
        except Exception:
            pass
    if try_parse_date(v):
        if re.search(r"\d{1,2}:\d{2}", v):
            return "datetime"
        return "date"
    if ADDRESS_HINT_RE.search(v):
        return "address"
    if PRODUCT_CODE_RE.match(v):
        return "other"
    if 8 <= len(v) <= 40 and " " not in v and re.search(r"[A-Za-z]", v) and re.search(r"\d", v):
        return "identifier"
    return None

def entropy_of(values: List[str]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent

def derive_boolean_sense_from_name(col_name: str) -> str:
    name = col_name.strip().lower()
    prefixes = ["is_", "has_", "was_", "can_", "should_", "flag_"]
    for p in prefixes:
        if name.startswith(p):
            name = name[len(p):]
            break
    suffixes = ["_flag", "_indicator", "_bool"]
    for s in suffixes:
        if name.endswith(s):
            name = name[: -len(s)]
            break
    name = name.strip("_- ")
    return name or "boolean"

def download_from_kaggle(dataset: str, filename: Optional[str], outdir: Path) -> Path:
    if not HAS_KAGGLE:
        raise RuntimeError("kaggle package is not installed. Run: pip install kaggle")

    api = KaggleApi()
    api.authenticate()
    outdir.mkdir(parents=True, exist_ok=True)

    if filename:
        api.dataset_download_file(dataset, filename, path=str(outdir), force=True, quiet=False)
        zipped = outdir / (filename + ".zip")
        if zipped.exists():
            import zipfile
            with zipfile.ZipFile(zipped, "r") as zf:
                zf.extractall(outdir)
            zipped.unlink()
        target = outdir / filename
        if not target.exists():
            candidates = list(outdir.glob("*" + Path(filename).suffix))
            if not candidates:
                raise FileNotFoundError("Downloaded but couldn't locate the requested file.")
            target = candidates[0]
        return target

    api.dataset_download_files(dataset, path=str(outdir), force=True, quiet=False, unzip=True)
    exts = (".csv", ".xlsx", ".xls", ".parquet", ".txt")
    candidates = [p for p in outdir.iterdir() if p.suffix.lower() in exts]
    if not candidates:
        raise FileNotFoundError("No tabular files (.csv/.xlsx/.xls/.parquet/.txt) found in the dataset.")
    candidates.sort(key=lambda p: (p.suffix.lower() != ".csv", p.name))
    return candidates[0]

def download_from_url(url: str, outdir: Path) -> Path:
    if not HAS_REQUESTS:
        raise RuntimeError("requests package is not installed. Run: pip install requests")

    outdir.mkdir(parents=True, exist_ok=True)
    name = url.split("?")[0].split("/")[-1] or "downloaded_file"
    target = outdir / name

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {name}") if (tqdm and total > 0) else None
        with open(target, "wb") as f:
            for chunk in r.iter_content(chunk_size=128 * 1024):
                if chunk:
                    f.write(chunk)
                    if bar:
                        bar.update(len(chunk))
        if bar:
            bar.close()
    return target


def profile_dataframe(df, max_rows: Optional[int]=None) -> List[Dict[str, Any]]:
    if max_rows is not None and len(df) > max_rows:
        df = df.iloc[:max_rows].copy()

    results: List[Dict[str, Any]] = []

    for idx, col in enumerate(df.columns):
        series = df[col]
        non_null = series.dropna()
        sample_values = non_null.head(2000).tolist()
        sample_as_str = [str(x) for x in sample_values]

        list_count = 0
        list_inner_types = Counter()
        for s in sample_as_str[:200]:
            parsed = parse_list_like(s)
            if parsed is not None:
                list_count += 1
                inner = [infer_scalar_type(x) for x in parsed]
                list_inner_types.update(inner)

        bool_count = sum(1 for x in sample_values if is_boolish(x))

        is_numeric = is_numeric_series(sample_values[:1000])

        # Semantic sampling
        semantic_cnt = Counter()
        for s in sample_as_str[:400]:
            sem = detect_semantic(s)
            if sem:
                semantic_cnt[sem] += 1

        inferred_type = "unknown"
        details: Dict[str, Any] = {}

        if list_count >= max(3, int(0.1 * max(1, len(sample_as_str)))):
            inferred_type = "array"
            if list_inner_types:
                inner, _ = list_inner_types.most_common(1)[0]
                details["array_inner_type"] = inner
        elif bool_count >= max(3, int(0.8 * max(1, len(sample_values)))):
            inferred_type = "boolean"
        elif is_numeric:
            inferred_type = "numeric"
            ints = 0
            seen = 0
            for x in sample_values[:1000]:
                if x is None or (isinstance(x, float) and math.isnan(x)):
                    continue
                seen += 1
                try:
                    f = float(str(x).replace(",", "."))
                    if abs(f - round(f)) < 1e-9:
                        ints += 1
                except Exception:
                    pass
            details["numeric_subtype"] = "integer" if (seen and ints / seen >= 0.95) else "float"
        else:
            inferred_type = "categorical"

        semantic = None
        confidence = 0.0
        if semantic_cnt:
            semantic, hits = semantic_cnt.most_common(1)[0]
            confidence = hits / max(1, len(sample_as_str[:400]))

        if inferred_type == "numeric" and semantic is None:
            try:
                vals = []
                for x in sample_values[:200]:
                    f = float(str(x).replace(",", "."))
                    vals.append(f)
                if vals:
                    v = sum(vals) / len(vals)
                    if 978307200 <= v <= 4102444800:
                        semantic = "unix_timestamp_seconds"
                    elif 978307200000 <= v <= 4102444800000:
                        semantic = "unix_timestamp_millis"
            except Exception:
                pass

        n_unique = None
        if inferred_type in ("categorical", "boolean"):
            n_unique = int(non_null.nunique())

        # High-cardinality categorical -> likely identifiers
        if inferred_type == "categorical" and n_unique is not None and n_unique > max(50, 0.7 * len(non_null)):
            if not semantic:
                for s in sample_as_str[:100]:
                    sem = detect_semantic(s)
                    if sem:
                        semantic = sem
                        break
            if not semantic:
                semantic = "identifier_like"

        if inferred_type == "boolean":
            derived = derive_boolean_sense_from_name(str(col))
            semantic = derived
            confidence = 1.0

        cname = str(col)
        if cname in KNOWN_SENSE_OVERRIDES:
            semantic = KNOWN_SENSE_OVERRIDES[cname]
            if inferred_type != "categorical":
                confidence = 1.0

        result = {
            "column_index": idx,
            "column_name": cname,
            "inferred_type": inferred_type,
            "semantic_sense": semantic,
            "semantic_confidence": round(confidence, 3),
            "n_unique_if_categorical": n_unique,
        }
        result.update(details)
        results.append(result)

    return results

def infer_scalar_type(x: Any) -> str:
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "boolean"
    if isinstance(x, (int, float)):
        return "numeric"
    if isinstance(x, (list, tuple)):
        return "array"
    if isinstance(x, dict):
        return "object"
    if isinstance(x, str):
        s = x.strip().replace(",", ".")
        try:
            float(s)
            return "numeric"
        except Exception:
            pass
    return "string"

def load_dataframe(path: str, sheet: Optional[str], delimiter: Optional[str], max_rows: Optional[int]):
    ext = path.lower().split(".")[-1]
    if ext in ("csv", "txt"):
        return pd.read_csv(path, sep=delimiter if delimiter else None, nrows=max_rows)
    if ext in ("xlsx","xls"):
        return pd.read_excel(path, sheet_name=sheet, nrows=max_rows)
    if ext in ("parquet",):
        return pd.read_parquet(path)
    return pd.read_table(path, sep=delimiter if delimiter else None, nrows=max_rows)


def main():
    ap = argparse.ArgumentParser(description="Profile columns in a tabular dataset.")
    ap.add_argument("--input", default=None, help="Path to CSV/XLSX/Parquet file")
    ap.add_argument("--sheet", default=None, help="Sheet name for Excel files")
    ap.add_argument("--delimiter", default=None, help="Delimiter for CSV if not comma")
    ap.add_argument("--max-rows", type=int, default=None, help="Optional row cap for very large files")
    ap.add_argument("--output", default=None, help="Optional path to write JSON profile")

    ap.add_argument("--kaggle-dataset", default=None,
                    help="Kaggle dataset in the form owner/dataset (e.g., spittman1248/cdc-data-nutrition-physical-activity-obesity)")
    ap.add_argument("--kaggle-file", default=None,
                    help="Specific file inside the Kaggle dataset (e.g., data.csv) — optional.")
    ap.add_argument("--url", default=None,
                    help="Direct HTTP/HTTPS URL to a tabular file to download before profiling")
    ap.add_argument("--workdir", default=None,
                    help="Directory to store downloaded files (default: system temp dir)")

    args = ap.parse_args()

    if not any([args.input, args.kaggle_dataset, args.url]):
        ap.error("Please provide at least one source: --input OR --kaggle-dataset OR --url")

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="profile_dl_"))

    input_path = args.input
    if args.kaggle_dataset:
        fetched = download_from_kaggle(args.kaggle_dataset, args.kaggle_file, workdir)
        print(f"[Kaggle] Downloaded: {fetched}")
        input_path = str(fetched)
    elif args.url:
        fetched = download_from_url(args.url, workdir)
        print(f"[URL] Downloaded: {fetched}")
        input_path = str(fetched)

    df = load_dataframe(input_path, args.sheet, args.delimiter, args.max_rows)
    results = profile_dataframe(df, max_rows=args.max_rows)

    print("\nColumn Profile")
    print("-" * 120)
    header = f"{'Idx':>3}  {'Name':<30}  {'Type':<12}  {'Sense':<22}  {'Conf':>5}  {'#uniq(cat)':>11}  {'Sub/Inner':<15}"
    print(header)
    print("-" * 120)
    for r in results:
        idx = r.get("column_index")
        name = r.get("column_name","")[:30]
        itype = r.get("inferred_type","")
        sense = (r.get("semantic_sense") or "")[:22]
        conf = f"{r.get('semantic_confidence',0):.2f}"
        nunq = r.get("n_unique_if_categorical")
        subinner = r.get("numeric_subtype") or r.get("array_inner_type") or ""
        print(f"{idx:>3}  {name:<30}  {itype:<12}  {sense:<22}  {conf:>5}  {str(nunq) if nunq is not None else '':>11}  {subinner:<15}")

    base_out = args.output or (input_path + "_profile.json")
    with open(base_out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("-" * 120)
    print(f"Wrote JSON profile to: {base_out}")

if __name__ == "__main__":
    main()
