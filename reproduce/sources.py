"""Raw public files: where they come from, their licences and their SHA-256 fingerprints.

download() fetches anything missing into reproduce/raw/ (not committed) and stops if a
file differs from the recorded fingerprint, so a silent revision by the CDC or UCI can't
change the activity's data.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

NHANES = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{}.xpt"
UCI_HEART = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/{}"
UCI_READMIT = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"

LICENCE = {
    "nhanes": "Public domain (US government work), CDC/NCHS",
    "uci": "CC BY 4.0, UCI Machine Learning Repository",
}

# name → (url, sha256, licence key). The two readmission files come out of one zip, so
# they are checked after extraction (UCI has re-zipped this archive before).
RAW = {
    "DEMO_J.xpt": (NHANES.format("DEMO_J"), "c0b46e0345ea19404928656277c8b0d10b0cca348a9b2fe4fc3c67e8b7ee73ec", "nhanes"),
    "BMX_J.xpt": (NHANES.format("BMX_J"), "8d675e42d8826ac98714b2c3dd4c5138a5e353fb4424f7eff5e6db4a01ce838a", "nhanes"),
    "BPX_J.xpt": (NHANES.format("BPX_J"), "f98f1d9a4c18173e715c749b4294f6f1525140d5ad5751316bb17cc038fb3423", "nhanes"),
    "GHB_J.xpt": (NHANES.format("GHB_J"), "35f07094573a0061a03ed609a5a363b34eb1b1c7065d1623b43d72e132a8a654", "nhanes"),
    "DIQ_J.xpt": (NHANES.format("DIQ_J"), "1ecbf5360dfc331d1efbf32198553dc30e9a1f4cff0a907ed30ca72bac797f89", "nhanes"),
    "HDL_J.xpt": (NHANES.format("HDL_J"), "9f4eb41b89f0c9f3d6262f873e671eeb2f2eb40581a8426b2a3c8fd4fc19439e", "nhanes"),
    "TCHOL_J.xpt": (NHANES.format("TCHOL_J"), "0291a6a4f6d82dac8392c8c992b519946824dfeda01647935520cf506ed9f4ab", "nhanes"),
    "processed.cleveland.data": (UCI_HEART.format("processed.cleveland.data"),
                                 "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8", "uci"),
    "heart-disease.names": (UCI_HEART.format("heart-disease.names"),
                            "ec7ed5aa4ed8321097808f47b836301c2a0c36f3bb09c40b8b43b622363e6b61", "uci"),
    "diabetic_data.csv": (UCI_READMIT, None, "uci"),
    "IDS_mapping.csv": (UCI_READMIT, None, "uci"),
}
ZIPPED = {
    "diabetic_data.csv": "0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97",
    "IDS_mapping.csv": "f1bb82b471cb34649352597572c9b1fb00bd27f77b9f5a22a03dc3eb1039749e",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as r:
        return r.read()


def download(raw: Path) -> dict[str, dict]:
    """Fetch missing raw files, check every fingerprint, and return their provenance."""
    raw.mkdir(parents=True, exist_ok=True)
    for name, (url, _, _) in RAW.items():
        if (raw / name).exists():
            continue
        print(f"  downloading {url}")
        blob = _fetch(url)
        if name in ZIPPED:
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                for member in ZIPPED:
                    hit = next(n for n in z.namelist() if n.endswith(member))
                    (raw / member).write_bytes(z.read(hit))
        else:
            (raw / name).write_bytes(blob)
    out = {}
    for name, (url, expected, lic) in RAW.items():
        expected = expected or ZIPPED[name]
        got = sha256(raw / name)
        if got != expected:
            raise SystemExit(f"{name}: SHA-256 {got} does not match the recorded {expected}. "
                             "The source file has changed; check it before updating sources.py.")
        out[name] = {"url": url, "sha256": got, "licence": LICENCE[lic]}
    return out
