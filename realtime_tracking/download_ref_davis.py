"""Fetch Ref-DAVIS17: text expressions plus DAVIS 2017 val 480p frames/masks.

Downloads only the requested videos from the 830 MB DAVIS zip using HTTP range requests.
Usage: python realtime_tracking/download_ref_davis.py car-shadow [more videos...]   (no args = all val videos)
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

from remotezip import RemoteZip

DAVIS_ZIP = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
TEXT_ZIP = "https://www.mpi-inf.mpg.de/fileadmin/inf/d2/khoreva/davis_text_annotations.zip"
ROOT = Path(__file__).resolve().parent / "data" / "ref_davis17"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if not (ROOT / "davis_text_annotations").exists():
        print("Downloading text annotations...")
        zipfile.ZipFile(io.BytesIO(urlopen(TEXT_ZIP).read())).extractall(ROOT)

    with RemoteZip(DAVIS_ZIP) as z:
        names = z.namelist()
        val = z.read("DAVIS/ImageSets/2017/val.txt").decode().split()
        videos = sys.argv[1:] or val
        for video in videos:
            if video not in val:
                raise SystemExit(f"{video!r} is not a DAVIS 2017 val video")
            wanted = [n for n in names if not n.endswith("/") and (
                f"/480p/{video}/" in n and ("JPEGImages" in n or "Annotations" in n))]
            print(f"{video}: {len(wanted)} files")
            for n in wanted:
                out = ROOT / n.removeprefix("DAVIS/")
                if not out.exists():
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(z.read(n))


if __name__ == "__main__":
    main()
