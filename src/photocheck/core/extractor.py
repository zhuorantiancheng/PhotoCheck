"""EXIF metadata extractor for PhotoCheck."""

import os
import piexif
from pathlib import Path
from typing import Optional
from datetime import datetime

import pandas as pd

from .models import PhotoMetadata


# Only the EXIF region at the head of the file is needed — typically < 64 KB.
# Reading the whole 30-50 MB RAW file wastes ~99.7% of I/O.
# 256 KB safely includes Sony/Nikon/Canon MakerNotes; any offset pointing
# beyond the buffer raises struct.error from piexif and we fall back to the
# full read.
EXIF_HEADER_BYTES = 256 * 1024


def is_valid_dt(dt) -> bool:
    """True if dt is a real datetime (not None and not pd.NaT).

    After a parquet round-trip, missing datetimes come back as pd.NaT
    (not None). NaT propagates silently through arithmetic and indexing,
    so callers must filter it explicitly before using dt as a real
    timestamp. Use this helper instead of `dt is not None`.
    """
    if dt is None:
        return False
    try:
        return not pd.isna(dt)
    except (ValueError, TypeError):
        return False


# EXIF tag mappings: tag_id -> (parent_key, field_name)
EXIF_TAGS = {
    37386: ("Exif", "focal_length"),         # FocalLength
    41993: ("Exif", "focal_length_35mm"),   # FocalLengthIn35mmFilm
    33437: ("Exif", "f_stop"),                # FNumber
    34855: ("Exif", "iso"),                  # ISOSpeedRatings
    33434: ("Exif", "shutter_speed"),        # ExposureTime
    42036: ("Exif", "lens_name"),             # LensModel
    36867: ("Exif", "datetime_original"),    # DateTimeOriginal
    36868: ("Exif", "datetime_digitized"),   # DateTimeDigitized
    271: ("0th", "camera_make"),             # Make
    272: ("0th", "camera_model"),            # Model
}

# DateTime tag in 0th IFD
EXIF_TAGS[306] = ("0th", "datetime_modified")  # DateTime


# 35mm-equivalent crop factors by camera model. The EXIF tag
# FocalLengthIn35mmFilm (0xA405) is unreliable across brands: Nikon and
# Canon APS-C bodies don't write it, Sony full-frame bodies don't write
# it, and Sony APS-C bodies sometimes write 0 at long focal lengths.
# When the tag does carry a plausible value it wins (the body computed
# it with its real crop state, including in-body APS-C crop modes);
# otherwise the effective factor is max(body, lens).
CROP_FACTORS: dict[str, float] = {
    # Sony E APS-C
    "ILCE-6700": 1.5,
    # Nikon Z DX
    "NIKON Z 30": 1.5,
    # Canon APS-C
    "Canon EOS 60D": 1.6,
    # Full-frame bodies: factor 1.0 keeps raw focal lengths
    "ILCE-7CM2": 1.0,
    "LEICA M11-P": 1.0,
    "DC-S5M2": 1.0,
}

# Lens-name markers that mark a lens as APS-C, independent of body.
# Deliberately NOT matching Sony's "E " prefix: Tamron full-frame lenses
# such as "E 28-200mm F2.8-5.6 A071" (Di III) show up with an E prefix
# in EXIF LensModel on Sony bodies, so the prefix is not trustworthy.
APS_C_LENS_MARKERS: tuple[str, ...] = (
    "DC DN",          # Sigma APS-C mirrorless (vs "DG DN" full-frame)
    "Di III-A",       # Tamron APS-C mirrorless (vs "Di III" full-frame)
    "DX ",            # Nikon DX (NIKKOR Z DX ...)
    "EF-S",           # Canon APS-C DSLR
    "E 17-70mm",      # Tamron 17-70 B070 for Sony E (Di III-A, EXIF name lacks it)
    "E 70-350mm",     # Sony native APS-C E 70-350 G OSS
)

# Known full-frame lens markers that must override APS_C_LENS_MARKERS
# (none currently shadow each other, kept for documentation).


def crop_factor_for(camera_model: Optional[str], lens_name: Optional[str] = None) -> float:
    """Effective crop factor: max of body factor and lens factor.

    A full-frame body with an APS-C lens mounted (or an in-body APS-C
    crop mode) captures an APS-C image, so the lens raises the factor
    above the body's own; an APS-C body with any lens stays at the
    body's factor. Unknown bodies/lenses default to 1.0.

    Limitation: an in-body APS-C crop mode with a full-frame lens has no
    portable EXIF marker; rely on FocalLengthIn35mmFilm when the body
    writes it there (Sony APS-C bodies do).
    """
    body = CROP_FACTORS.get((camera_model or "").strip(), 1.0)
    if body > 1.0:
        return body
    lens = lens_name or ""
    if any(marker in lens for marker in APS_C_LENS_MARKERS):
        return 1.5
    return body


# Pre-grouped by parent_key for faster lookups. Built once at import.
# Maps parent_key -> list of (tag_id, field_name)
_TAGS_BY_PARENT: dict[str, list[tuple[int, str]]] = {}
for _tag_id, (_parent, _field) in EXIF_TAGS.items():
    _TAGS_BY_PARENT.setdefault(_parent, []).append((_tag_id, _field))


def _parse_rational(value) -> Optional[float]:
    """Parse an EXIF rational.

    piexif returns a (num, den) tuple for RATIONAL tags but a plain int
    for SHORT/LONG tags (e.g. FocalLengthIn35mmFilm) — handle both.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, tuple) and len(value) == 2:
        return value[0] / value[1]
    return None


def _parse_datetime(value: bytes) -> Optional[datetime]:
    """Parse EXIF datetime string to datetime object.

    EXIF datetime format: "YYYY:MM:DD HH:MM:SS"
    """
    if value is None:
        return None
    try:
        dt_str = value.decode("utf-8")
        return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
    except (ValueError, TypeError):
        return None


def _parse_string(value: bytes) -> Optional[str]:
    """Parse bytes as UTF-8 string."""
    if value is None:
        return None
    try:
        return value.decode("utf-8").rstrip("\x00")
    except (ValueError, TypeError):
        return None


def _read_exif_bytes(path: Path) -> Optional[bytes]:
    """Read the EXIF source bytes, preferring a 256 KB header read.

    Returns the first EXIF_HEADER_BYTES of the file on success, or None if
    the file is small enough that the partial-read path offers no benefit
    (caller should pass the path to piexif, which reads it whole).

    piexif.load(bytes) supports TIFF/WEBP/JPEG sources; passing bytes
    avoids the f.read() of the entire 30-50 MB RAW file. If any IFD
    offset points beyond the buffer, piexif raises struct.error and
    the caller falls back to the full-file read.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    if size <= EXIF_HEADER_BYTES:
        # No truncation risk; let piexif read the whole small file.
        return None
    try:
        with open(path, "rb") as f:
            return f.read(EXIF_HEADER_BYTES)
    except OSError:
        return None


def extract_metadata(
    image_path: Path,
    crop_factor: float = 1.0,
) -> PhotoMetadata:
    """Extract EXIF metadata from a single image.

    Focal-length resolution (35mm-equivalent):
    - If EXIF FocalLengthIn35mmFilm (0xA405) carries a plausible non-zero
      value, use it directly.
    - Otherwise scale the raw FocalLength by the camera body's crop
      factor (see CROP_FACTORS); unknown bodies default to 1.0.

    Args:
        image_path: Path to the image file.
        crop_factor: Deprecated, ignored. Kept for API compat only.

    Returns:
        PhotoMetadata object with extracted values.
    """
    result = PhotoMetadata(file_path=image_path)

    # Fast path: read just the first 256 KB (EXIF lives in the file head).
    # On ARW this is ~50x less I/O than reading the whole 40 MB file.
    # piexif raises struct.error if any IFD offset points past our buffer;
    # in that case we fall back to the full-file read.
    head = _read_exif_bytes(image_path)
    if head is not None:
        try:
            exif_data = piexif.load(head)
        except Exception:
            exif_data = None
        if exif_data is not None and not _looks_empty(exif_data):
            return _populate_result(result, exif_data)
        # Partial read yielded nothing useful — fall through to full read.

    try:
        exif_data = piexif.load(str(image_path))
    except Exception as e:
        result.error = str(e)
        return result

    return _populate_result(result, exif_data)


def _looks_empty(exif_data: dict) -> bool:
    """True if piexif returned no tags — partial read likely missed them.

    For JPEG partial-read, piexif silently returns an empty dict when
    the APP1 marker is beyond our buffer; for TIFF partial-read, it
    either works fully or raises. Either way, an empty result with a
    large file means the partial path failed and the caller should
    fall back.
    """
    for section in ("0th", "Exif", "GPS", "Interop", "1st"):
        if exif_data.get(section):
            return False
    return True


def _populate_result(
    result: PhotoMetadata, exif_data: dict
) -> PhotoMetadata:
    """Pull our known tags out of a piexif dict into the result."""
    raw_focal: float | None = None

    for parent_key, tags in _TAGS_BY_PARENT.items():
        parent = exif_data.get(parent_key)
        if parent is None:
            continue
        for tag_id, field_name in tags:
            try:
                value = parent.get(tag_id)
                if value is None:
                    continue

                if field_name in ("focal_length", "focal_length_35mm", "f_stop", "shutter_speed"):
                    parsed = _parse_rational(value)
                    if field_name == "focal_length":
                        raw_focal = parsed
                    else:
                        setattr(result, field_name, parsed)
                elif field_name.startswith("datetime"):
                    parsed = _parse_datetime(value)
                    setattr(result, field_name, parsed)
                else:
                    parsed = _parse_string(value) if isinstance(value, bytes) else value
                    setattr(result, field_name, parsed)

            except Exception:
                continue

    # Focal-length resolution:
    # 1. EXIF FocalLengthIn35mmFilm when it carries a plausible value
    #    (the body computes it from its real crop state; Sony sometimes
    #    writes 0 at long focal lengths — treat 0 as absent);
    # 2. otherwise raw FocalLength scaled by the effective crop factor
    #    (max of body and lens, see crop_factor_for).
    if result.focal_length_35mm:
        result.focal_length = result.focal_length_35mm
    elif raw_focal is not None:
        factor = crop_factor_for(result.camera_model, result.lens_name)
        result.focal_length = round(raw_focal * factor, 1) if factor != 1.0 else raw_focal
    else:
        result.focal_length = None

    return result
