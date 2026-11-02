"""Offline tests for the non-GUI logic. SAM itself is replaced by a fake: no model download."""
import hashlib
import os

import cv2
import numpy as np
import pytest
from PIL import Image
from PySide6.QtGui import QImage

import BoxCoordinates as bc


def ann(x, y, w, h, area=None):
    return {"bbox": [x, y, w, h], "area": w * h if area is None else area, "predicted_iou": 0.9}


# --- ordering -----------------------------------------------------------------

def test_reading_order_with_uneven_rows():
    # Boxes from the README screenshot: in every row the right box sits a few pixels
    # higher or lower than the left one. Lid numbers must still run row by row.
    boxes = [(364, 337, 196, 219), (67, 44, 195, 224), (52, 645, 194, 220),
             (373, 49, 204, 222), (367, 649, 194, 220), (40, 347, 219, 192)]
    ordered = bc.sort_reading_order([ann(*b) for b in boxes])
    assert [a["bbox"][:2] for a in ordered] == [[67, 44], [373, 49], [40, 347], [364, 337], [52, 645], [367, 649]]


def test_reading_order_empty():
    assert bc.sort_reading_order([]) == []


def test_filter_keeps_largest_and_numbers_row_by_row():
    image_area = 1000 * 1000
    masks = [
        ann(0, 0, 999, 999),        # background: above max area
        ann(10, 10, 5, 5),          # speck: below min area
        ann(100, 100, 400, 40),     # thin bar: aspect ratio 0.1
        ann(600, 95, 200, 200),     # right, slightly higher
        ann(100, 100, 210, 210),    # left
        ann(100, 500, 150, 150),    # smaller than the target count allows
        ann(600, 500, 205, 205),
        ann(100, 520, 200, 200),
    ]
    lids = bc.filter_and_sort_masks(masks, image_area, 4, 0.01, 0.30, 0.5)
    assert [a["bbox"][:2] for a in lids] == [[100, 100], [600, 95], [100, 520], [600, 500]]


def test_aspect_ratio():
    assert bc.calculate_aspect_ratio([0, 0, 50, 100]) == 0.5
    assert bc.calculate_aspect_ratio([0, 0, 0, 100]) == 0


# --- scaling ------------------------------------------------------------------

def test_sam_input_size():
    assert bc.compute_sam_input_size(800, 600) == (800, 600)
    assert bc.compute_sam_input_size(2560, 1920) == (1280, 960)
    assert bc.compute_sam_input_size(2400, 3200) == (960, 1280)
    assert bc.compute_sam_input_size(100000, 10) == (1280, 1)


def test_scale_bbox_without_resize_is_identity():
    assert bc.scale_bbox_to_original([10, 20, 30, 40], (800, 600), (800, 600)) == (10, 20, 40, 60)


def test_scale_bbox_covers_whole_last_pixel():
    # Resized pixels 10..19 cover original pixels 20..39 at scale 2. Version 1.0 gave xmax 38.
    assert bc.scale_bbox_to_original([10, 10, 9, 9], (1280, 960), (2560, 1920)) == (20, 20, 39, 39)


def test_scale_bbox_full_image_with_uneven_scale():
    sam_w, sam_h = bc.compute_sam_input_size(3000, 2000)
    assert (sam_w, sam_h) == (1280, 853)
    assert bc.scale_bbox_to_original([0, 0, sam_w - 1, sam_h - 1], (sam_w, sam_h), (3000, 2000)) == (0, 0, 2999, 1999)


# --- output -------------------------------------------------------------------

def test_coordinate_lines_format_is_unchanged():
    coords = [{"xmin": 1, "ymin": 2, "xmax": 3, "ymax": 4}, {"xmin": 10, "ymin": 20, "xmax": 30, "ymax": 40}]
    assert bc.format_coordinate_lines(coords) == "1,2,3,4\n10,20,30,40"


def test_export_path_keeps_name_with_spaces_and_accents():
    path = bc.build_export_path(os.path.join("out dir", "exports"), os.path.join("scans met spatie", "schimmel ü 01.tif"), 1700000000)
    assert path == os.path.join("out dir", "exports", "schimmel ü 01_annotated_1700000000.png")


def test_save_annotated_image_creates_folder(tmp_path):
    image = QImage(64, 48, QImage.Format.Format_RGB32)
    image.fill(0x336699)
    export_dir = tmp_path / "Documenten met spatie" / "BoxCoordinates" / "exports"
    saved = bc.save_annotated_image(image, str(tmp_path / "schimmel ü.png"), str(export_dir), 1700000000)
    assert saved == str(export_dir / "schimmel ü_annotated_1700000000.png")
    with Image.open(saved) as reloaded:
        assert reloaded.size == (64, 48)
        assert reloaded.convert("RGB").getpixel((0, 0)) == (0x33, 0x66, 0x99)


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root can write anywhere")
def test_save_annotated_image_reports_the_reason(tmp_path):
    # A file where the export folder should be makes the folder impossible to create on every
    # system. (A read-only folder does not: Windows ignores the read-only flag on folders.)
    blocked = tmp_path / "exports"
    blocked.write_text("")
    with pytest.raises(OSError):
        bc.save_annotated_image(QImage(4, 4, QImage.Format.Format_RGB32), "x.png", str(blocked))


# --- input --------------------------------------------------------------------

def defaults_as_text(**overrides):
    values = {key: str(value) for key, value in bc.DEFAULT_PARAMS.items()}
    values.update(overrides)
    return values


def test_defaults_are_valid():
    params, error = bc.parse_params(defaults_as_text())
    assert error is None
    assert params == bc.DEFAULT_PARAMS


def test_decimal_comma_is_accepted():
    params, error = bc.parse_params(defaults_as_text(MIN_LID_AREA_FRACTION="0,02"))
    assert error is None and params["MIN_LID_AREA_FRACTION"] == 0.02


@pytest.mark.parametrize("key,text,needle", [
    ("NUM_LIDS_TO_PROCESS", "six", "Num Lids Target"),
    ("NUM_LIDS_TO_PROCESS", "0", "at least 1"),
    ("NUM_LIDS_TO_PROCESS", "-2", "at least 1"),
    ("SAM_POINTS_PER_SIDE", "0", "SAM Pts/Side"),
    ("SAM_POINTS_PER_SIDE", "1000", "SAM Pts/Side"),
    ("SAM_MIN_MASK_REGION_AREA", "-1", "negative"),
    ("MIN_LID_AREA_FRACTION", "0.5", "Area Fractions"),
    ("MAX_LID_AREA_FRACTION", "nan", "finite"),
    ("MIN_ASPECT_RATIO_FILTER", "0", "Aspect Ratio"),
    ("MIN_ASPECT_RATIO_FILTER", "1.5", "Aspect Ratio"),
])
def test_invalid_parameters_are_rejected(key, text, needle):
    params, error = bc.parse_params(defaults_as_text(**{key: text}))
    assert params is None
    assert needle in error


# --- image loading ------------------------------------------------------------

def test_16_bit_scan_is_stretched_not_clipped(tmp_path):
    values = np.linspace(1000, 4000, 200 * 100).reshape(100, 200).astype(np.uint16)
    path = tmp_path / "scan16.tif"
    Image.fromarray(values).save(path)
    rgb = np.asarray(bc.load_image_rgb(str(path)))
    assert rgb.shape == (100, 200, 3)
    assert rgb.min() == 0 and rgb.max() == 255
    assert np.mean(rgb == 255) < 0.05  # convert("RGB") alone turns all of it white


def test_8_bit_image_is_unchanged(tmp_path):
    values = np.arange(256, dtype=np.uint8).reshape(16, 16)
    path = tmp_path / "scan8.png"
    Image.fromarray(values).save(path)
    rgb = np.asarray(bc.load_image_rgb(str(path)))
    assert (rgb[:, :, 0] == values).all()


# --- model download -----------------------------------------------------------

class FakeResponse:
    def __init__(self, chunks, content_length=None, status_error=None):
        self.chunks = chunks
        self.headers = {} if content_length is None else {"content-length": str(content_length)}
        self.status_error = status_error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_error:
            raise self.status_error

    def iter_content(self, chunk_size):
        yield from self.chunks


def fake_get(response):
    return lambda url, stream, timeout: response


PAYLOAD = [b"a" * 1000, b"b" * 1000, b"c" * 500]
PAYLOAD_SHA = hashlib.sha256(b"".join(PAYLOAD)).hexdigest()


def leftovers(folder):
    return sorted(os.listdir(folder)) if os.path.isdir(folder) else []


def test_download_verifies_and_moves_into_place(tmp_path, monkeypatch):
    monkeypatch.setattr(bc.requests, "get", fake_get(FakeResponse(PAYLOAD, 2500)))
    dest = tmp_path / "models" / "sam.pth"
    progress = []
    bc.download_checkpoint("http://x", str(dest), 2500, PAYLOAD_SHA, progress_callback=lambda d, t: progress.append((d, t)))
    assert dest.read_bytes() == b"".join(PAYLOAD)
    assert leftovers(tmp_path / "models") == ["sam.pth"]
    assert progress[-1] == (2500, 2500)


def test_truncated_download_leaves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(bc.requests, "get", fake_get(FakeResponse(PAYLOAD[:2], 2500)))
    with pytest.raises(IOError, match="incomplete"):
        bc.download_checkpoint("http://x", str(tmp_path / "sam.pth"), 2500, PAYLOAD_SHA)
    assert leftovers(tmp_path) == []


def test_corrupt_download_leaves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(bc.requests, "get", fake_get(FakeResponse([b"x" * 2500], 2500)))
    with pytest.raises(IOError, match="checksum"):
        bc.download_checkpoint("http://x", str(tmp_path / "sam.pth"), 2500, PAYLOAD_SHA)
    assert leftovers(tmp_path) == []


def test_cancelled_download_leaves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(bc.requests, "get", fake_get(FakeResponse(PAYLOAD, 2500)))
    with pytest.raises(bc.DownloadCancelled):
        bc.download_checkpoint("http://x", str(tmp_path / "sam.pth"), cancel_requested=lambda: True)
    assert leftovers(tmp_path) == []


def test_http_error_leaves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(bc.requests, "get", fake_get(FakeResponse([], status_error=bc.requests.HTTPError("404"))))
    with pytest.raises(bc.requests.HTTPError):
        bc.download_checkpoint("http://x", str(tmp_path / "sam.pth"))
    assert leftovers(tmp_path) == []


def test_find_checkpoint_skips_incomplete_files(tmp_path, monkeypatch):
    user_dir, exe_dir = tmp_path / "user", tmp_path / "exe"
    user_dir.mkdir()
    exe_dir.mkdir()
    monkeypatch.setattr(bc, "SAM_CHECKPOINT_SIZE", 10)
    monkeypatch.setattr(bc, "SAM_CHECKPOINT_PATH", str(user_dir / bc.SAM_CHECKPOINT_FILENAME))
    monkeypatch.setattr(bc, "EXTRA_CHECKPOINT_DIRS", [str(exe_dir)])
    assert bc.find_sam_checkpoint() is None
    (user_dir / bc.SAM_CHECKPOINT_FILENAME).write_bytes(b"12345")       # half a download
    assert bc.find_sam_checkpoint() is None
    (exe_dir / bc.SAM_CHECKPOINT_FILENAME).write_bytes(b"0123456789")   # placed next to the exe
    assert bc.find_sam_checkpoint() == str(exe_dir / bc.SAM_CHECKPOINT_FILENAME)
    (user_dir / bc.SAM_CHECKPOINT_FILENAME).write_bytes(b"0123456789")
    assert bc.find_sam_checkpoint() == str(user_dir / bc.SAM_CHECKPOINT_FILENAME)


# --- whole pipeline with a fake SAM -------------------------------------------

# A 2400x3200 scan (resized to 960x1280 for SAM) with 2 x 3 light lids. The right
# column sits 15 px higher. Ground truth is inclusive (xmin, ymin, xmax, ymax).
SCAN_SIZE = (2400, 3200)
LIDS = [(300 + col * 1000, 300 + row * 1000 - 15 * col, 300 + col * 1000 + 799, 300 + row * 1000 - 15 * col + 699)
        for row in range(3) for col in range(2)]


class FakeSam:
    def to(self, device):
        return self

    def eval(self):
        return self


class FakeMaskGenerator:
    """Finds the bright rectangles with OpenCV and reports them the way SAM does."""

    def __init__(self, model, **kwargs):
        pass

    def generate(self, image):
        count, labels, stats, _ = cv2.connectedComponentsWithStats((image[:, :, 0] > 115).astype(np.uint8))
        masks = [{"segmentation": labels == i, "area": int(stats[i, 4]), "predicted_iou": 0.95,
                  "bbox": [int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2]) - 1, int(stats[i, 3]) - 1]}
                 for i in range(1, count)]
        h, w = labels.shape
        masks.append({"segmentation": np.ones((h, w), bool), "area": h * w, "predicted_iou": 0.9, "bbox": [0, 0, w - 1, h - 1]})
        return masks


class Recorder:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


class Signals:
    def __init__(self):
        self.status_updated, self.error, self.finished = Recorder(), Recorder(), Recorder()


@pytest.fixture
def fake_sam(monkeypatch):
    monkeypatch.setattr(bc, "sam_model_registry", {bc.MODEL_TYPE: lambda checkpoint: FakeSam()})
    monkeypatch.setattr(bc, "SamAutomaticMaskGenerator", FakeMaskGenerator)
    monkeypatch.setattr(bc, "find_sam_checkpoint", lambda: "fake.pth")


def make_scan(path):
    width, height = SCAN_SIZE
    scan = np.full((height, width), 30, np.uint8)
    for x0, y0, x1, y1 in LIDS:
        scan[y0:y1 + 1, x0:x1 + 1] = 200
    Image.fromarray(scan).save(path)


def test_pipeline_finds_lids_in_original_coordinates(tmp_path, fake_sam):
    scan_path = tmp_path / "scans met spatie" / "schimmel ü.png"
    scan_path.parent.mkdir()
    make_scan(scan_path)
    signals = Signals()
    bc.run_sam_processing(str(scan_path), dict(bc.DEFAULT_PARAMS), signals)
    assert signals.error.calls == []
    coordinates, qimage, params = signals.finished.calls[0]
    assert params == bc.DEFAULT_PARAMS
    assert not qimage.isNull()
    found = [(c["xmin"], c["ymin"], c["xmax"], c["ymax"]) for c in coordinates]
    assert len(found) == len(LIDS)
    for got, want in zip(found, LIDS):  # same order: row by row, left to right
        assert all(abs(g - w) <= 3 for g, w in zip(got, want)), (got, want)


def test_pipeline_reports_no_lids_once(tmp_path, fake_sam):
    blank = tmp_path / "blank.png"
    Image.new("RGB", (400, 300), (30, 30, 30)).save(blank)
    signals = Signals()
    bc.run_sam_processing(str(blank), dict(bc.DEFAULT_PARAMS), signals)
    assert signals.finished.calls == []
    assert len(signals.error.calls) == 1
