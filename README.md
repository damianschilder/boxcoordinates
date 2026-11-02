# Coordinates Extractor using SAM

A Windows desktop app that runs Meta's Segment Anything Model (SAM) locally to find the lids of containers in a scanned image and export their bounding-box coordinates.

![screenshot2](https://github.com/user-attachments/assets/c622615e-e02e-4037-a9d8-242b6272b181)

*Version 1.0. The status line shows "Error saving annotated image." and the labels lost their "&": both are fixed in 1.1, see the [changelog](CHANGELOG.md).*

## About the project

The tool was built for a researcher at Utrecht University who selected the coordinates of fungus container lids by hand. Her research scans the containers at time intervals and measures fungal growth through the grey values inside those coordinates.

The app puts a graphical interface on the Segment Anything Model: load an image, set the filter parameters, and get machine-readable coordinates for the objects it finds. It keeps the objects ("lids") that match a target count, size and shape.

It ships as a standalone Windows program with an installer, so it can be used without Python or a command line.

## Key features

* **SAM integration:** uses the `vit_b` variant of Meta's Segment Anything Model, on the CPU or a CUDA GPU.
* **Graphical interface:** built with PySide6 for Windows.
* **Automatic model download:** offers to download the SAM checkpoint (`sam_vit_b_01ec64.pth`, ~375 MB) on first run, and checks its size and SHA-256 checksum.
* **Configurable parameters:** filters for the number of lids, object size (min/max area) and shape (aspect ratio).
* **Visual feedback:** shows the image with segmentation masks and numbered bounding boxes.
* **Data export:** coordinates as `xmin,ymin,xmax,ymax` lines plus a detailed log; the annotated image is saved as a PNG.
* **Clipboard:** copy the coordinates with one click.

## Download

Download `BoxCoordinates_setup.exe` from the **[Releases page](https://github.com/damianschilder/BoxCoordinates/releases/latest)** (Windows, 64-bit), run it, and start **BoxCoordinates** from the Start menu.

## How to use

1. Start the app. On the first run, let it download the SAM model file (~375 MB). It is stored in `%LOCALAPPDATA%\BoxCoordinates`, so no administrator rights are needed.
2. Set the **Tuning Parameters**:
   * **Num Lids Target:** the number of objects you expect.
   * **Min/Max Area Fraction:** the object size relative to the whole image.
   * **Min Aspect Ratio:** the object shape (1.0 is a perfect square).
3. Click **Select Image & Process** and choose an image (TIFF, JPEG, PNG or BMP).
4. The annotated image appears on the left, and is saved to `Documents\BoxCoordinates\exports`.
5. The coordinates appear on the right. **Copy** puts them on the clipboard.
6. To process the same image with other parameters, change the values and click **Retry with Current Params**.

**Output.** One line per lid: `xmin,ymin,xmax,ymax`, in pixels of the original image, with `xmax` and `ymax` the last pixel inside the box. Lid 0 is top left; the lids are numbered row by row, left to right, matching the numbers in the annotated image.

## Tips for better results

* **Num Lids Target:** set it to the exact number of objects in the image.
* **SAM Pts/Side:** controls how densely the model samples the image. Raising it to `50` or `90` finds lids more reliably but takes longer.

## Troubleshooting

* **The model download fails or the PC is offline:** download [`sam_vit_b_01ec64.pth`](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) elsewhere and put it in `%LOCALAPPDATA%\BoxCoordinates\` or next to `BoxCoordinates.exe`. The app only accepts the complete file (375,042,383 bytes).
* **Processing is slow:** on a CPU one image can take a few minutes. Lower **SAM Pts/Side** for a faster, rougher run.

## How it was built

I built this app in June 2025, entirely with browser-based Gemini 2.5 Pro: it was my first fully AI-generated project, my first desktop app, and the first time I downloaded and ran a model locally. It did the job, but it had some issues.

In November 2026 I went back to it with Claude Code to find and fix those issues while keeping the app recognisable: same workflow, same parameters, same output format. The fixes are listed in the [changelog](CHANGELOG.md).

## Building from source

**Prerequisites:** Python 3.9+ and Git.

1. Clone the repository and create a virtual environment:

   ```bash
   git clone https://github.com/damianschilder/BoxCoordinates.git
   cd BoxCoordinates
   python -m venv .venv
   ```

   Activate it with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Linux/macOS.

2. Install the dependencies. For a smaller CPU-only PyTorch, install that first; otherwise pick the right build on the [PyTorch site](https://pytorch.org/get-started/locally/):

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   python BoxCoordinates.py
   ```

4. Run the tests (offline, SAM is replaced by a fake, no model download):

   ```bash
   pip install pytest
   python -m pytest -q
   ```

5. Build the Windows program with PyInstaller (one-folder build, which the installer script expects in `dist\BoxCoordinates\`):

   ```bash
   pip install pyinstaller
   pyinstaller --noconfirm --windowed --name BoxCoordinates --icon icon.ico --add-data "icon.ico;." BoxCoordinates.py
   ```

6. Compile `BoxCoordinates.iss` with [Inno Setup](https://jrsoftware.org/isinfo.php) 6. The installer is written to `Output\BoxCoordinates_setup.exe`.

## Acknowledgements

* **[Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)** by Meta AI.
* The GUI is built with **[PySide6](https://www.qt.io/qt-for-python)**.

## License

MIT License. See [LICENSE](LICENSE).
