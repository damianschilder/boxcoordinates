# Changelog

## 1.1 (November 2026)

Bug fixes made with Claude Code. The workflow, parameters, look and the `xmin,ymin,xmax,ymax` output format are unchanged.

### Fixed

- **"Error saving annotated image."** (the status line in the README screenshot). The image was saved to a relative `exports` folder in the working directory. For the installed app that is the install folder in Program Files, which a normal user cannot write to. Annotated images now go to `Documents\BoxCoordinates\exports`, and if saving still fails the status line says why. The app also no longer creates `exports` in the working directory at startup, which crashed it when that folder was not writable.
- **Model download needed administrator rights, or was repeated.** The checkpoint was written into the program's bundle folder (`sys._MEIPASS`): inside Program Files for the installed app (hence the old "run as administrator" advice), and a temporary folder that is deleted on exit for a one-file exe. It is now stored in `%LOCALAPPDATA%\BoxCoordinates`. A checkpoint next to `BoxCoordinates.exe`, or one downloaded by version 1.0, is still used.
- **Incomplete downloads were kept.** The model now downloads to a `.part` file and is only used after its size and SHA-256 checksum match; an incomplete file is ignored. Closing the window cancels a running download instead of waiting for it to finish.
- **Lids were numbered out of order.** They were sorted by `(ymin, xmin)`, so a box a few pixels higher than its left neighbour got the lower number (lids 2 and 3 in the README screenshot). In time series that mixes up containers between scans. Lids are now numbered row by row, left to right.
- **Boxes on large images were slightly too small.** For images over 1280 px (which are resized for SAM), `xmax`/`ymax` were mapped to the start of the last resized pixel instead of its end: up to 2 px short on the right and bottom edge at scale 2.5. The mapping is now exact. Images up to 1280 px give the same coordinates as before.
- **Labels lost their "&".** "Select Image & Process" showed as "Select Image  Process" and "Processing Log & Full Results" as "Processing Log  Full Results": Qt reads a single `&` as a keyboard-shortcut marker.
- **16-bit TIFF scans turned white.** Converting them to RGB clipped every value above 255. They are now stretched to 8 bit for SAM; the coordinates still refer to the original image.
- **The window froze at the end of each run.** SAM already ran in a worker thread, but drawing the annotated image sent one full-size overlay per lid through matplotlib's image resampler, which holds Python's GIL. In tests the GUI thread stalled for 1.4 to 24 s, depending on the machine's load. The masks are now blended with numpy and drawn once, giving the same picture without the stall.
- **Qt widgets were created outside the GUI thread.** matplotlib picked its Qt backend for the annotated figure, which is drawn in the worker thread ("Starting a Matplotlib GUI outside of the main thread will likely fail"). It now uses the non-GUI Agg backend.
- **Parameter checks.** Num Lids Target 0 or negative, SAM Pts/Side 0 or very large, and a negative SAM Min Mask Area were accepted. The error message now names the field, and a decimal comma (`0,01`) is accepted.
- **No lids found:** the empty result was shown and then immediately cleared by the error message; now only the error is shown.
- Long save paths wrap in the status line instead of widening the window.

### Changed

- Two installers: `BoxCoordinates_setup.exe` (PyTorch with CUDA 13.0: uses an NVIDIA RTX 20 series or newer card when present, the processor otherwise) and the much smaller `BoxCoordinates_setup_cpu.exe` (processor only).
- The installers are built by GitHub Actions on a Windows machine. Each build first runs the tests and a self-test of the built exe.
- Installer script renamed from `test.iss` to `BoxCoordinates.iss`, with paths relative to the repository instead of `C:\Users\Laptop\...`, and version 1.1.
- README: the download and install steps match the actual release (an installer, not a single exe), the build steps are complete again, and there is a "How it was built" section.
- Added `requirements.txt`, an offline test suite (`tests/`, SAM replaced by a fake), and `.gitignore` entries for virtual environments, model checkpoints and build output.
- Replaced the placeholder Windows app ID `mycompany.myproduct.subproduct.version`, and removed the template brackets around the name in `LICENSE`.

### Known limitations

- Every run loads the model again, which adds a few seconds.
- Closing the window while SAM is processing waits until that image is finished.
- If SAM returns both a lid and a region inside it, both can be selected. Tighten **Min/Max Area Fraction** in that case.

## 1.0 (June 2025)

First release: `BoxCoordinates_setup.exe`.
