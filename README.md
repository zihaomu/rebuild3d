# Rebuild3D

A native macOS app for reconstructing textured 3D models from photographs.

Rebuild3D uses SwiftUI and RealityKit Object Capture. Reconstruction runs locally;
input photographs are not uploaded. The initial implementation targets a single
stationary, textured object and exports USDZ with embedded materials and textures.

## Status

The v2 implementation includes direct photo import, recoverable drafts, content-based
format detection, exact duplicate detection, versioned project migration, reconstruction
input snapshots, progress and cancellation, an orbit/pan/zoom viewer, and USDZ export.
Public datasets containing 59 original HEIC photos and 33 high-resolution JPEG photos
have completed reconstruction and byte-preserving export with independently rendered
textures. The HEIC set has also passed draft recovery, in-app reconstruction, saving,
release-build reopening, and export. Seven original 24.5 MP iPhone 16 Pro HEIC photos
pass import, orientation, and persistence checks but fail during image alignment.
Controlled format, depth, resolution, and sensitivity experiments did not resolve that
Object Capture failure. A separate local research pipeline now reconstructs that fixed
seven-photo statue using VGGT predictions and silhouette-constrained depth fusion,
with explicitly marked inferred and completed regions. HDR display behavior and successful high-resolution resource baselines remain
unverified. Camera pose recovery and photo-aligned
comparison remain P1 work; offline pose and projection probes do not enable them in the app.

The implementation plan and current evidence are maintained in
[the v2 kickoff document](doc/Rebuild3D-项目启动文档-v2.md) and
[the v2 acceptance record](doc/验收记录-v2-2026-10-08.md).

The exploration is defined in the
[seven-photo statue goal](doc/GOAL-七张照片佛像重建探索.md): keep the same seven source
photos, seek a usable coarse 3D result, and explicitly label inferred or completed
geometry while checking it against all original views. The research pipeline exports
GLB, USDZ, per-face provenance, and seven-view comparisons. **Load Approximation**
imports its result after checking exact original-photo hashes, and **Sources** shows
inferred regions. This is a local research-result bridge; the regular **Reconstruct**
button still uses Object Capture. VGGT and its weights are not bundled with the app.
See the [execution record](doc/七张佛像-探索记录-2026-10-08.md) for commands and limitations. The existing
workflow and diagnosis are preserved by the `v0.2.0-sparse-baseline` tag.

## Requirements

- macOS 26 or later and a Mac supported by `PhotogrammetrySession.isSupported`.
- Swift 6 or later with the macOS 26 SDK (Xcode or Command Line Tools).
- Start with Recommended quality (the engine's reduced detail) on a 16 GB Mac.
  Only one session runs at a time.
- Full Xcode and a signing identity are needed for the later distribution workflow.

Verified development environment: Apple M5, 16 GB, macOS 26.4.1, Swift 6.3.1,
macOS SDK 26.4, Command Line Tools. Runtime Object Capture support is available.

## Build and run

```sh
swift build
./scripts/build-app.sh
open build/Rebuild3D.app
```

Open `Package.swift` in Xcode to edit the native Swift package. The packaging script
creates an ad-hoc signed local application with its project document type and both
license notices. It is not a notarized release. Use `./scripts/build-app.sh release`
for an optimized local build. The currently verified package is arm64; Intel and
other-machine execution have not been tested.

```sh
./scripts/test.sh
swift run rebuild3d-check doctor
./scripts/check-upstream.sh
```

The test script supplies Swift Testing framework/runtime paths when using Command
Line Tools. Storage tests use generated image fixtures and opaque model payloads;
they do not establish reconstruction quality or texture correctness.

`scripts/probe-disk-full.swift` additionally exercises real out-of-space failures and
retry behavior on a separately mounted, disposable filesystem of at most 128 MiB.
It checks manifest/model preservation, draft publication, export replacement, and
partial-copy cleanup. Build it with the core sources, as documented in the script;
keep its report directory outside the scratch volume. This is a storage check,
not a reconstruction-quality or UI test.

## Reconstruct an object

1. Choose **Add Photos** or drop a photo folder or multiple photos into the window.
2. A recoverable draft is created automatically; no project location is required yet.
3. Review import issues and remove unwanted photos from the list.
4. Choose **Start Reconstruction** with the recommended settings. Optional quality
   settings are in the collapsed **Reconstruction Options** section.
5. Inspect the model: drag to orbit, scroll or pinch to zoom, Shift-drag or
   right-drag to pan, and choose **Reset View** to frame the object.
6. Save the project to a new location and use **Export USDZ** to copy the complete asset elsewhere.

The **Recover Draft** menu lists locally retained drafts. Formal projects are never
overwritten by a draft save. Use **File → Discard Draft** only when you intend to delete
the draft's copied data; source files outside the project are not changed.

You can also exercise the same import/storage/engine pipeline from the terminal:

```sh
swift run rebuild3d-check reconstruct /path/to/photos /path/to/Object.rebuild3d
swift run rebuild3d-check rebuild /path/to/Object.rebuild3d
swift run rebuild3d-check export /path/to/Object.rebuild3d /path/to/export.usdz
```

The destination must not already exist. This creates a self-contained project,
runs reduced-quality reconstruction, and records a JSON report in `logs/`.
The `rebuild` command uses a saved project's settings and can exercise cancellation
with `--cancel-after 2`; intentional cancellation exits with code 130.

Use sharp photographs with substantial overlap, diffuse light, and coverage at
several heights. Roughly 80–150 photos around 12 MP are a starting experiment,
not a requirement or performance guarantee. Reflective, transparent, featureless,
moving, or deforming subjects are difficult. Folder import is nonrecursive and
accepts JPEG, common HEIC/HEIF, PNG, and single-image TIFF based on decoded content.
HEIF uses its primary image. RAW/DNG and animated or multipage non-HEIF inputs are
unsupported; videos, auxiliary files, hidden files, and nested folders are ignored.
An invalid file does not stop other valid inputs. Repeated source bytes are skipped,
including after saving and reopening. Thumbnails and photo previews use a consistent
color-managed SDR/sRGB policy. HEIC, JPEG, and PNG reconstruction inputs retain their
original bytes. Single-image TIFF files get full-resolution, orientation-normalized
SDR/sRGB PNG working copies because Object Capture's folder input omits TIFF files.
Original TIFF files are retained, with the conversion, pixel transform, output size,
and separate source/input digests recorded in the run snapshot.

Before starting, the app explains missing inputs or unsupported hardware and checks
the engine's current image-count and per-side pixel limits. Oversized photos are marked
in the list and must be removed from the input selection; their project originals remain.
Three photos are the minimum start condition, not a guarantee of a usable reconstruction.
Run `rebuild3d-check doctor` to inspect the limits reported by the current Mac.

## Project storage

```text
Object.rebuild3d/
  project.json        # Versioned metadata with relative paths
  images/             # Copied originals, including embedded metadata
  thumbnails/         # Regenerable thumbnails
  models/             # Immutable successful USDZ results
  runs/<run-id>/
    inputs.json       # Immutable source identity, working-image recipe, pixel transform
    research/         # Optional approximation bundle, source regions and hashed artifacts
  logs/               # Run status, warnings, duration, memory observations, size
  cache/              # Disposable active-run staging
```

Drafts live in `~/Library/Application Support/org.rebuild3d.app/Drafts/`. Successful
draft imports and settings changes are saved for recovery. Saving a draft copies it
to a staging package, validates it, publishes the destination, and then removes the
recovery copy. Cancellation or failure during saving preserves the draft.
For an existing formal project, Save commits input and settings changes.
Reconstruction saves current inputs first and commits a successful model automatically.
Move the complete package to relocate it; saved models reopen without reconstruction.

The current storage format is 3. Formats 1 and 2 migrate in memory when opened;
their manifest is upgraded atomically on a successful save. Existing photo IDs and
models are retained. Legacy models do not receive invented run or camera records.
Older apps reject format 3 instead of silently displaying an approximation without
its source labels. Approximate exports include a same-stem `.rebuild3d-result` folder
containing the source-region USDZ and complete research records. Keep this folder with
the ordinary USDZ. Choose a new name if that companion folder already exists; it is
never silently replaced.

New results become active only after an atomic manifest update. Cancellation or
failure preserves the previous model. Failed photo and model copies are cleaned up,
including copies that run out of disk space. Removed photos and superseded models are
retained for now; automatic garbage collection and reconstruction checkpoint resumption
are not implemented. Recovery covers data already persisted in a draft, not an
unfinished import or reconstruction. Do not edit the same project concurrently from multiple processes.
If a newly reconstructed model cannot be saved, check free disk space and the project's
write permissions, then retry reconstruction. Diagnostics and the run log retain the
underlying error; the previously saved model remains available.
Run memory metrics sample the application's resident memory, not total system or
all framework helper-process memory. The cache is disposable only when no run is active.

## Upstream and licensing

Based on [ekarad1um/Photogrammetry](https://github.com/ekarad1um/Photogrammetry),
pinned to commit `43c66741e3f912ddd29292858d6f6bc01441606e`.
The original source and Xcode project are preserved in `Vendor/Photogrammetry`.
See [upstream provenance](Vendor/UPSTREAM.md).

Rebuild3D's repository license is [Apache-2.0](LICENSE). Upstream and derived
portions retain [MIT, copyright (c) 2022 ekarad1um](Vendor/Photogrammetry/LICENSE).
Apple's Object Capture engine is a system framework, not an open-source component
of this repository.
