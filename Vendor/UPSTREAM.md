# Upstream provenance

- Repository: https://github.com/ekarad1um/Photogrammetry
- Commit: `43c66741e3f912ddd29292858d6f6bc01441606e`
- License: MIT, copyright (c) 2022 ekarad1um.
- Imported: 2026-10-08.

`Photogrammetry/` preserves the upstream application sources, Xcode project,
README, and license without modification. It is a reference baseline and is not
compiled into the Rebuild3D Swift package. Rebuild3D's reconstruction service and
RealityKit viewer derive from the upstream session and preview architecture.
The refactor preserves models in project storage and waits for session completion.

Rebuild3D's repository license remains Apache-2.0. The upstream code and derived
portions retain the MIT notice in `Photogrammetry/LICENSE`; the app build copies
both licenses into the application bundle.
