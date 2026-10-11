"""Smooth ordinary surface patches while keeping sharp rims and boundaries fixed."""
import numpy as np
import trimesh
from scipy.ndimage import map_coordinates


def silhouette_dominated_vertices(vertices, origin, spacing, field, hull):
    """Find vertices whose surrounding field is constrained by the visual hull.

    Fusion intersects depth with the hull using max(depth, hull). Interpolated
    equality identifies hull-dominated cells without an object-specific mask.
    The tolerance is only for roundoff, not a dilation or a hole-fill radius.
    """
    coordinates = ((vertices - origin) / spacing).T
    combined = map_coordinates(field, coordinates, order=1, mode="nearest", prefilter=False)
    silhouette = map_coordinates(hull, coordinates, order=1, mode="nearest", prefilter=False)
    return np.abs(combined - silhouette) <= spacing * 1e-6


def smooth_preserving_features(mesh, protected_region):
    """Keep thin-wall rims from opening further during Taubin smoothing.

    This does not fill holes or add geometry. Vertices on edges with a dihedral
    of at least 45 degrees in the supplied silhouette-dominated region, and on
    boundary/nonmanifold edges, stay at their extracted positions. All other
    vertices use the existing three passes. Depth-dominated features retain the
    old smoothing behavior: pinning these can reduce photographic coverage.
    Photo support must be recomputed on the resulting surface by the caller.
    """
    protected_region = np.asarray(protected_region, dtype=bool)
    if protected_region.shape != (len(mesh.vertices),):
        raise ValueError("Surface protection mask must match mesh vertices")
    sharp = mesh.face_adjacency_edges[mesh.face_adjacency_angles >= np.pi / 4]
    sharp = sharp[protected_region[sharp].any(axis=1)]
    counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    boundary = mesh.edges_unique[counts != 2]
    pinned = np.unique(np.concatenate((sharp.reshape(-1), boundary.reshape(-1))))
    laplacian = trimesh.smoothing.laplacian_calculation(mesh, pinned_vertices=pinned)
    before = mesh.vertices.copy()
    trimesh.smoothing.filter_taubin(mesh, lamb=.5, nu=.53, iterations=3,
                                   laplacian_operator=laplacian)
    if not np.array_equal(mesh.vertices[pinned], before[pinned]):
        raise RuntimeError("Surface smoothing displaced a protected edge")
    return {"policy": "hull-feature-pinned-taubin-v1", "sharpEdgeDegrees": 45,
            "silhouetteDominatedVertices": int(protected_region.sum()),
            "pinnedVertices": int(len(pinned)), "vertices": len(before),
            "sharpEdges": int(len(sharp)), "boundaryOrNonmanifoldEdges": int(len(boundary)),
            "lambda": .5, "nu": .53, "passes": 3}
