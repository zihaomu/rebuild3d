"""Normalize allocator failures into the application's finite resource retry.

These MLX patterns follow the pinned 0.32.2 Metal allocator, not arbitrary
messages containing 'memory' or 'allocate':
https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/allocator.cpp
"""
import re


MLX_BACKEND = 'mlx-aggregator-pytorch-heads-v1'
_MLX_LIMITS = (
    ('allocation-failed', re.compile(r'\[malloc\] Unable to allocate \d+ bytes\.')),
    ('buffer-size-limit', re.compile(
        r'\[metal::malloc\] Attempting to allocate \d+ bytes which is greater than'
        r' the maximum allowed buffer size of \d+ bytes\.')),
    ('buffer-count-limit', re.compile(r'\[metal::malloc\] Resource limit \(\d+\) exceeded\.')),
)


def resource_failure_reason(error, backend_id=None):
    """Return a narrow retry reason, or None for ordinary/capability failures."""
    if isinstance(error, MemoryError):
        return 'host-allocation-failed'
    if not isinstance(error, RuntimeError):
        return None
    message = str(error).strip()
    if type(error).__name__ == 'OutOfMemoryError' or message.startswith('MPS backend out of memory'):
        return 'pytorch-allocation-failed'
    if backend_id == MLX_BACKEND:
        for reason, pattern in _MLX_LIMITS:
            if pattern.fullmatch(message):
                return 'mlx-' + reason
    return None


def failure_fields(error, backend_id=None):
    reason = resource_failure_reason(error, backend_id)
    result = {'status': 'resource-limit' if reason else 'failed', 'error': repr(error)}
    if reason:
        result['resourceFailureReason'] = reason
    return result
