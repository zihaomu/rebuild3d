import unittest

from rebuild3d_worker.failures import MLX_BACKEND, failure_fields


class FailureProtocolTests(unittest.TestCase):
    def test_pinned_mlx_allocator_failures_enable_existing_finite_retry(self):
        messages = {
            '[malloc] Unable to allocate 4294967296 bytes.': 'mlx-allocation-failed',
            '[metal::malloc] Attempting to allocate 8000000000 bytes which is greater than the maximum allowed buffer size of 4294967296 bytes.': 'mlx-buffer-size-limit',
            '[metal::malloc] Resource limit (499000) exceeded.': 'mlx-buffer-count-limit',
        }
        for message, expected in messages.items():
            with self.subTest(message=message):
                error = RuntimeError(message)
                report = failure_fields(error, MLX_BACKEND)
                self.assertEqual(report['status'], 'resource-limit')
                self.assertEqual(report['resourceFailureReason'], expected)
                self.assertEqual(report['error'], repr(error))

    def test_capability_integrity_disk_and_unrelated_errors_do_not_retry(self):
        errors = [
            RuntimeError('[metal::load_device] No Metal device available.'),
            RuntimeError('Failed to load the default metallib.'),
            RuntimeError('Checkpoint SHA-256 mismatch'),
            RuntimeError('Cannot allocate camera parameters: invalid shape'),
            RuntimeError('memory budget configuration invalid'),
            ValueError('[malloc] Unable to allocate 4294967296 bytes.'),
            OSError(28, 'No space left on device'),
            PermissionError('Operation not permitted'),
            KeyboardInterrupt(),
        ]
        for error in errors:
            with self.subTest(error=repr(error)):
                report = failure_fields(error, MLX_BACKEND)
                self.assertEqual(report['status'], 'failed')
                self.assertNotIn('resourceFailureReason', report)

    def test_mlx_specific_text_requires_verified_mlx_backend(self):
        error = RuntimeError('[malloc] Unable to allocate 4294967296 bytes.')
        for backend in [None, 'pytorch-mps-v3', 'unknown']:
            self.assertEqual(failure_fields(error, backend)['status'], 'failed')

    def test_host_and_existing_pytorch_allocation_failures_keep_retry(self):
        for error in [MemoryError(), RuntimeError('MPS backend out of memory (MPS allocated: 8 GB)')]:
            self.assertEqual(failure_fields(error)['status'], 'resource-limit')


if __name__ == '__main__':
    unittest.main()
