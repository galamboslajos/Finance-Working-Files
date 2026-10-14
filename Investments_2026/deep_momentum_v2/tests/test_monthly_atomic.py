"""Atomic publication races and failed-write cleanup, using test-only evidence."""
import errno
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from dmv2.workflow import save_monthly_sample
from test_monthly_workflow import fixture_run


class AtomicPublicationTests(unittest.TestCase):
    def test_identical_concurrent_publication_is_verified_for_both_platform_errors(self):
        for error_code in (errno.EEXIST, errno.ENOTEMPTY):
            with self.subTest(error_code=error_code), tempfile.TemporaryDirectory() as temporary:
                def competing_writer(staged, destination):
                    # Another writer publishes the same complete bundle immediately
                    # before our rename. Darwin may report ENOTEMPTY instead of EEXIST.
                    shutil.copytree(staged, destination)
                    raise OSError(error_code, "Concurrent artifact already published")

                with patch.object(Path, "rename", new=competing_writer):
                    path = save_monthly_sample(temporary, fixture_run())
                self.assertTrue((path / "daily.parquet").is_file())
                self.assertTrue((path / "monthly.parquet").is_file())
                self.assertTrue((path / "audit.json").is_file())
                self.assertEqual(path, save_monthly_sample(temporary, fixture_run()))
                self.assertEqual(list(path.parent.glob(".monthly-tmp-*")), [])

    def test_unrelated_rename_failure_propagates_and_leaves_no_partial_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(Path, "rename", side_effect=OSError(errno.EACCES, "Denied")):
                with self.assertRaises(OSError) as caught:
                    save_monthly_sample(temporary, fixture_run())
            self.assertEqual(caught.exception.errno, errno.EACCES)
            parent = Path(temporary) / "outputs/notebook_audits"
            self.assertEqual(list(parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
