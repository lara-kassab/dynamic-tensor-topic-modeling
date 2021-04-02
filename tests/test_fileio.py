import os

from covid19.fileio import LinesFromGzipFiles


def test_iter_lines_from_gzip_files(datadir):
    filepaths = [os.path.join(datadir, "file1.gz"), os.path.join(datadir, "file2.gz")]
    lines = LinesFromGzipFiles(filepaths)

    assert list(lines) == [
        b"file 1 line 1\n",
        b"file 1 line 2\n",
        b"file 2 line 1\n",
        b"file 2 line 2\n",
    ]

    # Make sure the iterable can be reused.
    assert list(lines) == [
        b"file 1 line 1\n",
        b"file 1 line 2\n",
        b"file 2 line 1\n",
        b"file 2 line 2\n",
    ]

    assert len(lines) == 4
