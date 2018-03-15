"""Test data_prepare.py."""
import unittest
from pathlib import Path
import shutil

from ..src.data_prepare import *


class TestDataPrepare(unittest.TestCase):
    """Test data_prepare."""

    # def test_split_data(self):
    #     testdir = Path('./dir')
    #     Path.mkdir(testdir)
    #     testfiles = ['{}.txt'.format(i) for i in range(10)]
    #     for f in testfiles:
    #         Path.touch(testdir / f)
    #     train, test = split_data(testdir)
    #     self.assertEqual(len(list(train.iterdir())), 8)
    #     self.assertEqual(len(list(test.iterdir())), 2)
    #     shutil.rmtree(testdir.as_posix())

    # def test_gen_data(self):
    #     pass

    # def test_get_category(self):
    #     self.assertEqual(get_category('AF0301_1100_00F.jpg'), 'AF0301')
    #     self.assertEqual(get_category('BM0607_1100_90L.jpg'), 'BM0607')

    # def test_rename_img(self):
    #     self.assertEqual(rename_img('AF0301_1100_00F.jpg'), 'AF0301.jpg')
    #     self.assertEqual(rename_img('BM0607_1100_90L.jpg'), 'BM0607.jpg')

    def test_load_data(self):
        train, _ = load_data('./sample_data')
        self.assertEqual(train.shape[0], int(49 * 0.8), 'Image number wrong!')
        self.assertEqual(train.shape[1], 250, 'Image width wrong!')

    def test_load_label(self):
        train, _ = load_label('')


if __name__ == '__main__':
    unittest.main()
