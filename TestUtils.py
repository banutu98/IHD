import unittest

import pandas as pd

import utilities.utils as utils


class TestUtils(unittest.TestCase):

    def test_label_columns(self):
        data = utils.get_csv_train("data/train")
        columns = list(data.columns)
        self.assertTrue(
            columns == ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural'])

    def test_metadata_columns(self):
        data = pd.read_csv("data/train/metadata_train.csv")
        columns = list(data.columns)[1:]
        columns.remove('id')
        columns = set(columns)
        metadata_columns = ['BitsAllocated', 'BitsStored', 'Columns', 'HighBit',
                            'Modality', 'PatientID', 'PhotometricInterpretation',
                            'PixelRepresentation', 'RescaleIntercept', 'RescaleSlope',
                            'Rows', 'SOPInstanceUID', 'SamplesPerPixel', 'SeriesInstanceUID',
                            'StudyID', 'StudyInstanceUID', 'ImagePositionPatient1',
                            'ImagePositionPatient2', 'ImagePositionPatient3',
                            'ImageOrientationPatient1', 'ImageOrientationPatient2',
                            'ImageOrientationPatient3', 'ImageOrientationPatient4',
                            'ImageOrientationPatient5', 'ImageOrientationPatient6',
                            'PixelSpacing1', 'PixelSpacing2']
        metadata_columns = set(metadata_columns)
        self.assertTrue(columns == metadata_columns)

    def test_combine_meta_labels(self):
        expected = pd.read_csv("data/train/train_meta.csv")
        result = utils.combine_labels_metadata("data/train")
        expected = list(expected.columns[1:])
        result = list(result)
        expected.remove('Unnamed: 0.1')
        result.remove('Unnamed: 0')
        self.assertTrue(result == expected)


if __name__ == '__main__':
    unittest.main()
