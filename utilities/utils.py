import os
import pandas as pd
import glob
import pydicom
import numpy as np


def print_error(message):
    c_red = '\033[91m'
    c_end = '\033[0m'
    print(c_red + message + c_end)


def get_csv_train(data_prefix='../data'):
    train_df = pd.read_csv(os.path.join(data_prefix, 'stage_1_train.csv'))
    train_df[['ID', 'subtype']] = train_df['ID'].str.rsplit('_', 1,
                                                            expand=True)
    train_df = train_df.rename(columns={'ID': 'id', 'Label': 'label'})
    train_df = pd.pivot_table(train_df, index='id',
                              columns='subtype', values='label')
    return train_df


def extract_csv_partition():
    df = get_csv_train()
    meta_data_train = combine_labels_metadata('../data/train')
    negative, positive = df.loc[df['any'] == 0], df.loc[df['any'] == 1]
    negative_study_uids = list(meta_data_train.query("any == 0")['StudyInstanceUID'])
    indices = np.arange(min(len(negative_study_uids), len(positive.index)))
    np.random.shuffle(indices)
    negative_study_uids = np.array(negative_study_uids)[indices]
    selected_negative_studies = meta_data_train.loc[meta_data_train['StudyInstanceUID'].isin(negative_study_uids)]
    selected_negative_studies = selected_negative_studies.drop(set(selected_negative_studies.columns).intersection(set(negative.columns)), axis=1)
    negative = negative.merge(selected_negative_studies, how='left', on='id').dropna()
    negative = negative.drop(selected_negative_studies.columns, axis=1)
    return pd.concat([positive, negative])


def extract_metadata(data_prefix='../data'):
    filenames = glob.glob(os.path.join(data_prefix, "*.dcm"))
    get_id = lambda p: os.path.splitext(os.path.basename(p))[0]
    ids = map(get_id, filenames)
    dcms = map(pydicom.dcmread, filenames)
    columns = ['BitsAllocated', 'BitsStored', 'Columns', 'HighBit',
               'Modality', 'PatientID', 'PhotometricInterpretation',
               'PixelRepresentation', 'RescaleIntercept', 'RescaleSlope',
               'Rows', 'SOPInstanceUID', 'SamplesPerPixel', 'SeriesInstanceUID',
               'StudyID', 'StudyInstanceUID', 'ImagePositionPatient',
               'ImageOrientationPatient', 'PixelSpacing']
    meta_dict = {col: [] for col in columns}
    for img in dcms:
        for col in columns:
            meta_dict[col].append(getattr(img, col))
    meta_df = pd.DataFrame(meta_dict)
    del meta_dict
    meta_df['id'] = pd.Series(ids, index=meta_df.index)
    split_cols = ['ImagePositionPatient1', 'ImagePositionPatient2',
                  'ImagePositionPatient3', 'ImageOrientationPatient1',
                  'ImageOrientationPatient2', 'ImageOrientationPatient3',
                  'ImageOrientationPatient4', 'ImageOrientationPatient5',
                  'ImageOrientationPatient6', 'PixelSpacing1',
                  'PixelSpacing2']
    meta_df[split_cols[:3]] = pd.DataFrame(meta_df.ImagePositionPatient.values.tolist())
    meta_df[split_cols[3:9]] = pd.DataFrame(meta_df.ImageOrientationPatient.values.tolist())
    meta_df[split_cols[9:]] = pd.DataFrame(meta_df.PixelSpacing.values.tolist())
    meta_df = meta_df.drop(['ImagePositionPatient', 'ImageOrientationPatient', 'PixelSpacing'], axis=1)
    return meta_df


def combine_labels_metadata(data_prefix='../data'):
    meta_df = extract_metadata(data_prefix)
    df = get_csv_train(data_prefix)
    df = df.merge(meta_df, how='left', on='id').dropna()
    df.sort_values(by='ImagePositionPatient3', inplace=True, ascending=False)
    df.to_csv(os.path.join(data_prefix, 'train_meta.csv'))
    return df


def get_study_sequences(data_prefix="../data"):
    df = pd.read_csv(os.path.join(data_prefix, "train_meta.csv"))
    sequences = df.groupby("StudyInstanceUID")['id'].apply(list)
    return sequences


if __name__ == '__main__':
    # partition = extract_csv_partition()
    # print(partition.index.values)
    print(combine_labels_metadata())
