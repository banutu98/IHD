import os
import time


def get_file_names(file):
    files_names = list()
    with open(file) as f:
        for line in f:
            if 'ID' not in line:
                continue
            files_names.append(line.split(',', 1)[0] + '.dcm')
    return files_names


if __name__ == '__main__':
    files = get_file_names(r'D:\Proiecte\IHD\data\train\labels.csv')  # path to labels.csv file
    TRAIN_DIR = r'D:\Proiecte\IHD\data\train'
    TEST_DIR = r'D:\Proiecte\IHD\data\test'
    command = 'kaggle competitions download rsna-intracranial-hemorrhage-detection -f '
    for index in range(len(files[:1000])):
        try:
            if index % 10 == 0:
                time.sleep(2)
            os.chdir(TRAIN_DIR)
            msg = os.popen(command + 'stage_1_train_images/' + files[index]).read()
            if '404' in msg:
                os.chdir(TEST_DIR)
                os.popen(command + 'stage_1_test_images/' + files[index]).read()
        except Exception as e:
            print(e)
            continue
