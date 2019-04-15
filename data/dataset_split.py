import os
import sys
import shutil
import getopt
import random

TRAIN_FOLDER = "train"
TEST_FOLDER = "test"

def split_dataset(ORIGIN_FOLDER="mlt", test_size=0.1, isImageAndLabel=True):
    print("ORIGIN_FOLDER: ", ORIGIN_FOLDER)
    print("test_size: ", test_size)
    print("isImageAndLabel: ", isImageAndLabel)
    print("********** Start Process **********")
    images_dir = ORIGIN_FOLDER
    labels_dir = ORIGIN_FOLDER
    if isImageAndLabel:
        images_dir = os.path.join(ORIGIN_FOLDER, "image")
        labels_dir = os.path.join(ORIGIN_FOLDER, "label")

    image_paths = get_all_image_path(images_dir)
    random.shuffle(image_paths)
    image_paths, label_paths, invalid_data=check_valid(image_paths, labels_dir)
    nums = len(image_paths)
    trainNum = nums * (1 - test_size)
    trainNum = int(trainNum)
    train_image_paths = image_paths[0:trainNum]
    train_label_paths = label_paths[0:trainNum]
    test_image_paths = image_paths[trainNum:nums]
    test_label_paths = image_paths[trainNum:nums]

    change_path(train_image_paths, train_label_paths, True)
    change_path(test_image_paths, test_label_paths, False)

    print("********** Process Success **********")
    print("train_data: ", len(train_image_paths))
    print("test_data: ", len(test_image_paths))
    print("invalid_data: ", len(invalid_data))
    for i in invalid_data:
        print(i)

def check_valid(image_paths, labels_dir):
    new_image_paths = []
    invalid_image_paths = []
    label_paths = []
    for path in image_paths:
        filename = os.path.basename(path)
        filename = filename.split('.')[0] + ".txt"
        filepath = os.path.join(labels_dir, filename)
        if os.path.isfile(filepath) and not os.path.getsize(filepath)==0:
            # print(path, filepath)
            new_image_paths.append(path)
            label_paths.append(filepath)
        else:
            invalid_image_paths.append(filepath)

    return new_image_paths, label_paths, invalid_image_paths

def get_all_image_path(images_dir):
    image_paths = []
    extens = ['jpg', 'png', 'jpeg']
    if os.path.exists(images_dir):
        path = os.listdir(images_dir)
        for p in path:
            if p.split('.')[-1] in extens:
                image_paths.append(os.path.join(images_dir, p))
    return image_paths

def change_path(image_paths, label_paths, isTrain=True):
    if isTrain:
        if not os.path.exists(TRAIN_FOLDER):
            os.mkdir(TRAIN_FOLDER)
        for i in range(0, len(image_paths)):
            move_image_path = os.path.join(TRAIN_FOLDER, os.path.basename(image_paths[i]))
            shutil.copyfile(image_paths[i], move_image_path)
            move_label_path = os.path.join(TRAIN_FOLDER, os.path.basename(label_paths[i]))
            shutil.copyfile(label_paths[i], move_label_path)
    else:
        if not os.path.exists(TEST_FOLDER):
            os.mkdir(TEST_FOLDER)
        for i in range(0, len(image_paths)):
            move_image_path = os.path.join(TEST_FOLDER, os.path.basename(image_paths[i]))
            shutil.copyfile(image_paths[i], move_image_path)
            move_label_path = os.path.join(TEST_FOLDER, os.path.basename(label_paths[i]))
            shutil.copyfile(label_paths[i], move_label_path)

def main(argv):
    origin_folder = "mlt"
    test_size = "0.1"
    ttsplit = "1"
    try:
        opts, args = getopt.getopt(argv, "h",longopts=["help","origin_folder=","test_size=","ttsplit="])
    except:
        print("Usage: dataset_split.py --origin_folder=<origin_folder> --test_size=<test_size> --ttsplit=<ttsplit>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("Usage: dataset_split.py --origin_folder=<origin_folder> --test_size=<test_size> --ttsplit=<ttsplit>")
            sys.exit()
        elif opt in ["--origin_folder",]:
            origin_folder = arg
        elif opt in ["--test_size",]:
            test_size = arg
        elif opt in ["--ttsplit",]:
            ttsplit = arg

    if not os.path.exists(origin_folder):
        print(origin_folder,"is not exist")
        return

    test_size = float(test_size)
    ttsplit = int(ttsplit)
    if ttsplit==1:
        ttsplit=True
    else:
        ttsplit=False

    split_dataset(origin_folder, test_size, ttsplit)


if __name__ == '__main__':
    split_dataset("mlt")