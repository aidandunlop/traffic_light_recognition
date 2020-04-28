# LISA dataset
# - script for moving all the training/testing images into same directories
import os
import sys
import getopt
from shutil import move, rmtree


def main(argv):
    LISA = ""
    try:
        opts, _ = getopt.getopt(argv, "hd:", ["dataset="])
    except getopt.GetoptError:
        print("merge_dataset.py -d <path_to_LISA_dataset>")
        sys.exit()

    if len(opts) < 1:
        print("merge_dataset.py -d <path_to_LISA_dataset>")
        sys.exit()

    for opt, arg in opts:
        if opt == "-h":
            print("merge_dataset.py -d <path_to_LISA_dataset>")
            sys.exit()
        elif opt in ("-d", "--dataset"):
            LISA = arg

    # check dataset folder exists
    if not os.path.isdir(LISA):
        print(LISA, "does not exist.")
        sys.exit()

    # create base images folder
    imagesDirectory = os.path.join(LISA, "images")
    if not os.path.exists(imagesDirectory):
        os.makedirs(imagesDirectory, exist_ok=True)

    # move all training images to base training folder
    trainingSequences = ["dayTrain", "nightTrain"]
    trainingPaths = map(lambda path: os.path.join(LISA, path, path), trainingSequences)

    testingSequences = [
        "daySequence1",
        "daySequence2",
        "nightSequence1",
        "nightSequence2",
    ]
    testingPaths = map(lambda path: os.path.join(LISA, path, path), testingSequences)

    if all(
        [not os.path.isdir(path) for path in list(trainingPaths) + list(testingPaths)]
    ):
        print("LISA Dataset not in the original format, exiting...")
        sys.exit()

    for directory in trainingPaths:
        for files in filter(lambda x: not x.startswith("."), os.listdir(directory)):
            filesPath = os.path.join(directory, files, "frames")
            for f in os.listdir(filesPath):
                path_file = os.path.join(filesPath, f)
                move(path_file, imagesDirectory)

    # move all testing images to base testing folder

    for directory in testingPaths:
        for files in filter(lambda x: not x.startswith("."), os.listdir(directory)):
            filesPath = os.path.join(directory, files)
            for f in os.listdir(filesPath):
                path_file = os.path.join(filesPath, f)
                move(path_file, imagesDirectory)

    # delete unneeded folders
    for directory in trainingSequences:
        rmtree(os.path.join(LISA, directory))

    for directory in testingSequences:
        rmtree(os.path.join(LISA, directory))


if __name__ == "__main__":
    main(sys.argv[1:])
