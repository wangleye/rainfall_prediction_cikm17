from pymongo import MongoClient
import numpy as np

db = MongoClient().rainfall_sz
TRAIN_DATA_PATH = "E:/CIKM2017_train/train.txt"
TEST_DATA_PATH = "E:/CIKM2017_train/testA.txt"


def create_idx_train():
    for t in range(15):
        for h in range(4):
            print("create indext for t{}h{}".format(t, h))
            db["train_t{}h{}".format(t, h)].create_index("train_id")


def create_idx_test():
    for t in range(15):
        for h in range(4):
            print("create indext for t{}h{}".format(t, h))
            db["test_t{}h{}".format(t, h)].create_index("test_id")


def insert_to_table(train_id, y, x):
    rainfall = y
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            time_slot = i
            height = j
            spatial_data = x[i, j].tolist()
            insert_data = {
                "train_id": train_id,
                "rainfall": rainfall,
                "time_slot": time_slot,
                "height": height,
                "spatial_data": spatial_data
            }
            table = db["train_t{}h{}".format(time_slot, height)]
            table.insert_one(insert_data)


def insert_test_to_table(test_id, x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            time_slot = i
            height = j
            spatial_data = x[i, j].tolist()
            insert_data = {
                "test_id": test_id,
                "time_slot": time_slot,
                "height": height,
                "spatial_data": spatial_data
            }
            table = db["test_t{}h{}".format(time_slot, height)]
            table.insert_one(insert_data)


def load_training_data_to_db():
    """
    load training data from the file
    """
    # raw image x,y dimensions are 101 * 101
    with open(TRAIN_DATA_PATH) as input_train:
        for line_train in input_train:
            words = line_train.split(',')
            train_id = words[0]
            print(train_id)
            train_y = float(words[1])
            xs = words[2].split(' ')
            train_x = np.asarray(xs).reshape(15, 4, 101, 101)
            insert_to_table(train_id, train_y, train_x)


def load_test_data_to_db():
    """
    load test data from the file
    """
    # raw image x,y dimensions are 101 * 101
    with open(TEST_DATA_PATH) as input_test:
        for line_test in input_test:
            words = line_test.split(',')
            test_id = words[0]
            print(test_id)
            xs = words[2].split(' ')
            train_x = np.asarray(xs).reshape(15, 4, 101, 101)
            insert_test_to_table(test_id, train_x)


if __name__ == "__main__":
    load_test_data_to_db()
    create_idx_test()
