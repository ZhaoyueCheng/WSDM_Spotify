import glob

train_set_dir = "/media/data2/Data/wsdm2019/python/data/second_stage_test/"
new_file_dir = "/media/data2/Data/wsdm2019/python/data/second_stage_test_processed/"
train_files = sorted(glob.glob(train_set_dir+"*.csv"))

for test_file in train_files:
    s = ''
    with open(test_file, "r") as f:
        new_file = new_file_dir + test_file[-38:]
        for line in f:
            if line != "0\n":
                s += line

    with open(new_file, "a") as myfile:
        myfile.write(s)
