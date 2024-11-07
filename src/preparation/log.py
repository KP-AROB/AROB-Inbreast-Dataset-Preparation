
def log_dataset_statistics(df):
    print("\n\033[32mPreparing INBreast Dataset ... \033[0m\n")
    print("\n\033[32mDataset info\033[0m")
    print("Dataset columns : \n")
    for i in list(df.columns):
        print("    - {}".format(i))
    print("\n")
    print("Dataset length : {}".format(len(df)))
    return