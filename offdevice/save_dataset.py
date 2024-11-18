import os
import numpy as np
import data_processing as dp
import dataset_definition as dtdef


def save_training_data(
    dataset, dataset_path, subject, session, compressed_method="minmax", nb_bits=8, time_length=25, save_folder_path=""
):
    """
    Save the training data for the tensorflow model

    @param dataset the class that represents the dataset
    @param dataset_path the path to the dataset
    @param subject the subject to use, must be 000, 001, ...
    @param session the session to use, must be 001, 002
    @param compressed_method the compression method used
    @param nb_bits the number of bits to compress to
    @param time_length the length of the time window to use in ms
    @param dimension the number of electrode in each axis (x,y)
    @param save_folder_path the path of the folder to save the data in

    """
    window_length = int(dataset.sampling_rate*time_length/1000)

    main_folder_path = save_folder_path
    if not os.path.exists(main_folder_path):
        os.makedirs(main_folder_path)

    data_array = dataset.load_dataset(dataset_path, subject, session)

    averages_data = dp.preprocess_data(data_array, window_length=window_length, filtering_utility=not dataset.utility_filtered)
    X, y = dp.extract_with_labels(averages_data)

    X_compressed = dp.compress_data(X, method=compressed_method, residual_bits=nb_bits)
    filename = "%s/%s_%s_%s_%sbits.npz" % (
        main_folder_path,
        subject,
        session,
        compressed_method,
        nb_bits
    )
    X_rolled = dp.roll_data(X_compressed, 2, v_dim=dataset.sensors_dim[0], h_dim=dataset.sensors_dim[1])

    # Copy the labels to be the same size as the data
    nb_rolled = int(np.floor(X_rolled.shape[0] / y.shape[0]))
    y_rolled = np.tile(y, nb_rolled)
    y_rolled = np.array(y_rolled, dtype=np.uint8)

    np.savez(filename, data=X_rolled, label=y_rolled)


def save_raw_data(dataset, dataset_path, subject, session, save_folder_path="dataset/raw/"):
    """
    Save the training data for the tensorflow model

    @param dataset the class that represents the dataset
    @param dataset_path the path to the dataset
    @param subject the subject to use, must be 000, 001, ...
    @param session the session to use, must be 001, 002
    @param save_folder_path the path of the folder to save the data in

    """
    main_folder_path = save_folder_path
    if not os.path.exists(main_folder_path):
        os.makedirs(main_folder_path)

    filename = "%s/%s_%s_raw.npz" % (main_folder_path, subject, session)
    data_array = dataset.load_dataset(dataset_path, subject, session)
    np.savez(filename, data=data_array)


if __name__ == "__main__":
    Emager = dtdef.EmagerDataset()
    dataset_path = "dataset/emager"
    subjects = ["00","01","02","03","04","05","06","07","08","09","10", "11"]
    sessions = ["1", "2"]
    bits = [3,4,5,6,7,8]
    compressed_methods = ["minmax", "msb", "smart", "root", "baseline"]
    #compressed_methods = ["baseline"]

    for subject in subjects:
        for session in sessions:
            for compressed_method in compressed_methods:
                for bit in bits:
                    save_training_data(
                        Emager,
                        dataset_path,
                        subject,
                        session,
                        compressed_method=compressed_method,
                        nb_bits=bit,
                        save_folder_path="dataset/train/emager/%s" % (compressed_method),
                    )
            save_raw_data(Emager, dataset_path, subject, session, "dataset/raw/emager/")

    # Capgmyo = dtdef.CapgmyoDataset()
    # dataset_path = "dataset/capgmyo"
    # subjects = ["01","02","03","04","05","06","07","08","09","10"]
    # sessions = ["1", "2"]
    # bits = [3,4,5,6,7,8]
    # compressed_methods = ["minmax", "msb", "smart", "root", "baseline"]
    # #compressed_methods = ["baseline"]

    # for subject in subjects:
    #     for session in sessions:
    #         for compressed_method in compressed_methods:
    #             for bit in bits:
    #                 save_training_data(
    #                     Capgmyo,
    #                     dataset_path,
    #                     subject,
    #                     session,
    #                     compressed_method=compressed_method,
    #                     nb_bits=bit,
    #                     save_folder_path="dataset/train/capgmyo/%s" % (compressed_method),
    #                 )
    #         save_raw_data(Capgmyo, dataset_path, subject, session, "dataset/raw/capgmyo/")