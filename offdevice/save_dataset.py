import os
import numpy as np
import data_processing as dp


def getData_EMG(path, user_id, session_nb, differential=False):
    """
    Load EMG data from EMaGer v1 dataset.

    Params:
        - path : path to EMaGer root
        - user_id : subject id
        - session_nb : session number

    Returns the loaded data with shape (nb_gesture, nb_repetition, time_length, num_channels)
    """

    # Parameters
    # user_id = "001"
    # session_nb = "000"
    nb_gesture = 6
    nb_repetition = 10
    nb_pts = 5000
    start_path = "%s/%s/session_%s/" % (path, user_id, session_nb)  # ordi UL
    data_array = np.zeros((nb_gesture, nb_repetition, 64, nb_pts), dtype=int)

    first_file = os.listdir(start_path)[0]
    arm_used = "right" if "right" in first_file else "left"
    for gest in range(nb_gesture):
        for rep in range(nb_repetition):
            path = (
                start_path
                + user_id
                + "-"
                + session_nb
                + "-00"
                + str(gest)
                + "-00"
                + str(rep)
                + "-"
                + arm_used
                + ".csv"
            )
            one_file = np.transpose(np.loadtxt(path, delimiter=","))
            data_array[gest, rep, :, :] = one_file[:, -nb_pts:]
    if differential:
        data_array = np.reshape(data_array, (nb_gesture, nb_repetition, 16, 4, nb_pts))
        final_array = data_array[:, :, :, 0:3, :] - data_array[:, :, :, 1:4, :]
        final_array = np.reshape(final_array, (nb_gesture, nb_repetition, 48, nb_pts))
    else:
        final_array = data_array

    return np.swapaxes(final_array, 2, 3)


def concat_gestures(path: str):
    """
    Load, process and concatenate all subjects and sessions from `path`.

    Params:
        - path : EMaGer dataset root

    Returns a list of shape: 6*[(n, 64)] where n is the number of samples loaded.
    """
    subjects = [f"{i:03d}" for i in range(12)]
    session = ["001", "002"]
    concat = [np.ndarray((0, 64)) for _ in range(6)]
    for sub in subjects:
        for ses in session:
            print(f"Processing subject {sub} session {ses}")
            try:
                data_array = getData_EMG(path, sub, ses, differential=False)
                procd = dp.preprocess_data(data_array)
                for i in range(6):
                    shaped = np.reshape(procd[i], (procd.shape[1] * procd.shape[2], 64))
                    concat[i] = np.vstack([concat[i], shaped])
                    # print(concat[i].shape)
            except FileNotFoundError:
                continue
    return concat


def save_training_data(
    dataset_path, subject, session, compressed_method="minmax", save_folder_path=""
):
    """
    Save the training data for the tensorflow model

    @param dataset_path the path to the dataset
    @param subject the subject to use, must be 000, 001, ...
    @param session the session to use, must be 001, 002
    @param time_length the length of the window
    @param save_folder_path the path of the folder to save the data in

    """
    main_folder_path = save_folder_path
    if not os.path.exists(main_folder_path):
        os.makedirs(main_folder_path)

    filename = "%s/%s_%s_%s.npz" % (
        main_folder_path,
        subject,
        session,
        compressed_method,
    )

    data_array = getData_EMG(dataset_path, subject, session, differential=False)
    averages_data = dp.preprocess_data(data_array)
    X, y = dp.extract_with_labels(averages_data)

    if compressed_method == "baseline":
        X_compressed = X
    else:
        X_compressed = dp.compress_data(X, method=compressed_method)
    X_rolled = dp.roll_data(X_compressed, 2)

    # Copy the labels to be the same size as the data
    nb_rolled = int(np.floor(X_rolled.shape[0] / y.shape[0]))
    y_rolled = np.tile(y, nb_rolled)
    y_rolled = np.array(y_rolled, dtype=np.uint8)

    np.savez(filename, data=X_rolled, label=y_rolled)


def save_raw_data(dataset_path, subject, session, save_folder_path="dataset/raw/"):
    """
    Save the training data for the tensorflow model

    @param dataset_path the path to the dataset
    @param subject the subject to use, must be 000, 001, ...
    @param session the session to use, must be 001, 002
    @param time_length the length of the window
    @param save_folder_path the path of the folder to save the data in

    """
    main_folder_path = save_folder_path
    if not os.path.exists(main_folder_path):
        os.makedirs(main_folder_path)

    filename = "%s/%s_%s_raw.npz" % (main_folder_path, subject, session)
    data_array = getData_EMG(dataset_path, subject, session, differential=False)
    np.savez(filename, data=data_array)


if __name__ == "__main__":
    dataset_path = "dataset/emager"
    subjects = ["000", "001", "002"]
    sessions = ["001", "002"]
    compressed_methods = ["minmax", "msb", "smart", "root", "baseline"]

    for subject in subjects:
        for session in sessions:
            for compressed_method in compressed_methods:
                save_training_data(
                    dataset_path,
                    subject,
                    session,
                    compressed_method=compressed_method,
                    save_folder_path="dataset/train/%s" % (compressed_method),
                )
            save_raw_data(dataset_path, subject, session)
