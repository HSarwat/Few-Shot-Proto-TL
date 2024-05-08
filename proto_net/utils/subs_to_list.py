from processing.constants import ALL_ACTION_LIST, device
import numpy as np
import torch
from joblib import load

'''
This function takes the input data, which is a nested dictionary and turns it into a list.
'''
def subs_to_list(data, subs, trials, ft):
    sc = load(f'D:\Few-Shot Proto TL\processing\Data\scaler.bin')

    if not isinstance(subs, list):
        subs = [subs]
    X = []
    x = []
    for j, action in enumerate(ALL_ACTION_LIST):
        k = []
        X.append(k)
        x.append(k)
        for trial in trials:
            for sub in subs:
                X[j].append([data.data_dict[sub][action][trial][sensor] for sensor in ["FMG", "IMU", "EMG"]])

        FMG_input = np.vstack([X[j][i][0] for i in range(len(X[j]))]).astype("float")
        IMU_input = np.vstack([X[j][i][1] for i in range(len(X[j]))]).astype("float")
        EMG_input = np.vstack([X[j][i][2] for i in range(len(X[j]))]).astype("float")

        x[j] = []
        x[j].append(IMU_input)
        x[j].append(FMG_input)
        x[j].append(EMG_input)
        x[j] = np.hstack(x[j])
        x[j] = sc.transform(x[j])
        x[j] = ft.transform(x[j])

    x = np.array(x)

    x = torch.from_numpy(x).to(device, torch.float32)
    return x




