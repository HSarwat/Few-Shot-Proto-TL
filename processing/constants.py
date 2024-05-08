'''This files stores the constants of the project'''

# Enter your device's configuration for torch
device = 'cuda:0'

# Assign the path to the project files
import os
root_dir = os.path.dirname(os.path.abspath(__file__))

# (Optional)
# Assign the path to the pickle file that has the data
PICKLES_PATH = r"processing\Data"

# Assign the number of subjects you have
SUB_NUM = 20
ALL_SUB_LIST = [i for i in range(1, SUB_NUM+1)]

# Assign the number of trials performed by the subjects
TRIAL_NUM = 5
ALL_TRIAL_LIST = [i for i in range(1, TRIAL_NUM+1)]

# Assign the number of actions/gestures you need to classify
ACTION_NAMES = {1: "MF", 2: "ME", 3: "Rest", 4: "WVF", 5: "WDF", 6: "FP", 7: "FS"}
ALL_ACTION_LIST = [i for i in range(1,8)]
ACTION_NUM = len(ALL_ACTION_LIST)

# (Optional)
# This value is used to normalize the subjects
sup_subs = [9]

