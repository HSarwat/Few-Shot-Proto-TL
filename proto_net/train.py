from proto_net.protonet.model import Protonet
from proto_net.utils.engine import Engine
from proto_net.protonet.dataLoader import ProtoNetDataLoader,ProtoNetDataSet
from processing.constants import *
from processing.data_structure import data_structure
from sklearn.model_selection import KFold
import numpy as np
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import torch
import traceback
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay



def proto_train(params):

    results = []

    '''Subs'''
    subs = np.arange(1,21)
    max_subs = len(subs)

    '''Load Data'''
    data = data_structure(pickle_name=f"D:\Few-Shot Proto TL\processing\Data\data.pkl", ask_force_reset=False)

    '''Create a Fold Split'''
    kf = KFold(max_subs, shuffle=False)
    fold_num = 0
    for train_subs_indices, test_subs_indices in kf.split(subs):


        '''Create model'''
        proto = Protonet.default_encoder(input=params["fn"], output=params["op"]).to('cuda:0')


        '''Assign test and train subject'''
        fold_num += 1
        test_subs = [subs[i] for i in test_subs_indices]
        train_subs = [subs[i] for i in train_subs_indices]

        '''Create Dimensionality Reduction model'''
        # The Feature Reduction needs to be done inside the dataloader due to
        # the way pytorch handles its data. This 'features' is then given
        # as an input parameter to the dataloader. Can be replaced with your
        # dimensionality reduction technique
        features = data.dimensionality_reduction(params['fn'])

        '''Assign train dataloader with seg, ft, and fn'''
        ds_train = ProtoNetDataSet(data=data,query_subs=train_subs, sup_subs=test_subs, trials=ALL_TRIAL_LIST,
                                n_shots=params['shots'], ft=features)
        dl_train = ProtoNetDataLoader(ds_train, batch_size=20,shuffle=False)

        '''Train model'''
        config = {'lr':params["lr"]}
        eng = Engine()
        eng.train(model=proto, loader=dl_train, optim_method=params["optimizer"], optim_config=config,max_epoch=200)


        '''Assign train dataloader with ft, and fn'''
        ds_test = ProtoNetDataSet(data=data,query_subs=test_subs, sup_subs=test_subs, trials=ALL_TRIAL_LIST,
                                  n_shots=params["shots"], ft=features)
        dl_test = ProtoNetDataLoader(ds_test, batch_size=params["bs"],shuffle=False)

        '''Predict output'''
        out = eng.evaluate(model=proto, loader=dl_test, desc=None)
        sum_acc = 0
        ln = len(out['outputs'])
        y_test = []
        y_pred = []
        for i in range(ln):
            sum_acc+=out['outputs'][i].get('acc')
            y_test.append(out['outputs'][i].get('y_true'))
            y_pred.append(out['outputs'][i].get('y_pred'))

        # y_test = np.hstack(y_test)
        # y_pred = np.hstack(y_pred)

        score = (sum_acc/ln)*100
        results.append(score)

    mean_results = np.mean(results)
    results_std = np.std(results)
    return mean_results, results_std

if __name__ == "__main__":
    try:
        params = {
            "ft": "features",
            "fn": 80,
            "seg": 0.2,
            "lr": 0.0005,
            "bs": 20,
            "op": 72,
            "optimizer": torch.optim.Adam,
            "shots" : 1
        }
        results, results_std = proto_train(params)
        print(f"{results:.2f} ± {results_std:.2f}")

    except:
        traceback.print_exc()


