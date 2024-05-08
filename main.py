from proto_net.train import proto_train
import torch
import traceback

if __name__ == "__main__":
    try:
        params = {
            "lr": 0.0005,
            "bs": 20,
            "op": 72,
            "fn": 80,
            "optimizer": torch.optim.Adam,
            "shots": 1
        }
        results, results_std = proto_train(params)
        print(f"{results:.2f} ± {results_std:.2f}")

    except:
        traceback.print_exc()


