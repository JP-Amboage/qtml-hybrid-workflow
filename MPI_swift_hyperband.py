import shutil
from mpi4py import MPI
import platform
import click
from timeit import default_timer as timer

from math import floor
from math import ceil
from math import log

import numpy as np

from utils_MPI import fun_worker
from utils_MPI import send_stop_all_workers
from utils_MPI import top_k
from utils_MPI import config_with_smallest_loss

EPOCHS = 0
Predictor_class = None


def run_then_return_validation_loss(
    comm, T, size, r_prev, r_i, n_next, d, k, CURVES_TABLE, MODELS, known_curve=0.5
):
    global EPOCHS
    global Predictor_class
    avail_workers = [i for i in range(1, size)]
    all_workers = [i for i in range(1, size)]
    index = 0
    PARTIALLY_TRAINED = []
    FULLY_TRAINED = []
    n_fully_trained = 0
    n_sent_to_fully_train = 0
    # FULLY TRAIN A MINIMUM of TRIALS and PARTIALLY TRAIN the rest
    while True:
        # if more workers available than experiments left, remove workers
        while index + len(avail_workers) > len(T):
            removed_worker = avail_workers.pop()
            all_workers.remove(removed_worker)

        # send a partial train to every available worker
        for worker in avail_workers:
            # print("Inital sending config id {} to worker {}".format(T[index]["id"], worker))
            comm.send(
                {
                    "trial": T[index],
                    "iteration": 0,
                    "epochs": ceil((r_i - r_prev) * known_curve),
                },
                dest=worker,
            )
            EPOCHS = EPOCHS + ceil((r_i - r_prev) * known_curve)
            index = index + 1
        # afterwards no worker is available
        avail_workers = []

        # we will receive an answer for every working worker
        for _ in all_workers:
            status = MPI.Status()
            # collect the answer
            message = comm.recv(None, source=MPI.ANY_SOURCE, status=status)
            # collect the sender
            source = status.Get_source()
            # print("Received val acc {} from rank {}".format(message["trial"]['curve'][-1], source))

            # locate  and update the trial
            for i in range(len(T)):
                if T[i]["id"] == message["trial"]["id"]:
                    T[i] = message["trial"]
                    t = T[i]
                    break

            # if the answer is the result of a partially trained configuration
            if message["iteration"] == 1:
                # if we don't reach the minimum ammount of fully trained configurations
                if (
                    n_sent_to_fully_train < k * n_next
                    or n_sent_to_fully_train - n_fully_trained
                    < d - len(CURVES_TABLE[r_i])
                ):
                    # fully train the configuration
                    # print("CONTINUE config with id {} to worker {}".format(t['id'], source))
                    comm.send(
                        {"trial": t, "iteration": 1, "epochs": r_i - len(t["curve"])},
                        dest=source,
                    )
                    n_sent_to_fully_train = n_sent_to_fully_train + 1
                    EPOCHS = EPOCHS + r_i - len(t["curve"])
                else:
                    # we partially trained the config and we had already sent to fully train the ammount of needed configurations
                    # set the worker as free so that it can be used to partially train some other configuration (if there's any remaining)
                    avail_workers.append(source)
                    # we save the partially trained model to decide if it will fully trained later (once the minimun fully trained requirement is satisfied)
                    PARTIALLY_TRAINED.append(t)

            # if the answer if the result of a fully trained
            else:
                n_fully_trained = n_fully_trained + 1
                # print("Trial with id {} finished with acc {}".format(t['id'], t['curve'][-1]))
                # save config+learning curve in the table that will be used for training the SVR
                lr_curve = t["curve"]
                # save full config + curve in list
                CURVES_TABLE[r_i].append(lr_curve)
                # the worker is released
                avail_workers.append(source)
                FULLY_TRAINED.append(t)

        if index >= len(T) and set(avail_workers) == set(all_workers):
            # print("Stop")
            # print(index)
            break
    ##### end of partial trains and the needed fully trains

    # if its not trained and its possible to train it, train the performance predictor
    if (
        not r_prev + ceil((r_i - r_prev) * known_curve) in MODELS[r_i]
        and len(CURVES_TABLE[r_i]) >= d
    ):
        MODELS[r_i][r_prev + ceil((r_i - r_prev) * known_curve)] = Predictor_class()
        MODELS[r_i][r_prev + ceil((r_i - r_prev) * known_curve)].train(
            CURVES_TABLE[r_i], n_features=r_prev + ceil((r_i - r_prev) * known_curve)
        )

    predictions = []
    selected = []

    threshold = []
    for t in FULLY_TRAINED:
        threshold.append(t["curve"][-1])

    threshold = np.quantile(threshold, 0.25)

    # make the predictions for the partially trained models
    for i in range(len(PARTIALLY_TRAINED)):
        prediction = MODELS[r_i][r_prev + ceil((r_i - r_prev) * known_curve)].predict(
            PARTIALLY_TRAINED[i]["curve"]
        )
        predictions.append(prediction)
        # mark the trials that will be fully trained
        if prediction < threshold:
            selected.append(PARTIALLY_TRAINED[i])

    # print(predictions)
    sorted_idx = np.argsort(predictions)
    # print(sorted_idx.shape)
    # print(sorted_idx)

    # if the total ammount of trials that will be fully trained is lower than the number of trials that will make it to the next round, we select some more partially trained configurations to fully train them
    if len(selected) + n_fully_trained < n_next:
        selected = []
        for i in range(n_next - n_fully_trained):
            selected.append(PARTIALLY_TRAINED[sorted_idx[i]])
    # print(selected)

    """
	# we can fully train as many configurations as the number of worker we have "for free"
	if len(selected) > 0 and len(selected)%(size-1)!=0:
		num_to_be_selected = len(selected) + size - len(selected)%(size-1)
		if num_to_be_selected <= len(PARTIALLY_TRAINED):
			selected = []
			for i in range(num_to_be_selected):
				selected.append(PARTIALLY_TRAINED[sorted_idx[i]])
	"""
    # finish training for selected candidates

    # first reset woekers
    avail_workers = [i for i in range(1, size)]
    all_workers = [i for i in range(1, size)]
    index = 0
    while True:
        # if more workers available than experiments left, remove workers
        while index + len(avail_workers) > len(selected):
            removed_worker = avail_workers.pop()
            all_workers.remove(removed_worker)

        # send a configuration to be finished to every available worker
        for worker in avail_workers:
            # print("Final sending config id {} to worker {}, idx {}".format(selected[index]['id'], worker, index))
            comm.send(
                {
                    "trial": selected[index],
                    "iteration": 1,
                    "epochs": r_i - len(selected[index]["curve"]),
                },
                dest=worker,
            )
            EPOCHS = EPOCHS + r_i - len(selected[index]["curve"])
            index = index + 1

        # we will receive an answer for every working worker
        for _ in all_workers:
            status = MPI.Status()
            # collect the answer
            message = comm.recv(None, source=MPI.ANY_SOURCE, status=status)
            # collect the sender
            source = status.Get_source()
            # print("Received val acc {} from rank {}".format(message['trial']['curve'][-1], source))
            # locate  and update the trial
            for i in range(len(T)):
                if T[i]["id"] == message["trial"]["id"]:
                    T[i] = message["trial"]
                    t = T[i]
                    break

        if index >= len(selected) and set(avail_workers) == set(all_workers):
            # print("Stop")
            # print(index)
            break

    # losses list in the order of the Trials
    L = [np.inf] * len(T)

    for i in range(len(T)):
        # if the trial only was partially trained we set a high loss so that it gets killed
        if len(T[i]["curve"]) >= r_i:
            L[i] = T[i]["curve"][-1]
    # print(T)
    # print("========================================")
    return L, T


def swift_hyperband(
    R,  # Max resources allocated to any configuration
    eta,  # controls the proportion of configurations discarded in each round of SuccessiveHalving
    d,  # num of points required to train performance predictors
    k,  # proportion of models to train
    known_curve,
    configGenerator,
    comm,
    size,
):

    D = {}
    M = {}
    s_max = floor(log(R, eta))
    B = (s_max + 1) * R

    selected = []
    for s in range(s_max, -1, -1):
        n = ceil((B / R) * ((eta ** s) / (s + 1)))
        r = R * (eta ** -s)
        # Begin SuccessiveHalving with (n,r) inner loop
        print(f"=====Begin SuccessiveHalving with {(n,r)} inner loop=====")
        T = configGenerator.get_hyperparameter_configuration(n)
        r_prev = 0
        for i in range(0, s + 1, 1):
            n_i = floor(n * (eta ** -i))
            r_i = round(r * (eta ** i))
            if not r_i in D:
                D[r_i] = []
            if not r_i in M:
                M[r_i] = {}
            print((n_i, r_i))
            n_next = floor(n_i / eta) if i != s else 1
            L, T = run_then_return_validation_loss(
                comm=comm,
                T=T,
                size=size,
                r_prev=r_prev,
                r_i=r_i,
                n_next=n_next,
                d=d,
                k=k,
                CURVES_TABLE=D,
                MODELS=M,
                known_curve=known_curve,
            )
            if len(L) != len(T):
                print("PROBLEM!!!!")
            # print(T)
            T = top_k(T, L, max(floor(n_i / eta), 1))
            r_prev = r_i
        selected.append(T)

    selected = selected
    return config_with_smallest_loss(selected)


@click.command()
@click.option(
    "--dir_name",
    default="saved_models_swift",
    help="name of folder where models are saved",
)
@click.option("--seed", default=0, help="seed")
@click.option("--model_name", default="lstm", help="target model for the HPO process")
@click.option("--pred_type", default="quantum", help="quantum | classical")
@click.option("--save_models", default=False, help="False | True")
@click.option("--r", default=100, help="R value for Hyperband")
@click.option("--eta", default=2, help="eta value for Hyperband")
@click.option(
    "--d",
    default=15,
    help="minimum number of samples for training each performance predictor",
)
@click.option("--k", default=0.5, help="proportion of model to train in the round")
@click.option(
    "--known_curve", default=0.5, help="position of the extra decision points"
)
def main(dir_name, seed, model_name, pred_type, save_models, r, eta, d, k, known_curve):
    R = r
    global Predictor_class

    """
	CHANGE THIS TO ADD MORE MODELS
	"""
    if model_name == "cifar10_tf":
        from cifar10_tf_methods import ConfigGeneratorCifar10

        ConfigGenerator_class = ConfigGeneratorCifar10
        from cifar10_tf_methods import train_cifar

        train_model_method = train_cifar
    elif model_name == "cfiar10_torch":
        from cifar10_torch_methods import ConfigGeneratorCifar10

        ConfigGenerator_class = ConfigGeneratorCifar10
        from cifar10_tf_methods import train_cifar

        train_model_method = train_cifar
        
    elif model_name == 'tiny_img':
        from tiny_img_torch_methods import ConfigGeneratorTinyImg
        ConfigGenerator_class = ConfigGeneratorTinyImg
        from tiny_img_torch_methods import train_tinyimg
        train_model_method = train_tinyimg
        
    else:
        if model_name != "lstm":
            print("WARNING: model_name not valid -> using lstm as default choice")
        from lstm_methods import ConfigGeneratorLSTM

        ConfigGenerator_class = ConfigGeneratorLSTM
        from lstm_methods import train_LSTM

        train_model_method = train_LSTM
    """
	END OF THE MODEL DEPENDENT PART
	"""

    comm = MPI.COMM_WORLD
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    print(f"------> task {rank} running from {platform.node()}")
    if rank == 0:
        if pred_type == "classical":
            from predictor import Predictor

            Predictor_class = Predictor
        else:
            if pred_type != "quantum":
                print(
                    "WARNING: pred_type not valid -> using QPredictor as default choice"
                )
            from qpredictor import QPredictor

            Predictor_class = QPredictor

        rs = seed
        print(
            f"R={R}, eta={eta}, d={d}, k={k}, size={size}, rs={rs}, known_curve={known_curve}"
        )
        global EPOCHS
        EPOCHS = 0
        print("MPI WORLD SIZE: ", size)
        generator = ConfigGenerator_class(random_state=rs)

        start = timer()
        res = swift_hyperband(
            R=R,  # Max resources allocated to any configuration
            eta=eta,  # controls the proportion of configurations discarded in each round of SuccessiveHalving
            d=d,  # num of points required to train performance predictors
            k=k,  # proportion of models to train
            known_curve=known_curve,
            configGenerator=generator,
            comm=comm,
            size=size,
        )
        end = timer()

        print("#################################################################")
        print("#################################################################")
        # print out results
        print(f"Best loss found: {res[0]['curve'][-1]}")
        print(f"Needed time: {end-start}s")
        print(f"Total Epochs: {EPOCHS}")
        print(f"Config ID: {res[0]['id']}")
        print(
            f"R={R}, eta={eta}, d={d}, k={k}, size={size}, rs={rs}, known_curve={known_curve}"
        )
        print("#################################################################")
        print("#################################################################")

        if save_models != True:
            shutil.rmtree(dir_name)
        send_stop_all_workers(size, comm)

    else:
        fun_worker(
            rank=rank,
            dir_name=dir_name,
            comm=comm,
            train_model_method=train_model_method,
        )


if __name__ == "__main__":
    main()
