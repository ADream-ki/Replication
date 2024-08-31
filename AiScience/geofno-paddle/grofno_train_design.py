import hydra
from os import path as osp
from omegaconf import DictConfig

from utilities3 import *
from catheter import *
import numpy as np
import ppsci
from ppsci.utils import logger
from ppsci.optimizer import Adam, lr_scheduler


# build data
def getdata(x_path, y_path, para_path, output_path, n_data, s, is_train=True):
    # load data
    inputX_raw = np.load(x_path)[:, 0:n_data]
    inputY_raw = np.load(y_path)[:, 0:n_data]
    inputPara_raw = np.load(para_path)[:, 0:n_data]
    output_raw = np.load(output_path)[:, 0:n_data]

    # preprocess data
    inputX = inputX_raw[:, 0::3]
    inputY = inputY_raw[:, 0::3]
    inputPara = inputPara_raw[:, 0::3]
    output = (output_raw[:, 0::3] + output_raw[:, 1::3] + output_raw[:, 2::3]
              ) / 3.0

    inputX = paddle.to_tensor(
        data=inputX, dtype='float32').transpose(perm=[1, 0])
    inputY = paddle.to_tensor(
        data=inputY, dtype='float32').transpose(perm=[1, 0])
    input = paddle.stack(x=[inputX, inputY], axis=-1)
    output = paddle.to_tensor(
        data=output, dtype='float32').transpose(perm=[1, 0])
    if (is_train):
        index = paddle.randperm(n=n_data)
        index = index[:n_data]

        x = paddle.index_select(input, index)
        y = paddle.index_select(output, index)
        x = x.reshape([n_data, s, 2])
    else:
        x = input.reshape([n_data, s, 2])
        y = output

    return {"x": x}, {"y": y}#输入与label


def train(cfg: DictConfig):
    # set random seed for reproducibility OK
    ppsci.utils.misc.set_random_seed(42)
    # initialize logger OK
    # logger.init_logger("ppsci", osp.join(
    #     cfg.output_dir, f"{cfg.mode}.log"), "info")

    # set model OK
    model = FNO1d(64, 64, padding=100, input_channel=2,
                  output_np=2001)

    # set dataloader config OK
    train_input, train_label = getdata("AiScience/geofno-paddle/data/training/x_1d_structured_mesh.npy", "AiScience/geofno-paddle/data/training/y_1d_structured_mesh.npy",
                                       "AiScience/geofno-paddle/data/training/data_info.npy", "AiScience/geofno-paddle/data/training/density_1d_data.npy", 3000, 2001, is_train=True)
    train_dataloader_cfg = {
        "name": "ContinuousNamedArrayDataset ",
        "input": train_input,
        "label": train_label,
    }

    # set constraint OK
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        dataloader_cfg={"dataset": train_dataloader_cfg,
                        "batch_size": 20,
                        "sampler": {
                            "name": "BatchSampler",
                            "drop_last": False,
                            "shuffle": True,
                        }, },
        loss=ppsci.loss.FunctionalLoss(
            LpLoss(size_average=False, name="pred")),
        output_expr={"pred": lambda out: paddle.exp(out["pred"])},
        name="geofno_train")

    # wrap constraints together
    constraint = {sup_constraint.name: sup_constraint}

    # set optimizer OK
    tmp_lr = lr_scheduler.Step(epochs=1,
                               iters_per_epoch=1,
                               learning_rate=0.001,
                               step_size=1,
                               gamma=0.5)()

    optimizer = Adam(learning_rate=tmp_lr,
                     weight_decay=0.0001)(model)

    # set validator OK
    test_input, test_label = getdata("AiScience/geofno-paddle/data/test/x_1d_structured_mesh.npy", "AiScience/geofno-paddle/data/test/y_1d_structured_mesh.npy",
                                     "AiScience/geofno-paddle/data/test/data_info.npy", "AiScience/geofno-paddle/data/test/density_1d_data.npy", 100, 2001, is_train=False)
    test_dataloader_cfg = {
        "name": "NamedArrayDataset",
        "input": test_input,
        "label": test_label,
    }
    eval_dataloader_cfg = {
        "dataset": test_dataloader_cfg,
        "batch_size": 20,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }

    test_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(
            LpLoss(size_average=False, name="pred")),
        output_expr={"pred": lambda out: paddle.exp(out["pred"])},
        metric={"TEST": ppsci.metric.FunctionalMetric(
            LpLoss(size_average=False, name="TEST"))},
        name="TEST_validator",
    )
    validator = {test_validator.name: test_validator}

    # initialize solver OK
    solver = ppsci.solver.Solver(
        model,
        constraint,
        "./",
        optimizer,
        None,
        1,
        1,
        save_freq=1,
        eval_during_train=False,
        eval_freq=20,
        validator=validator,
        eval_with_no_grad=True,)

    # train model OK
    solver.train()


# @hydra.main(version_base=None, config_path="./conf", config_name="geofno.yaml")
def main(cfg: DictConfig):
    # if cfg.mode == "train":
    train(cfg)
    # elif cfg.mode == "eval":
    #     evaluate(cfg)
    # elif cfg.mode == "export":
    #     export(cfg)
    # elif cfg.mode == "infer":
    #     inference(cfg)
    # else:
    #     raise ValueError(
    #         f"cfg.mode should in ['train', 'eval', 'export', 'infer'], but got '{cfg.mode}'"
    #     )


if __name__ == "__main__":
    paddle.seed(0)
    np.random.seed(0)
    paddle.set_device('gpu')
    main({})
