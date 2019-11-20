import os
import yaml
import time
from datetime import datetime
import shutil
import torch
import random
import argparse
import numpy as np
import logging

from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.utils import data

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter, running_side_score
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer

# Set this to True to increase training speed in PyTorch 1.0 by around x3
# However, it might need more storage on the GPU
# from torch.backends import cudnn
torch.backends.cudnn.benchmark = True

# Set maximum number of validation samples to add to summaries
NUM_IMG_SAMPLES = 3


def train(cfg, writer, logger):

    # Setup random seeds
    torch.manual_seed(cfg.get('seed', 1860))
    torch.cuda.manual_seed(cfg.get('seed', 1860))
    np.random.seed(cfg.get('seed', 1860))
    random.seed(cfg.get('seed', 1860))

    # Setup device
    if cfg["device"]["use_gpu"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU instead!")
    else:
        device = torch.device("cpu")

    # Setup augmentations
    augmentations = cfg['training'].get('augmentations', None)
    data_aug = get_composed_augmentations(augmentations)
    if "rcrop" in augmentations.keys():
        data_aug_val = get_composed_augmentations({"rcrop": augmentations["rcrop"]})

    # Setup dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']
    if 'depth_scaling' not in cfg['data'].keys():
        cfg['data']['depth_scaling'] = None
    if 'max_depth' not in cfg['data'].keys():
        logger.warning("Key d_max not found in configuration file! Using default value")
        cfg['data']['max_depth'] = 256
    if 'min_depth' not in cfg['data'].keys():
        logger.warning("Key d_min not found in configuration file! Using default value")
        cfg['data']['min_depth'] = 1
    t_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['train_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        augmentations=data_aug,
        depth_scaling=cfg['data']['depth_scaling'],
        n_bins=cfg['data']['depth_bins'],
        max_depth=cfg['data']['max_depth'],
        min_depth=cfg['data']['min_depth'])

    v_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['val_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        augmentations=data_aug_val,
        depth_scaling=cfg['data']['depth_scaling'],
        n_bins=cfg['data']['depth_bins'],
        max_depth=cfg['data']['max_depth'],
        min_depth=cfg['data']['min_depth'])

    trainloader = data.DataLoader(t_loader,
                                  batch_size=cfg['training']['batch_size'],
                                  num_workers=cfg['training']['n_workers'],
                                  shuffle=True,
                                  drop_last=True)

    valloader = data.DataLoader(v_loader,
                                batch_size=cfg['validation']['batch_size'],
                                num_workers=cfg['validation']['n_workers'],
                                shuffle=True,
                                drop_last=True)

    # Check selected tasks
    if sum(cfg["data"]["tasks"].values()) > 1:
        logger.info("Running multi-task training with config: {}".format(
                    cfg["data"]["tasks"]))

    # Get output dimension of the network's final layer
    n_classes_d_cls = None
    if cfg["data"]["tasks"]["d_cls"]:
        n_classes_d_cls = t_loader.n_classes_d_cls

    # Setup metrics for validation
    if cfg["data"]["tasks"]["d_cls"]:
        running_metrics_val_d_cls = runningScore(n_classes_d_cls)
    if cfg["data"]["tasks"]["d_reg"]:
        running_metrics_val_d_reg = running_side_score()

    # Setup model
    model = get_model(cfg['model'],
                      cfg["data"]["tasks"],
                      n_classes_d_cls=n_classes_d_cls).to(device)
    # model = d_regResNet().to(device)

    # Setup multi-GPU support
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        logger.info("Running multi-gpu training on {} GPUs".format(n_gpus))
        model = torch.nn.DataParallel(model, device_ids=range(n_gpus))

    # Setup multi-task loss
    task_weights = {}
    update_weights = True if \
        cfg["training"]["task_weight_policy"] == 'update' else False
    for task, weight in cfg["training"]["task_weight_init"].items():
        task_weights[task] = torch.tensor(weight).float()
        task_weights[task] = task_weights[task].to(device)
        task_weights[task] = task_weights[task].requires_grad_(update_weights)
    logger.info("Task weights were initialized with {}".format(
        cfg["training"]["task_weight_init"]))

    # Setup optimizer and lr_scheduler
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg['training']['optimizer'].items()
                        if k != 'name'}

    objective_params = list(model.parameters()) + list(task_weights.values())
    optimizer = optimizer_cls(objective_params, **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg['training']['lr_schedule'])
    logger.info("Using learning-rate scheduler {}".format(scheduler))

    # Setup task-specific loss functions
    # logger.debug("setting loss functions")
    loss_fns = {}
    for task, selected in cfg["data"]["tasks"].items():
        if selected:
            logger.info("Task " + task + " was selected for training.")
            loss_fn = get_loss_function(cfg, task)
            logger.info("Using loss function {} for task {}".format(
                        loss_fn, task))
            loss_fns[task] = loss_fn

    # Load weights from old checkpoint if set
    # logger.debug("checking for resume checkpoint")
    start_iter = 0
    if cfg['training']['resume'] is not None:
        if os.path.isfile(cfg['training']['resume']):
            logger.info("Loading model and optimizer from checkpoint '{}'"
                        .format(cfg['training']['resume']))
            logger.info("Loading file...")
            checkpoint = torch.load(cfg['training']['resume'], map_location="cpu")
            logger.info("Loading model...")
            model.load_state_dict(checkpoint["model_state"])
            model.to("cpu")
            model.to(device)
            logger.info("Restoring task weights...")
            task_weights = checkpoint["task_weights"]
            for task, state in task_weights.items():
                # task_weights[task] = state.to(device)
                task_weights[task] = torch.tensor(state.data).float()
                task_weights[task] = task_weights[task].to(device)
                task_weights[task] = task_weights[task].requires_grad_(update_weights)
            logger.info("Loading scheduler...")
            scheduler.load_state_dict(checkpoint["scheduler_state"])
#            scheduler.to("cpu")
            start_iter = checkpoint["iteration"]

            # Add loaded parameters to optimizer
            # NOTE task_weights will not update otherwise!
            logger.info("Loading optimizer...")
            optimizer_cls = get_optimizer(cfg)
            objective_params = list(model.parameters()) + \
                list(task_weights.values())
            optimizer = optimizer_cls(objective_params, **optimizer_params)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            # for state in optimizer.state.values():
            #     for k, v in state.items():
            #         if torch.is_tensor(v):
            #             state[k] = v.to(device)

            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg['training']['resume'], checkpoint["iteration"]
                )
            )
        else:
            logger.error("No checkpoint found at '{}'. Re-initializing params!"
                         .format(cfg['training']['resume']))

    # Initialize meters for various metrics
    # logger.debug("initializing metrics")
    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    # Setup other utility variables
    i = start_iter
    flag = True
    timer_training_start = time.time()

    logger.info("Starting training phase...")

    logger.debug("model device cuda?")
    logger.debug(next(model.parameters()).is_cuda)
    logger.debug("d_reg weight device:")
    logger.debug(task_weights["d_reg"].device)
    logger.debug("cls weight device:")
    logger.debug(task_weights["d_cls"].device)

    while i <= cfg['training']['train_iters'] and flag:
        for (images, labels) in trainloader:

            start_ts = time.time()
            scheduler.step()
            model.train()

            # Forward pass
            # logger.debug("sending images to device")
            images = images.to(device)
            optimizer.zero_grad()
            # logger.debug("forward pass")
            outputs = model(images)

            # Clip predicted depth to min/max
            # logger.debug("clamping outputs")
            if cfg["data"]["tasks"]["d_reg"]:
                if cfg["data"]["depth_scaling"] is not None:
                    if cfg["data"]["depth_scaling"] == "clip":
                        logger.warning("Using deprecated clip function!")
                        outputs["d_reg"] = torch.clamp(outputs["d_reg"], 0, cfg["data"]["max_depth"])

            # Calculate single-task losses
            # logger.debug("calculate loss")
            st_loss = {}
            for task, loss_fn in loss_fns.items():
                labels[task] = labels[task].to(device)
                st_loss[task] = loss_fn(input=outputs[task],
                                        target=labels[task])

            # Calculate multi-task loss
            # logger.debug("calculate mt loss")
            mt_loss = 0
            if len(st_loss) > 1:
                for task, loss in st_loss.items():
                    s = task_weights[task]       # s := log(sigma^2)
                    r = s * 0.5                  # regularization term
                    if task in ["d_cls"]:
                        w = torch.exp(-s)        # weighting (class.)
                    elif task in ["d_reg"]:
                        w = 0.5 * torch.exp(-s)  # weighting (regr.)
                    else:
                        raise ValueError("Weighting not implemented!")
                    mt_loss += loss * w + r
            else:
                mt_loss = list(st_loss.values())[0]

            # Backward pass
            # logger.debug("backward pass")
            mt_loss.backward()
            # logger.debug("update weights")
            optimizer.step()

            time_meter.update(time.time() - start_ts)

            # Output current training status
            # logger.debug("write log")
            if i == 0 or (i + 1) % cfg['training']['print_interval'] == 0:
                pad = str(len(str(cfg['training']['train_iters'])))
                print_str = ("Training Iteration: [{:>" + pad + "d}/{:d}]"
                             + "  Loss: {:>14.4f}"
                             + "  Time/Image: {:>7.4f}").format(
                                i + 1,
                                cfg['training']['train_iters'],
                                mt_loss.item(),
                                time_meter.avg / cfg['training']['batch_size'])
                logger.info(print_str)

                # Add training status to summaries
                writer.add_scalar('learning_rate',
                                  scheduler.get_lr()[0],
                                  i + 1)
                writer.add_scalar('batch_size',
                                  cfg['training']['batch_size'],
                                  i + 1)
                writer.add_scalar('loss/train_loss', mt_loss.item(), i + 1)
                for task, loss in st_loss.items():
                    writer.add_scalar("loss/single_task/" + task, loss, i + 1)
                for task, weight in task_weights.items():
                    writer.add_scalar("task_weights/" + task, weight, i + 1)
                time_meter.reset()

                # Add latest input image to summaries
                train_input = images[0].cpu().numpy()[::-1, :, :]
                writer.add_image("training/input", train_input, i + 1)

                # Add d_cls predictions and gt for latest sample to summaries
                if cfg["data"]["tasks"]["d_cls"]:
                    train_pred = outputs["d_cls"].detach().cpu().numpy().max(0)[1].astype(np.uint8)
                    # train_pred = np.array(outputs["d_cls"][0].data.max(0)[1],
                    #                       dtype=np.uint8)
                    train_pred = t_loader.decode_segmap(train_pred)
                    train_pred = torch.tensor(np.rollaxis(train_pred, 2, 0))
                    writer.add_image("training/d_cls/prediction",
                                     train_pred,
                                     i + 1)

                    train_gt = t_loader.decode_segmap(
                        labels["d_cls"][0].data.cpu().numpy())
                    train_gt = torch.tensor(np.rollaxis(train_gt, 2, 0))
                    writer.add_image("training/d_cls/label", train_gt, i + 1)

                # Add d_reg predictions and gt for latest sample to summaries
                if cfg["data"]["tasks"]["d_reg"]:
                    train_pred = outputs["d_reg"][0]
                    train_pred = np.array(train_pred.data.cpu().numpy())
                    train_pred = t_loader.visualize_depths(
                        t_loader.restore_metric_depths(train_pred))
                    writer.add_image("training/d_reg/prediction",
                                     train_pred,
                                     i + 1)

                    train_gt = labels["d_reg"][0].data.cpu().numpy()
                    train_gt = t_loader.visualize_depths(
                        t_loader.restore_metric_depths(train_gt))
                    if len(train_gt.shape) < 3:
                        train_gt = np.expand_dims(train_gt, axis=0)
                    writer.add_image("training/d_reg/label", train_gt, i + 1)

            # Run mid-training validation
            if (i + 1) % cfg['training']['val_interval'] == 0:
                # or (i + 1) == cfg['training']['train_iters']:

                # Output current status
                # logger.debug("Training phase took " + str(timedelta(seconds=time.time() - timer_training_start)))
                timer_validation_start = time.time()
                logger.info("Validating model at training iteration"
                            + " {}...".format(i + 1))

                # Evaluate validation set
                model.eval()
                with torch.no_grad():
                    i_val = 0
                    pbar = tqdm(total=len(valloader), unit="batch")
                    for (images_val, labels_val) in valloader:

                        # Forward pass
                        images_val = images_val.to(device)
                        outputs_val = model(images_val)

                        # Clip predicted depth to min/max
                        if cfg["data"]["tasks"]["d_reg"]:
                            if cfg["data"]["depth_scaling"] is None:
                                logger.warning("Using deprecated clip function!")
                                outputs_val["d_reg"] = torch.clamp(outputs_val["d_reg"],
                                                              0, cfg["data"]["max_depth"])
                            else:
                                outputs_val["d_reg"] = torch.clamp(outputs_val["d_reg"],
                                                              0, 1)

                        # Calculate single-task losses
                        st_loss_val = {}
                        for task, loss_fn in loss_fns.items():
                            labels_val[task] = labels_val[task].to(device)
                            st_loss_val[task] = loss_fn(
                                input=outputs_val[task],
                                target=labels_val[task])

                        # Calculate multi-task loss
                        mt_loss_val = 0
                        if len(st_loss) > 1:
                            for task, loss_val in st_loss_val.items():
                                s = task_weights[task]
                                r = s * 0.5
                                if task in ["d_cls"]:
                                    w = torch.exp(-s)
                                elif task in ["d_reg"]:
                                    w = 0.5 * torch.exp(-s)
                                else:
                                    raise ValueError("Weighting not implemented!")
                                mt_loss_val += loss_val * w + r
                        else:
                            mt_loss_val = list(st_loss.values())[0]

                        # Accumulate metrics for summaries
                        val_loss_meter.update(mt_loss_val.item())

                        if cfg["data"]["tasks"]["d_cls"]:
                            running_metrics_val_d_cls.update(
                                labels_val["d_cls"].data.cpu().numpy(),
                                outputs_val["d_cls"].data.cpu().numpy().argmax(1))

                        if cfg["data"]["tasks"]["d_reg"]:
                            running_metrics_val_d_reg.update(
                                v_loader.restore_metric_depths(
                                    outputs_val["d_reg"].data.cpu().numpy()),
                                v_loader.restore_metric_depths(
                                    labels_val["d_reg"].data.cpu().numpy()))

                        # Update progressbar
                        i_val += 1
                        pbar.update()

                        # Stop validation early if max_iter key is set
                        if "max_iter" in cfg["validation"].keys() and \
                                i_val >= cfg["validation"]["max_iter"]:
                            logger.warning("Stopped validation early "
                                           + "because max_iter was reached")
                            break

                # Add sample input images from latest batch to summaries
                num_img_samples_val = min(len(images_val), NUM_IMG_SAMPLES)
                for cur_s in range(0, num_img_samples_val):
                    val_input = images_val[cur_s].cpu().numpy()[::-1, :, :]
                    writer.add_image("validation_sample_" + str(cur_s + 1)
                                     + "/input", val_input, i + 1)

                # Add predictions/ground-truth for d_cls to summaries
                    if cfg["data"]["tasks"]["d_cls"]:
                        val_pred = outputs_val["d_cls"][cur_s].data.max(0)[1]
                        val_pred = np.array(val_pred, dtype=np.uint8)
                        val_pred = t_loader.decode_segmap(val_pred)
                        val_pred = torch.tensor(np.rollaxis(val_pred, 2, 0))
                        writer.add_image("validation_sample_" + str(cur_s + 1)
                                         + "/prediction_d_cls",
                                         val_pred,
                                         i + 1)
                        val_gt = t_loader.decode_segmap(
                            labels_val["d_cls"][cur_s].data.cpu().numpy())
                        val_gt = torch.tensor(np.rollaxis(val_gt, 2, 0))
                        writer.add_image("validation_sample_" + str(cur_s + 1)
                                         + "/label_d_cls",
                                         val_gt,
                                         i + 1)

                # Add predictions/ground-truth for d_reg to summaries
                    if cfg["data"]["tasks"]["d_reg"]:
                        val_pred = outputs_val["d_reg"][cur_s].cpu().numpy()
                        val_pred = v_loader.visualize_depths(
                            v_loader.restore_metric_depths(val_pred))
                        writer.add_image("validation_sample_" + str(cur_s + 1)
                                         + "/prediction_d_reg",
                                         val_pred,
                                         i + 1)

                        val_gt = labels_val["d_reg"][cur_s].data.cpu().numpy()
                        val_gt = v_loader.visualize_depths(
                            v_loader.restore_metric_depths(val_gt))
                        if len(val_gt.shape) < 3:
                            val_gt = np.expand_dims(val_gt, axis=0)
                        writer.add_image("validation_sample_" + str(cur_s + 1)
                                         + "/label_d_reg",
                                         val_gt,
                                         i + 1)

                # Add evaluation metrics for d_cls predictions to summaries
                if cfg["data"]["tasks"]["d_cls"]:
                    score, class_iou = running_metrics_val_d_cls.get_scores()
                    for k, v in score.items():
                        writer.add_scalar(
                            'validation/d_cls_metrics/{}'.format(k[:-3]),
                            v, i + 1)
                        for k, v in class_iou.items():
                            writer.add_scalar(
                                'validation/d_cls_metrics/class_{}'.format(k),
                                v, i + 1)
                    running_metrics_val_d_cls.reset()

                # Add evaluation metrics for d_reg predictions to summaries
                if cfg["data"]["tasks"]["d_reg"]:
                    writer.add_scalar('validation/d_reg_metrics/rel',
                                      running_metrics_val_d_reg.rel,
                                      i + 1)
                    running_metrics_val_d_reg.reset()

                # Add validation loss to summaries
                writer.add_scalar('loss/val_loss', val_loss_meter.avg, i + 1)

                # Output current status
                logger.info(("Validation Loss at Iteration {}: "
                             + "{:>14.4f}").format(i + 1,
                                                   val_loss_meter.avg))
                val_loss_meter.reset()
                # logger.debug("Validation phase took {}".format(timedelta(seconds=time.time() - timer_validation_start)))
                timer_training_start = time.time()

                # Close progressbar
                pbar.close()

            # Save checkpoint
            if (i + 1) % cfg['training']['checkpoint_interval'] == 0 or \
               (i + 1) == cfg['training']['train_iters'] or \
               i == 0:
                state = {
                    "iteration": i + 1,
                    "model_state": model.state_dict(),
                    "task_weights": task_weights,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict()
                }
                save_path = os.path.join(writer.file_writer.get_logdir(),
                                         "{}_{}_checkpoint_iter_".format(
                                            cfg['model']['arch'],
                                            cfg['data']['dataset'])
                                         + str(i + 1) + ".pkl")
                torch.save(state, save_path)
                logger.info("Saved checkpoint at iteration {} to: {}".format(
                                i + 1, save_path))

            # Stop training if current iteration == max iterations
            if (i + 1) == cfg['training']['train_iters']:
                flag = False
                break

            i += 1


if __name__ == "__main__":

    # Get config file from arguments and read it
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/example_config.yml",
        help="Configuration file to use"
    )
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    # Set logdir
    if cfg["logging"]["log_name"] == "id":
        run_id = random.randint(1, 100000)
        log_name = str(run_id)
    elif cfg["logging"]["log_name"] == "timestamp":
        log_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    else:
        log_name = cfg["logging"]["log_name"]
    logdir = os.path.join(cfg["logging"]["log_dir"],
                          os.path.basename(args.config)[:-4], log_name)
    writer = SummaryWriter(log_dir=logdir)

    # Setup logger
    log_lvls = {"debug": logging.DEBUG,
                "info": logging.INFO,
                "warning": logging.WARNING}
    logger = get_logger(logdir, lvl=log_lvls[cfg["logging"]["log_level"]])
    logger.info("Set logging level to " + str(logger.level))
    logger.info("Saving logs and checkpoints to {}".format(logdir))
    shutil.copy(args.config, logdir)

    # Start training
    logger.info('Starting training')
    train(cfg, writer, logger)
