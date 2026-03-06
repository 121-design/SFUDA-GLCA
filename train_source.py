import os
import shutil
import torch
import numpy as np
from tqdm import tqdm
from model.SFUniDA import SFUniDA
from dataset.dataset import BearingDataset
from torch.utils.data.dataloader import DataLoader

from config1.model_config import build_args
from util.net_utils import set_logger, set_random_seed
from util.net_utils import compute_h_score, CrossEntropyLabelSmooth, Entropy


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def train(args, model, dataloader, criterion, optimizer, epoch_idx=0.0):
    model.train()
    loss_stack = []

    iter_idx = epoch_idx * len(dataloader)
    iter_max = args.epochs * len(dataloader)

    for x_train, _, y, _, _ in tqdm(dataloader, ncols=60):
        iter_idx += 1
        x_train = x_train.cuda()
        y = y.cuda()

        _, pred_cls = model(x_train, apply_softmax=True)
        y_onehot = torch.zeros_like(pred_cls).scatter(1, y.unsqueeze(1), 1)

        loss = criterion(pred_cls, y_onehot)

        lr_scheduler(optimizer, iter_idx, iter_max)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_stack.append(loss.cpu().item())

    return np.mean(loss_stack)


@torch.no_grad()
def test(args, model, dataloader, src_flg=True, open_thresh=0.5, grid_search=False):
    model.eval()
    gt_label_stack = []
    pred_cls_stack = []

    if src_flg:
        class_list = args.source_class_list
        open_flg = False
    else:
        class_list = args.target_class_list
        open_flg = args.target_private_class_num > 0

    for _, x_test, y, _, _ in tqdm(dataloader, ncols=60):
        x_test = x_test.cuda()
        _, pred_cls = model(x_test, apply_softmax=True)
        gt_label_stack.append(y)
        pred_cls_stack.append(pred_cls.cpu())

    gt_label_all = torch.cat(gt_label_stack, dim=0)
    pred_cls_all = torch.cat(pred_cls_stack, dim=0)

    if not grid_search:
        h_score, known_acc, unknown_acc, per_cls_acc = compute_h_score(
            args, class_list, gt_label_all, pred_cls_all, open_flg, None, open_thresh
        )

        # overall ACC
        pred_label_all = torch.max(pred_cls_all, dim=1)[1]
        if open_flg:
            cls_num = pred_cls_all.shape[1]
            pred_unc_all = Entropy(pred_cls_all) / np.log(cls_num)
            unc_idx = torch.where(pred_unc_all > open_thresh)[0]
            pred_label_all[unc_idx] = cls_num
        acc = float((pred_label_all == gt_label_all).float().mean().item())

        return h_score, known_acc, unknown_acc, per_cls_acc, acc

    else:
        thresh_list = np.arange(0.05, 1.00, 0.05)
        h_score_list = []
        known_acc_list = []
        unknown_acc_list = []
        per_cls_acc_list = []
        acc_list = []

        for thresh in thresh_list:
            h_score, known_acc, unknown_acc, per_cls_acc = compute_h_score(
                args, class_list, gt_label_all, pred_cls_all, open_flg, None, open_thresh=thresh
            )
            pred_label_all = torch.max(pred_cls_all, dim=1)[1]
            if open_flg:
                cls_num = pred_cls_all.shape[1]
                pred_unc_all = Entropy(pred_cls_all) / np.log(cls_num)
                unc_idx = torch.where(pred_unc_all > thresh)[0]
                pred_label_all[unc_idx] = cls_num
            acc = float((pred_label_all == gt_label_all).float().mean().item())

            h_score_list.append(h_score)
            known_acc_list.append(known_acc)
            unknown_acc_list.append(unknown_acc)
            per_cls_acc_list.append(per_cls_acc)
            acc_list.append(acc)

        return thresh_list, h_score_list, known_acc_list, unknown_acc_list, per_cls_acc_list, acc_list


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    this_dir = os.path.join(os.path.dirname(__file__), ".")

    model = SFUniDA(args)
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        save_dir = os.path.dirname(args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        save_dir = os.path.join(this_dir, "checkpoints_glc_plus", args.dataset, "source_{}".format(args.task),
                                "source_{}_{}".format(args.source_train_type, args.target_label_type))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    model.cuda()
    args.save_dir = save_dir
    shutil.copy("./train_source.py", os.path.join(args.save_dir, "train_source.py"))

    logger = set_logger(args, log_name="log_source_training.txt")

    params_group = []
    for _, v in model.backbone_layer.named_parameters():
        params_group += [{"params": v, 'lr': args.lr * 0.1}]
    for _, v in model.feat_embed_layer.named_parameters():
        params_group += [{"params": v, 'lr': args.lr}]
    for _, v in model.class_layer.named_parameters():
        params_group += [{"params": v, 'lr': args.lr}]

    optimizer = torch.optim.SGD(params_group)
    optimizer = op_copy(optimizer)

    source_dataset = BearingDataset(args, args.source_files, d_type="source")
    source_dataloader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, drop_last=True)

    target_dataset = BearingDataset(args, args.target_files, d_type="target")
    target_dataloader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, drop_last=False)

    if args.source_train_type == "smooth":
        criterion = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0.1, reduction=True)
    elif args.source_train_type == "vanilla":
        criterion = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0.0, reduction=True)
    else:
        raise ValueError("Unknown source_train_type:", args.source_train_type)

    logger.info("START TRAINING ON SOURCE: {} {}".format(args.task, args.target_label_type))

    for epoch_idx in tqdm(range(args.epochs), ncols=60):
        train_loss = train(args, model, source_dataloader, criterion, optimizer, epoch_idx)
        logger.info("Epoch:{}/{} train_loss:{:.3f}".format(epoch_idx, args.epochs, train_loss))

        source_h, source_known, source_unknown, src_per_cls_acc, source_acc = test(
            args, model, source_dataloader, src_flg=True
        )
        logger.info("SOURCE: H:{:.3f}, Known:{:.3f}, Unknown:{:.3f}, ACC:{:.3f}".format(
            source_h, source_known, source_unknown, source_acc
        ))

        torch.save({
            "epoch": epoch_idx,
            "model_state_dict": model.state_dict()}, os.path.join(save_dir, "latest_source_checkpoint.pth"))

    # evaluate on target (fixed threshold)
    h, k, u, _, acc = test(
    args, model, target_dataloader, src_flg=False, open_thresh=args.w_0, grid_search=False
    )
    logger.info("TARGET (fixed): H:{:.3f}, Known:{:.3f}, Unknown:{:.3f}, ACC:{:.3f}, Thresh:{:.3f}".format(
    h, k, u, acc, args.w_0
    ))
    # evaluate on target (grid search)
    thresh_list, hscore_list, knownacc_list, unknownacc_list, _, acc_list = test(
        args, model, target_dataloader, src_flg=False, grid_search=True
    )
    for idx, thresh in enumerate(thresh_list):
        logger.info("OpenThresh:{:.3f}, H:{:.3f}, Known:{:.3f}, Unknown:{:.3f}, ACC:{:.3f}".format(
            thresh, hscore_list[idx], knownacc_list[idx], unknownacc_list[idx], acc_list[idx]
        ))

if __name__ == "__main__":
    args = build_args()
    set_random_seed(args.seed)
    main(args)
