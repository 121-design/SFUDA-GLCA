import os
import copy
import csv
import faiss
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from model.SFUniDA import SFUniDA
from dataset.dataset import BearingDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

from config1.model_config import build_args
from util.net_utils import set_logger, set_random_seed
from util.net_utils import compute_h_score_with_private_discovery, Entropy, label_matching

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr0"] * decay
        param_group["weight_decay"] = 1e-3
        param_group["momentum"] = 0.9
        param_group["nesterov"] = True
    return optimizer


class BearingSubset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x1, x2, label, pri, _ = self.base[self.indices[i]]
        return x1, x2, label, pri, i


def split_indices_by_label(raw_labels, train_ratio=0.5):
    raw_labels = np.asarray(raw_labels)
    idxs = np.arange(len(raw_labels))
    train_idx = []
    test_idx = []
    for lbl in np.unique(raw_labels):
        cls_idx = idxs[raw_labels == lbl]
        np.random.shuffle(cls_idx)
        n_train = int(len(cls_idx) * train_ratio)
        if len(cls_idx) > 1:
            n_train = max(1, min(n_train, len(cls_idx) - 1))
        else:
            n_train = len(cls_idx)
        train_idx.extend(cls_idx[:n_train])
        test_idx.extend(cls_idx[n_train:])
    return train_idx, test_idx


def build_pred_labels(pred_cls_all, open_flag, open_thresh):
    pred_label_all = torch.max(pred_cls_all, dim=1)[1]
    if open_flag:
        cls_num = pred_cls_all.shape[1]
        pred_unc_all = Entropy(pred_cls_all) / np.log(cls_num)
        unc_idx = torch.where(pred_unc_all > open_thresh)[0]
        pred_label_all[unc_idx] = cls_num
    return pred_label_all


def build_confusion_data(args, pred_cls_all, embed_feat_all, gt_label_all, gt_private_all):
    shared_num = args.shared_class_num
    sp_num = args.source_private_class_num
    tp_num = args.target_private_class_num

    label_names = [str(x) for x in args.shared_labels]
    if sp_num > 0:
        label_names += [str(x) for x in args.source_private_labels]
    if tp_num > 0:
        label_names += [str(x) for x in args.target_private_labels]

    total_num = len(label_names)
    label_ids = list(range(total_num))

    gt_conf = gt_label_all.clone()
    if tp_num > 0:
        pri_mask = gt_private_all >= 0
        gt_conf[pri_mask] = shared_num + sp_num + gt_private_all[pri_mask]

    pred_label = torch.max(pred_cls_all, dim=1)[1]
    pred_conf = pred_label.clone()

    if tp_num > 0:
        pred_unc_all = Entropy(pred_cls_all) / np.log(pred_cls_all.shape[1])
        unk_mask = pred_unc_all > args.w_0

        if unk_mask.sum() > 0:
            feat_unk = embed_feat_all[unk_mask].cpu().numpy()
            if tp_num <= 1 or feat_unk.shape[0] < tp_num:
                cluster_labels = np.zeros(unk_mask.sum().item(), dtype=np.int64)
            else:
                kmeans = KMeans(n_clusters=tp_num, random_state=0).fit(feat_unk)
                cluster_labels = kmeans.labels_

            gt_private_unk = gt_private_all[unk_mask].cpu().numpy()
            valid_mask = gt_private_unk >= 0

            if valid_mask.sum() > 0 and tp_num > 1:
                mapped_valid = label_matching(gt_private_unk[valid_mask], cluster_labels[valid_mask])
                mapping = {}
                for c, m in zip(cluster_labels[valid_mask], mapped_valid):
                    if c not in mapping:
                        mapping[c] = m
                mapped = []
                for c in cluster_labels:
                    m = mapping.get(c, c)
                    if m < 0:
                        m = c
                    mapped.append(m)
                mapped = np.asarray(mapped, dtype=np.int64)
            else:
                mapped = cluster_labels

            pred_conf[unk_mask] = shared_num + sp_num + torch.from_numpy(mapped).to(pred_conf.device)

    return gt_conf.cpu().numpy(), pred_conf.cpu().numpy(), label_ids, label_names


def save_confusion_matrix(cm, labels, save_path, normalize=False):
    if normalize:
        cm = cm.astype(np.float32)
        row_sum = cm.sum(axis=1, keepdims=True) + 1e-6
        cm = cm / row_sum

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix" + (" (Norm)" if normalize else ""))
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.ylabel("True")
    plt.xlabel("Pred")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_tsne_plot(feat, labels, save_path, max_points=3000):
    feat = np.asarray(feat)
    labels = np.asarray(labels)

    if feat.shape[0] > max_points:
        idx = np.random.choice(feat.shape[0], max_points, replace=False)
        feat = feat[idx]
        labels = labels[idx]

    tsne = TSNE(n_components=2, init="pca", random_state=0)
    feat_2d = tsne.fit_transform(feat)

    csv_path = save_path.replace(".png", ".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "label"])
        for (x, y), lab in zip(feat_2d, labels):
            writer.writerow([x, y, int(lab)])

    plt.figure(figsize=(6, 5))
    plt.scatter(feat_2d[:, 0], feat_2d[:, 1], c=labels, cmap="tab20", s=6)
    plt.title("t-SNE")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_metrics_curve(metrics, save_dir):
    csv_path = os.path.join(save_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "h_score", "known_acc", "unknown_acc", "ncd_acc", "acc"])
        for row in metrics:
            writer.writerow([row["epoch"], row["h_score"], row["known_acc"], row["unknown_acc"], row["ncd_acc"], row["acc"]])

    epochs = [m["epoch"] for m in metrics]
    h = [m["h_score"] for m in metrics]
    k = [m["known_acc"] for m in metrics]
    u = [m["unknown_acc"] for m in metrics]
    n = [m["ncd_acc"] for m in metrics]
    a = [m["acc"] for m in metrics]

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, h, label="H")
    plt.plot(epochs, k, label="Known")
    plt.plot(epochs, u, label="Unknown")
    plt.plot(epochs, n, label="NCD")
    plt.plot(epochs, a, label="ACC")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics.png"), dpi=300)
    plt.close()


best_KK = None


def reset_global_state():
    global best_KK
    best_KK = None


@torch.no_grad()
def obtain_feature_banks(args, model, dataloader):
    model.eval()

    data_num = len(dataloader.dataset)
    pred_cls_bank = torch.zeros(data_num, args.class_num).cuda()
    embed_feat_bank = torch.zeros(data_num, args.embed_feat_dim).cuda()

    for _, x_test, _, _, idx in tqdm(dataloader, ncols=60):
        x_test = x_test.cuda(non_blocking=True)
        idx = idx.cuda(non_blocking=True)

        embed_feat, pred_cls = model(x_test, apply_softmax=True)
        pred_cls_bank[idx] = pred_cls
        embed_feat_bank[idx] = embed_feat

    embed_feat_bank = embed_feat_bank / (torch.norm(embed_feat_bank, p=2, dim=1, keepdim=True) + 1e-8)
    return pred_cls_bank, embed_feat_bank


def train(args, model, train_dataloader, test_dataloader, optimizer, epoch_idx=0.0):
    model.eval()
    pred_cls_bank, embed_feat_bank = obtain_feature_banks(args, model, test_dataloader)

    model.train()
    local_KNN = args.local_K

    all_pred_loss_stack = []
    psd_pred_loss_stack = []
    knn_pred_loss_stack = []
    reg_pred_loss_stack = []

    iter_idx = epoch_idx * len(train_dataloader)
    iter_max = args.epochs * len(train_dataloader)

    global best_KK
    if best_KK is None:
        if args.batch_size <= 1:
            best_KK = 1
        else:
            best_KK = max(2, min(args.class_num, args.batch_size - 1))

    for x_train, _, _, _, idx in tqdm(train_dataloader, ncols=60):
        iter_idx += 1
        idx = idx.cuda()
        x_train = x_train.cuda()

        embed_feat, pred_cls = model(x_train, apply_softmax=True)
        embed_feat = embed_feat / (torch.norm(embed_feat, p=2, dim=-1, keepdim=True) + 1e-8)
        embed_feat_detach = embed_feat.detach()

        with torch.no_grad():
            pred_cls_bank[idx] = pred_cls
            embed_feat_bank[idx] = embed_feat_detach

            feat_dist = torch.einsum("bd, nd -> bn", embed_feat_detach, embed_feat_bank)
            nn_feat_idx = torch.topk(feat_dist, k=local_KNN + 1, dim=-1, largest=True)[1]
            nn_feat_idx = nn_feat_idx[:, 1:]

            nn_pred_cls = torch.mean(pred_cls_bank[nn_feat_idx], dim=1)
            nn_embed_feat = embed_feat_bank[nn_feat_idx]

        pos_feat_simi = torch.einsum("bd, bkd -> bk", embed_feat, nn_embed_feat)
        neg_feat_simi = torch.einsum("bd, nd -> bn", embed_feat, embed_feat_detach)
        neg_feat_simi = torch.sort(neg_feat_simi, dim=-1, descending=True)[0][:, 1:]

        cur_bs = embed_feat.size(0)
        hard_neg_start_idx = int(np.ceil(cur_bs / max(best_KK, 1)))
        hard_neg_start_idx = min(hard_neg_start_idx, max(0, neg_feat_simi.size(1) - local_KNN))

        reg_pred_loss = torch.mean(
            torch.sum(neg_feat_simi[:, hard_neg_start_idx:hard_neg_start_idx + local_KNN], dim=-1) -
            torch.sum(pos_feat_simi, dim=-1)
        )
        knn_pred_loss = torch.mean(torch.sum(-nn_pred_cls * torch.log(pred_cls + 1e-5), dim=-1))
        psd_pred_loss = torch.tensor(0.0, device=pred_cls.device)

        loss = args.lam_knn * knn_pred_loss + args.lam_reg * reg_pred_loss
        lr_scheduler(optimizer, iter_idx, iter_max)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_pred_loss_stack.append(loss.item())
        psd_pred_loss_stack.append(float(psd_pred_loss.item()))
        knn_pred_loss_stack.append(knn_pred_loss.item())
        reg_pred_loss_stack.append(reg_pred_loss.item())

    return {
        "all_pred_loss": np.mean(all_pred_loss_stack),
        "psd_pred_loss": np.mean(psd_pred_loss_stack),
        "knn_pred_loss": np.mean(knn_pred_loss_stack),
        "reg_pred_loss": np.mean(reg_pred_loss_stack),
    }


@torch.no_grad()
def test(args, model, dataloader, src_flg=False, return_details=False):
    model.eval()
    gt_label_stack = []
    gt_private_stack = []
    pred_cls_stack = []
    embed_feat_stack = []

    if src_flg:
        class_list = args.source_class_list
        open_flg = False
    else:
        class_list = args.target_class_list
        open_flg = args.target_private_class_num > 0

    for _, x_test, y, pri, _ in tqdm(dataloader, ncols=60):
        x_test = x_test.cuda()
        embed_feat, pred_cls = model(x_test, apply_softmax=True)
        gt_label_stack.append(y)
        pred_cls_stack.append(pred_cls.cpu())
        gt_private_stack.append(pri)
        embed_feat_stack.append(embed_feat.cpu())

    gt_label_all = torch.cat(gt_label_stack, dim=0)
    pred_cls_all = torch.cat(pred_cls_stack, dim=0)
    gt_private_all = torch.cat(gt_private_stack, dim=0)
    embed_feat_all = torch.cat(embed_feat_stack, dim=0)

    h_score, known_acc, unknown_acc, novel_discovery_acc = compute_h_score_with_private_discovery(
        args, class_list, gt_label_all, pred_cls_all,
        gt_private_all, embed_feat_all,
        open_flg, pred_unc_all=None, open_thresh=args.w_0
    )

    gt_c, pred_c, _, _ = build_confusion_data(
        args, pred_cls_all, embed_feat_all, gt_label_all, gt_private_all
    )
    acc = float(np.mean(gt_c == pred_c)) if len(gt_c) > 0 else 0.0

    if not return_details:
        return h_score, known_acc, unknown_acc, novel_discovery_acc, acc

    details = {
        "gt_label_all": gt_label_all,
        "pred_cls_all": pred_cls_all,
        "gt_private_all": gt_private_all,
        "embed_feat_all": embed_feat_all,
    }
    return h_score, known_acc, unknown_acc, novel_discovery_acc, acc, details


def build_target_model(args):
    model = SFUniDA(args)
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError("Please set source checkpoint for target adaptation.")
    model = model.cuda()

    for _, v in model.class_layer.named_parameters():
        v.requires_grad = False
    return model


def build_target_optimizer(args, model):
    param_group = []
    for _, v in model.backbone_layer.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 0.1}]
    for _, v in model.feat_embed_layer.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    optimizer = torch.optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    return optimizer


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    this_dir = os.path.join(os.path.dirname(__file__), ".")

    if not hasattr(args, "lam_knn"):
        args.lam_knn = 1.0
    if not hasattr(args, "lam_reg"):
        args.lam_reg = 0.1
    if not hasattr(args, "local_K"):
        args.local_K = 5
    if not hasattr(args, "w_0"):
        args.w_0 = 0.4

    model = SFUniDA(args)
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError("Please set source checkpoint for target adaptation.")

    model = model.cuda()
    save_dir = os.path.join(this_dir, "checkpoints_glc_plus", args.dataset, "model1",
                            args.target_label_type, args.note)
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir
    args.logger = set_logger(args, log_name="log_model1_training.txt")

    target_dataset = BearingDataset(args, args.target_files, d_type="target")
    train_ratio = float(getattr(args, "target_train_ratio", 0.5))
    train_ratio = min(max(train_ratio, 0.0), 1.0)

    raw_labels = np.asarray(target_dataset.index_raw_label)
    train_idx, test_idx = split_indices_by_label(raw_labels, train_ratio=train_ratio)

    target_train_dataset = BearingSubset(target_dataset, train_idx)
    target_test_dataset = BearingSubset(target_dataset, test_idx)

    target_train_dataloader = DataLoader(target_train_dataset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers, drop_last=True)
    target_test_dataloader = DataLoader(target_test_dataset, batch_size=args.batch_size * 2, shuffle=False,
                                        num_workers=args.num_workers, drop_last=False)

    reset_global_state()
    model = build_target_model(args)
    optimizer = build_target_optimizer(args, model)

    metrics = []
    best_score = -1.0
    best_state = None
    best_epoch = -1

    for epoch_idx in tqdm(range(args.epochs), ncols=60):
        loss_dict = train(args, model, target_train_dataloader, target_train_dataloader, optimizer, epoch_idx)
        hscore, knownacc, unknownacc, ncd_acc, acc = test(args, model, target_test_dataloader, src_flg=False)

        metrics.append({
            "epoch": epoch_idx + 1,
            "h_score": float(hscore),
            "known_acc": float(knownacc),
            "unknown_acc": float(unknownacc),
            "ncd_acc": float(ncd_acc),
            "acc": float(acc),
        })

        args.logger.info("Epoch:{}/{} all:{:.3f} psd:{:.3f} knn:{:.3f} reg:{:.3f}".format(
            epoch_idx + 1, args.epochs,
            loss_dict["all_pred_loss"], loss_dict["psd_pred_loss"],
            loss_dict["knn_pred_loss"], loss_dict["reg_pred_loss"]))
        args.logger.info("Current: H:{:.3f}, Known:{:.3f}, Unknown:{:.3f}, NCD:{:.3f}, ACC:{:.3f}".format(
            hscore, knownacc, unknownacc, ncd_acc, acc))

        cur_score = knownacc if args.target_private_class_num == 0 else hscore
        if cur_score >= best_score:
            best_score = cur_score
            best_epoch = epoch_idx + 1
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"epoch": epoch_idx, "model_state_dict": model.state_dict()},
                       os.path.join(save_dir, "best_model1_checkpoint.pth"))

    save_metrics_curve(metrics, save_dir)

    viz_dir = os.path.join(save_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    hscore, knownacc, unknownacc, ncd_acc, acc, details = test(
        args, model, target_test_dataloader, src_flg=False, return_details=True
    )
    gt_c, pred_c, label_ids, label_names = build_confusion_data(
        args, details["pred_cls_all"], details["embed_feat_all"],
        details["gt_label_all"], details["gt_private_all"]
    )
    cm = confusion_matrix(gt_c, pred_c, labels=label_ids)
    np.savetxt(os.path.join(viz_dir, "confusion_last.csv"), cm, delimiter=",", fmt="%d")
    save_confusion_matrix(cm, label_names, os.path.join(viz_dir, "confusion_last.png"), normalize=False)
    save_confusion_matrix(cm, label_names, os.path.join(viz_dir, "confusion_last_norm.png"), normalize=True)
    np.savetxt(os.path.join(viz_dir, "confusion_last_norm.csv"), cm.astype(np.float32) /
               (cm.sum(axis=1, keepdims=True) + 1e-6), delimiter=",", fmt="%.4f")

    save_tsne_plot(details["embed_feat_all"].numpy(), gt_c,
                   os.path.join(viz_dir, "tsne_last_gt.png"))
    save_tsne_plot(details["embed_feat_all"].numpy(), pred_c,
                   os.path.join(viz_dir, "tsne_last_pred.png"))

    if best_state is not None:
        model.load_state_dict(best_state)
        hscore, knownacc, unknownacc, ncd_acc, acc, details = test(
            args, model, target_test_dataloader, src_flg=False, return_details=True
        )
        gt_c, pred_c, label_ids, label_names = build_confusion_data(
            args, details["pred_cls_all"], details["embed_feat_all"],
            details["gt_label_all"], details["gt_private_all"]
        )
        cm = confusion_matrix(gt_c, pred_c, labels=label_ids)
        np.savetxt(os.path.join(viz_dir, "confusion_best.csv"), cm, delimiter=",", fmt="%d")
        save_confusion_matrix(cm, label_names, os.path.join(viz_dir, "confusion_best.png"), normalize=False)
        save_confusion_matrix(cm, label_names, os.path.join(viz_dir, "confusion_best_norm.png"), normalize=True)
        np.savetxt(os.path.join(viz_dir, "confusion_best_norm.csv"), cm.astype(np.float32) /
                   (cm.sum(axis=1, keepdims=True) + 1e-6), delimiter=",", fmt="%.4f")

        save_tsne_plot(details["embed_feat_all"].numpy(), gt_c,
                       os.path.join(viz_dir, "tsne_best_gt.png"))
        save_tsne_plot(details["embed_feat_all"].numpy(), pred_c,
                       os.path.join(viz_dir, "tsne_best_pred.png"))

    args.logger.info("Best epoch: {}".format(best_epoch))


if __name__ == "__main__":
    args = build_args()
    set_random_seed(args.seed)
    args.checkpoint = os.path.join("checkpoints_glc_plus", args.dataset, "source_{}".format(args.task),
                                   "source_{}_{}".format(args.source_train_type, args.target_label_type),
                                   "latest_source_checkpoint.pth")
    main(args)
