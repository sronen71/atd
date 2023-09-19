import torch
import os
import numpy as np
import dicomsdl
import cv2
import glob
from monai.transforms import Resize
from torch import nn
import timm
import segmentation_models_pytorch as smp

from timm.layers.conv2d_same import Conv2dSame
from conv3d_same import Conv3dSame
import threading
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt


preprocessed1_dir = "preprocessed1"


def load_dicom_2d(path, resize=None):  # image_seg_size
    dicom = dicomsdl.open(path)
    data = dicom.pixelData(storedvalue=True)
    if dicom.PixelRepresentation == 1:
        bit_shift = dicom.BitsAllocated - dicom.BitsStored
        dtype = data.dtype
        data = (data << bit_shift).astype(dtype) >> bit_shift

    data = data * dicom.RescaleSlope + dicom.RescaleIntercept

    if resize:
        data = cv2.resize(data, (resize[0], resize[1]), interpolation=cv2.INTER_LINEAR)
    orient = dicom.ImageOrientationPatient
    pos = dicom.ImagePositionPatient
    data = np.transpose(data, [1, 0])

    return data, orient, pos


# data, o, p = load_dicom_2d(
#    "/kaggle/input/rsna-2023-abdominal-trauma-detection/train_images/3934/41894/434.dcm"
# )


def window_images(images, img_min=-300, img_max=400):
    images[images < img_min] = img_min
    images[images > img_max] = img_max

    # normalization
    images = (images - img_min) / (img_max - img_min)
    return (images * 255).astype(np.uint8)


def check_orient(paths):
    dicom1 = dicomsdl.open(paths[0])
    dicom2 = dicomsdl.open(paths[-1])
    orient = dicom1.ImageOrientationPatient
    positions = [dicom1.ImagePositionPatient, dicom2.ImagePositionPatient]
    imaging_axis = np.cross(orient[:3], orient[3:])
    dot = np.dot(np.array(positions[1]) - np.array(positions[0]), imaging_axis)
    return dot


def stack_images(images, orient, positions):
    # make sure images are ordered correctly (superior is z+)
    imaging_axis = np.cross(orient[:3], orient[3:])
    distance_projection = np.dot(np.stack(positions), imaging_axis)
    images = np.stack(images, -1)
    images = images[:, :, np.argsort(distance_projection)]
    return images


def load_dicom_3d(paths, resize_2d=None):
    images = []
    positions = []
    for filename in paths:
        img, orient, pos = load_dicom_2d(filename, resize=resize_2d)
        images.append(img)
        positions.append(pos)

    images = stack_images(images, orient, positions)
    images = window_images(images)
    return images


def load_dicom_folder(folder, num, resize_2d=None):
    t_paths = sorted(
        glob.glob(os.path.join(folder, "*")), key=lambda x: int(x.split("/")[-1].split(".")[0])
    )
    # print(t_paths, folder)
    n_scans = len(t_paths)
    if num is not None:
        indices = np.quantile(list(range(n_scans)), np.linspace(0.0, 1.0, num)).round().astype(int)
        t_paths = [t_paths[i] for i in indices]
    return load_dicom_3d(t_paths, resize_2d=resize_2d)


def resize3d(full3d, resize):
    indices = (
        np.quantile(list(range(full3d.shape[2])), np.linspace(0.0, 1.0, resize[2]))
        .round()
        .astype(int)
    )
    image3d = full3d[:, :, indices]
    images = []
    for i in range(image3d.shape[2]):
        image = cv2.resize(image3d[:, :, i], (resize[0], resize[1]), interpolation=cv2.INTER_LINEAR)
        images.append(image)
    image3d = np.stack(images, -1)
    return image3d


class TimmSegModel(nn.Module):
    def __init__(
        self,
        backbone="resnet18d",
        segtype="unet",
        out_dim=6,
        pretrained=False,
        drop_rate=0,
        drop_path_rate=0,
    ):
        super(TimmSegModel, self).__init__()

        self.encoder = timm.create_model(
            backbone,
            in_chans=1,
            features_only=True,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained,
        )
        g = self.encoder(torch.rand(1, 1, 64, 64))
        encoder_channels = [1] + [_.shape[1] for _ in g]
        decoder_channels = [256, 128, 64, 32, 16]
        n_blocks = 4
        if segtype == "unet":
            self.decoder = smp.decoders.unet.decoder.UnetDecoder(
                encoder_channels=encoder_channels[: n_blocks + 1],
                decoder_channels=decoder_channels[:n_blocks],
                n_blocks=n_blocks,
            )

        self.segmentation_head = nn.Conv2d(
            decoder_channels[n_blocks - 1],
            out_dim,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.n_blocks = n_blocks

    def forward(self, x):
        global_features = [0] + self.encoder(x)[: self.n_blocks]
        seg_features = self.decoder(*global_features)
        seg_features = self.segmentation_head(seg_features)
        return seg_features


def convert_3d(module):
    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d):
        module_output = torch.nn.BatchNorm3d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    elif isinstance(module, Conv2dSame):
        module_output = Conv3dSame(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
        )
        module_output.weight = torch.nn.Parameter(
            module.weight.unsqueeze(-1).repeat(1, 1, 1, 1, module.kernel_size[0])
        )

    elif isinstance(module, torch.nn.Conv2d):
        module_output = torch.nn.Conv3d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
        )
        module_output.weight = torch.nn.Parameter(
            module.weight.unsqueeze(-1).repeat(1, 1, 1, 1, module.kernel_size[0])
        )

    elif isinstance(module, torch.nn.MaxPool2d):
        module_output = torch.nn.MaxPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            ceil_mode=module.ceil_mode,
        )
    elif isinstance(module, torch.nn.AvgPool2d):
        module_output = torch.nn.AvgPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            ceil_mode=module.ceil_mode,
        )

    for name, child in module.named_children():
        module_output.add_module(name, convert_3d(child))
    del module

    return module_output


def get_seg_model(pretrained=True):
    model = TimmSegModel(pretrained=pretrained)
    model = convert_3d(model)
    return model


def load_organ(image_full, mask, cid, cropped_images):
    n_slice_per_c = [24, 24, 12, 12, 128]  # liver,spleen,left kidney, right kidney,bowel (all)
    n_ch = 3
    image_size_cls = 296
    mask_size = mask.shape[2]
    n_scans = image_full.shape[2]
    organ = []
    if cid < 4:  # liver,spleen,kindey left and right
        mask1 = mask == (cid + 1)
    else:
        mask1 = mask  # bowel -> extend to all

    n_slice = n_slice_per_c[cid]

    if np.sum(mask1) > 0:
        x = np.where(mask1.sum(1).sum(1) > 0)[0]
        y = np.where(mask1.sum(0).sum(1) > 0)[0]
        z = np.where(mask1.sum(0).sum(0) > 0)[0]
        x1, x2 = max(0, x[0] - 1), min(mask1.shape[0], x[-1] + 1)
        y1, y2 = max(0, y[0] - 1), min(mask1.shape[1], y[-1] + 1)
        z1, z2 = max(0, z[0] - 1), min(mask1.shape[2], z[-1] + 1)

        zz1, zz2 = int(z1 / mask_size * n_scans), int(z2 / mask_size * n_scans)  # in original
        inds = np.linspace(zz1, zz2 - 1, n_slice).astype(int)  # in original
        inds_ = np.linspace(z1, z2 - 1, n_slice).astype(int)  # in mask

        for ind, ind_ in zip(inds, inds_):
            mask_this = mask1[:, :, ind_]
            istart = ind - n_ch // 2
            iend = istart + n_ch
            istart1 = max(istart, 0)
            iend1 = min(iend, n_scans)
            select_indices = list(range(istart1, iend1))

            image = image_full[:, :, select_indices]
            if istart < 0:
                pad = np.zeros((image.shape[0], image.shape[1], -istart), dtype=np.uint8)
                image = np.concatenate((pad, image), axis=2)
            if iend > iend1:
                pad = np.zeros((image.shape[0], image.shape[1], iend - iend1), dtype=np.uint8)
                image = np.concatenate((image, pad), axis=2)

            mask_this = mask_this[x1:x2, y1:y2]
            xx1 = int(x1 / mask_size * image.shape[0])
            xx2 = int(x2 / mask_size * image.shape[0])
            yy1 = int(y1 / mask_size * image.shape[1])
            yy2 = int(y2 / mask_size * image.shape[1])
            image = image[xx1:xx2, yy1:yy2]
            image = cv2.resize(
                image, (image_size_cls, image_size_cls), interpolation=cv2.INTER_LINEAR
            )
            mask_this = (mask_this * 32).astype(np.uint8)
            mask_this = cv2.resize(
                mask_this, (image_size_cls, image_size_cls), interpolation=cv2.INTER_NEAREST
            )
            image = np.concatenate([image, mask_this[:, :, np.newaxis]], -1)
            organ.append(image)
        organ = np.stack(organ, 0)  # (n_slice, H,W,channel)

    else:
        organ = np.zeros((n_slice_per_c[cid], image_size_cls, image_size_cls, n_ch + 1))

    organ = np.transpose(organ, (0, 3, 1, 2))  # (n_slice, channels,H,W)

    cropped_images[cid] = organ


def get_crops(image_full, msk):
    threads = [None] * 5
    cropped_images = [None] * 5

    for cid in range(5):
        threads[cid] = threading.Thread(
            target=load_organ, args=(image_full, msk, cid, cropped_images)
        )
        threads[cid].start()
    for cid in range(5):
        threads[cid].join()
    cropped_images = np.concatenate(cropped_images, axis=0)
    return cropped_images


def visualize(image, mask=None, title=None, alpha=0.4):
    # rcParams['figure.figsize'] = 20,8

    color_dict = {0: "black", 1: "blue", 2: "green", 3: "wheat", 4: "pink", 5: "brown"}
    cm = ListedColormap(color_dict.values())

    image = image[:, ::-1, :]
    slice1 = image.shape[2] // 2
    slice2 = image.shape[1] // 2
    slice3 = image.shape[0] // 2

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(image[:, :, slice1].T, cmap="gray", origin="lower")
    ax[1].imshow(image[:, slice2, :].T, cmap="gray", origin="lower")
    ax[2].imshow(image[slice3, :, :].T, cmap="gray", origin="lower")
    if mask is not None:
        mask = mask[:, ::-1, :]
        ax[0].imshow(
            mask[:, :, slice1].T, alpha=alpha, origin="lower", cmap=cm, interpolation="nearest"
        )
        ax[1].imshow(
            mask[:, slice2, :].T, alpha=alpha, origin="lower", cmap=cm, interpolation="nearest"
        )
        ax[2].imshow(
            mask[slice3, :, :].T, alpha=alpha, origin="lower", cmap=cm, interpolation="nearest"
        )
    fig.suptitle(title)


class LSKModel(nn.Module):
    def __init__(
        self,
        backbone,
        pretrained=False,
        features=None,
        drop_rate=0,
        drop_path_rate=0,
        drop_rate_last=0,
        image_size=296,
        nslice=24,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.drop_rate_last = drop_rate_last
        self.image_size = image_size
        self.nslice = nslice
        self.encoder = timm.create_model(
            backbone,
            in_chans=4,
            num_classes=3,
            features_only=False,
            drop_rate=self.drop_rate,
            drop_path_rate=self.drop_path_rate,
            pretrained=pretrained,
        )
        self.features = features

        if "efficient" in backbone:
            hdim = self.encoder.conv_head.out_channels
            self.encoder.classifier = nn.Identity()
        elif "convnext" in backbone:
            hdim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()

        self.lstm = nn.LSTM(
            hdim, 64, num_layers=1, dropout=drop_rate, bidirectional=True, batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(drop_rate_last),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 3),
        )

    def forward(self, x):  # (bs, nslice, ch, sz, sz)
        bs = x.shape[0]
        x = x.view(bs * self.nslice, 4, self.image_size, self.image_size)
        feat1 = self.encoder(x)
        feat1 = feat1.view(bs, self.nslice, -1)
        feat2, _ = self.lstm(feat1)
        feat2 = feat2.contiguous().view(bs * self.nslice, -1)
        out = self.head(feat2)
        out = out.view(bs, self.nslice, 3).contiguous()
        if self.features == "feat1":
            return feat1
        elif self.features == "feat2":
            return feat2
        else:
            return out


class Attention(nn.Module):
    def __init__(self, feature_dim, **kwargs):
        super().__init__(**kwargs)

        self.supports_masking = True

        self.feature_dim = feature_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

    def forward(self, x, mask=None):
        step_dim = x.shape[1]
        feature_dim = self.feature_dim
        eij = torch.mm(x.contiguous().view(-1, feature_dim), self.weight).view(-1, step_dim)

        eij = torch.tanh(eij)
        a = torch.exp(eij)  # slice importances

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10  # normalize across slices

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class BowelModel(nn.Module):
    def __init__(
        self,
        backbone,
        pretrained=False,
        features=False,
        drop_rate=0,
        drop_path_rate=0,
        drop_rate_last=0,
        image_size=296,
    ):
        super().__init__()
        self.features = features
        self.encoder = timm.create_model(
            backbone,
            in_chans=4,
            num_classes=2,
            features_only=False,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained,
        )
        self.drop_rate_last = drop_rate_last

        if "efficient" in backbone:
            hdim = self.encoder.conv_head.out_channels
            self.encoder.classifier = nn.Identity()
        elif "convnext" in backbone:
            hdim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()

        hlstm = 64
        self.lstm = nn.LSTM(
            hdim, hlstm, num_layers=1, dropout=drop_rate, bidirectional=True, batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hlstm, hlstm),
            nn.BatchNorm1d(hlstm),
            nn.Dropout(drop_rate_last),
            nn.LeakyReLU(0.1),
            nn.Linear(hlstm, 2),
        )
        self.head2 = nn.Linear(4 * hlstm, 2)
        self.attention = Attention(2 * hlstm)
        self.image_size = image_size

    def forward(self, x):  # (bs, nslice, ch, sz, sz)
        bs = x.shape[0]
        nslices = x.shape[1]
        x = x.view(bs * nslices, 4, self.image_size, self.image_size)
        feat = self.encoder(x)
        feat = feat.view(bs, nslices, -1)
        feat, _ = self.lstm(feat)

        featv = feat.contiguous().view(bs * nslices, -1)
        out = self.head(featv)
        out = out.view(bs, nslices, 2).contiguous()

        att = self.attention(feat)
        max_ = feat.max(dim=1)[0]
        conc = torch.cat((att, max_), dim=1)
        out2 = self.head2(conc)

        if self.features:
            return feat
        else:
            return out, out2
