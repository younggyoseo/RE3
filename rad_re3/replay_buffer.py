import numpy as np

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import hydra
import utils
import random


class ReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        state_shape,
        capacity,
        image_size,
        random_encoder,
        aug_type,
        use_drq,
        device,
    ):
        self.capacity = capacity
        self.aug_type = aug_type
        self.image_size = image_size
        self.use_drq = use_drq
        self.device = device

        self.candidates = dict()

        self.crop = kornia.augmentation.RandomCrop((image_size, image_size))
        self.center_crop = kornia.augmentation.CenterCrop((image_size, image_size))
        self.random_encoder = random_encoder
        self.s_ent_stats = utils.TorchRunningMeanStd(shape=[1], device=device)

        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        self.random_feats = np.empty(
            (capacity, random_encoder.feature_dim), dtype=np.float32
        )

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def random_translate(self, imgs, return_hw=False, h1=None, w1=None):
        n, c, h, w = imgs.shape
        pad = (self.image_size - h) // 2
        imgs = F.pad(imgs, [pad * 2, pad * 2, pad * 2, pad * 2])

        out = h + 2 * pad
        crop_max = 2 * pad + 1
        if w1 is None:
            w1 = np.random.randint(0, crop_max, n)
        if h1 is None:
            h1 = np.random.randint(0, crop_max, n)
        cropped = []
        for img, h11, w11 in zip(imgs, h1, w1):
            cropped.append(img[:, h11 : h11 + out, w11 : w11 + out])
        cropped = torch.stack(cropped)
        if return_hw:
            return cropped, (h1, w1)
        else:
            return cropped

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        with torch.no_grad():
            channel = self.random_encoder.channel
            obs = torch.FloatTensor(obs[-channel:]).to(self.device)
            obs = obs.unsqueeze(0)
            random_feat = self.random_encoder(obs)
            random_feat = random_feat.squeeze(0).cpu().numpy()
        np.copyto(self.random_feats[self.idx], random_feat)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size, magnitude=None):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        raw_obses = self.obses[idxs]
        raw_next_obses = self.next_obses[idxs]

        raw_obses = torch.as_tensor(raw_obses, device=self.device).float()
        raw_next_obses = torch.as_tensor(raw_next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(
            self.not_dones_no_max[idxs], device=self.device
        )

        if self.full:
            random_feats = self.random_feats
        else:
            random_feats = self.random_feats[: self.idx]
        tgt_random_feats = torch.as_tensor(random_feats, device=self.device)
        src_random_feats = torch.as_tensor(self.random_feats[idxs], device=self.device)

        if self.aug_type == "crop":
            obses = self.crop(raw_obses)
            next_obses = self.crop(raw_next_obses)
            if self.use_drq:
                obses_aug = self.crop(raw_obses)
                next_obses_aug = self.crop(raw_next_obses)
            else:
                obses_aug, next_obses_aug = None, None
        elif self.aug_type == "translate":
            obses, (h1, w1) = self.random_translate(raw_obses, return_hw=True)
            next_obses = self.random_translate(raw_next_obses, h1=h1, w1=w1)
            if self.use_drq:
                obses_aug, (h1, w1) = self.random_translate(raw_obses, return_hw=True)
                next_obses_aug = self.random_translate(raw_next_obses, h1=h1, w1=w1)
            else:
                obses_aug, next_obses_aug = None, None
        else:
            raise ValueError(self.aug_type)

        return (
            obses,
            obses_aug,
            actions,
            rewards,
            next_obses,
            next_obses_aug,
            not_dones_no_max,
            src_random_feats,
            tgt_random_feats,
        )
