import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import custom_transforms
import models
from utils import tensor2array, save_checkpoint, save_path_formatter, log_output_tensorboard

from models.pose_exp_net import PoseExpNet
from models.disp_net import DispNetS
from loss_functions import photometric_reconstruction_loss, explainability_loss, smooth_loss
from loss_functions import compute_depth_errors, compute_pose_errors
from inverse_warp import pose_vec2mat

class SfmLearner:
    def __init__(
        self,
        intrinsics,
        mask_loss_weight=1,
        photo_loss_weight=1,
        smooth_loss_weight=0.1,
        sequence_length=2,
        lr=2e-4,
        momentum=0.9,
        beta=0.999,
        weight_decay=0,
        rotation_mode='euler',  # either 'euler' or 'quat', TODO: make enum
        padding_mode='zeros'  # either 'zeros' or 'border', TODO: make enum
    ):
        self.intrinsics = intrinsics
        self.intrinsics_inv = torch.linalg.inv(intrinsics)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.disp_net = DispNetS().to(self.device)
        self.output_exp = mask_loss_weight > 0
        self.mask_loss_weight = mask_loss_weight
        self.photo_loss_weight = photo_loss_weight
        self.smooth_loss_weight = smooth_loss_weight
        self.pose_exp_net = PoseExpNet(
            nb_ref_imgs=sequence_length - 1,
            output_exp=mask_loss_weight > 0
        ).to(self.device)

        self.disp_net = torch.nn.DataParallel(self.disp_net)
        self.pose_exp_net = torch.nn.DataParallel(self.pose_exp_net)

        self.optim_params = [
            {'params': self.disp_net.parameters(), 'lr': lr},
            {'params': self.pose_exp_net.parameters(), 'lr': lr}
        ]
        self.optimizer = torch.optim.Adam(
            self.optim_params,
            betas=(momentum, beta),
            weight_decay=weight_decay
        )
        self.rotation_mode = rotation_mode
        self.padding_mode = padding_mode

    def train(self, batch):
        intrinsics = self.intrinsics
        intrinsics_inv = self.intrinsics_inv
        for (i, imgs) in enumerate(batch):
            tgt_img_idx = len(imgs) // 2
            tgt_img = imgs[tgt_img_idx]
            ref_imgs = imgs[:tgt_img_idx] + imgs[tgt_img_idx + 1:]
            # log_losses = i > 0 and n_iter % args.print_freq == 0
            # log_output = args.training_output_freq > 0 and n_iter % args.training_output_freq == 0

            # measure data loading time
            # data_time.update(time.time() - end)
            tgt_img = tgt_img.to(self.device)
            ref_imgs = [img.to(self.device) for img in ref_imgs]
            intrinsics = intrinsics.to(self.device)

            # compute output
            disparities = self.disp_net(tgt_img)
            depth = [1 / disp for disp in disparities]
            explainability_mask, pose = self.pose_exp_net(tgt_img, ref_imgs)

            loss_1, warped, diff = photometric_reconstruction_loss(
                tgt_img,
                ref_imgs,
                intrinsics,
                depth,
                explainability_mask,
                pose,
                self.rotation_mode,
                self.padding_mode
            )
            if self.mask_loss_weight > 0:
                loss_2 = explainability_loss(explainability_mask)
            else:
                loss_2 = 0
            loss_3 = smooth_loss(depth)

            w1 = self.photo_loss_weight
            w2 = self.mask_loss_weight
            w3 = self.smooth_loss_weight

            loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3

            # if log_losses:
            #     tb_writer.add_scalar('photometric_error', loss_1.item(), n_iter)
            #     if w2 > 0:
            #         tb_writer.add_scalar('explanability_loss', loss_2.item(), n_iter)
            #     tb_writer.add_scalar('disparity_smoothness_loss', loss_3.item(), n_iter)
            #     tb_writer.add_scalar('total_loss', loss.item(), n_iter)

            # if log_output:
            #     tb_writer.add_image('train Input', tensor2array(tgt_img[0]), n_iter)
            #     for k, scaled_maps in enumerate(zip(depth, disparities, warped, diff, explainability_mask)):
            #         log_output_tensorboard(tb_writer, "train", 0, " {}".format(k), n_iter, *scaled_maps)

            # record loss and EPE
            # losses.update(loss.item(), args.batch_size)

            # compute gradient and do Adam step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            # batch_time.update(time.time() - end)
            # end = time.time()

            # with open(args.save_path/args.log_full, 'a') as csvfile:
            #     writer = csv.writer(csvfile, delimiter='\t')
            #     writer.writerow([loss.item(), loss_1.item(), loss_2.item() if w2 > 0 else 0, loss_3.item()])
            # logger.train_bar.update(i+1)
            # if i % args.print_freq == 0:
            #     logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
            # if i >= epoch_size - 1:
            #     break
            #
            # n_iter += 1

        return loss
        # return losses.avg[0]
