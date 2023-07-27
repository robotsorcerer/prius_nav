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
import sfm.custom_transforms
import sfm.models
from sfm.utils import tensor2array, save_checkpoint, save_path_formatter, log_output_tensorboard

from train_util import Artifacts
from sfm.models.pose_exp_net import PoseExpNet
from sfm.models.disp_net import DispNetS
from sfm.loss_functions import photometric_reconstruction_loss, explainability_loss, smooth_loss
from sfm.loss_functions import compute_depth_errors, compute_pose_errors
from sfm.inverse_warp import pose_vec2mat

class SfmLearner:
    def __init__(
        self,
        intrinsics: np.ndarray,
        mask_loss_weight=1,
        photo_loss_weight=1,
        smooth_loss_weight=0.1,
        sequence_length=3,
        lr=2e-4,
        momentum=0.9,
        beta=0.999,
        weight_decay=0,
        rotation_mode='euler',  # either 'euler' or 'quat', TODO: make enum
        padding_mode='zeros'  # either 'zeros' or 'border', TODO: make enum
    ):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.intrinsics = torch.tensor(intrinsics, requires_grad=False).float().to(self.device)
        self.intrinsics_inv = torch.linalg.inv(self.intrinsics).to(self.device)
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

    def train_step(self, imgs, return_outputs=False):
        """
        One gradient update for a batch of images.

        :params:
            imgs: List[Tensor]
                List of image batches, where the list represents a continguous sequence
                of images in time (i.e., a video)
        :returns:
            loss: float
                Mean loss
        """
        batch_size = imgs[0].shape[0]

        # Repeat intrinsics matrix across batch
        intrinsics = self.intrinsics.unsqueeze(0).expand(batch_size, 3, 3)
        intrinsics_inv = self.intrinsics_inv.unsqueeze(0).expand(batch_size, 3, 3)

        tgt_img_idx = len(imgs) // 2
        tgt_img = imgs[tgt_img_idx]
        ref_imgs = imgs[:tgt_img_idx] + imgs[tgt_img_idx + 1:]
        # log_losses = i > 0 and n_iter % args.print_freq == 0
        # log_output = args.training_output_freq > 0 and n_iter % args.training_output_freq == 0

        # measure data loading time
        # data_time.update(time.time() - end)
        tgt_img = tgt_img.to(self.device)
        ref_imgs = [img.to(self.device) for img in ref_imgs]

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

        graphics = dict()
        if return_outputs:
            graphics = {
                'depth': depth
            }

        other = dict()
        if return_outputs:
            other = {
                'pose': pose,
                'exp': explainability_mask,
            }

        return Artifacts(
            metrics={
                'loss/total': loss.item(),
                'loss/photo': loss_1.item(),
                'loss/mask': loss_2.item(),
                'loss/smooth': loss_2.item(),
            },
            graphics=graphics,
            other=other
        )

        # return losses.avg[0]
