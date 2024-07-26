import numpy as np
import torch
import util
from models import MultiGPUModelWrapper
from optimizers.base_optimizer import BaseOptimizer

class PPSTOptimizer(BaseOptimizer):
    """ Class for running the optimization of the model parameters.
    Implements Generator / Discriminator training, R1 gradient penalty,
    decaying learning rates, and reporting training progress.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--lr", default=0.001, type=float)
        parser.add_argument("--beta1", default=0.0, type=float)
        parser.add_argument("--beta2", default=0.99, type=float)
        parser.add_argument(
            "--R1_once_every", default=16, type=int,
            help="lazy R1 regularization. R1 loss is computed "
                 "once in 1/R1_freq times",
        )
        return parser

    def __init__(self, model: MultiGPUModelWrapper):
        self.opt = model.opt
        opt = self.opt
        self.model = model
        self.train_mode_counter = 0
        self.discriminator_iter_counter = 0

        self.Gparams = self.model.get_parameters_for_mode("generator")
        self.E1params = self.model.get_parameters_for_mode("contentencoder")
        self.E2params = self.model.get_parameters_for_mode("colorencoder")
        self.Dparams = self.model.get_parameters_for_mode("discriminator")

        self.optimizer_G = torch.optim.Adam(
            self.Gparams, lr=opt.lr, betas=(opt.beta1, opt.beta2)
        )
        self.optimizer_E1 = torch.optim.Adam(
            self.E1params, lr=opt.lr, betas=(opt.beta1, opt.beta2)
        )
        self.optimizer_E2 = torch.optim.Adam(
            self.E2params, lr=opt.lr, betas=(opt.beta1, opt.beta2)
        )

        c = opt.R1_once_every / (1 + opt.R1_once_every)
        self.optimizer_D = torch.optim.Adam(
            self.Dparams, lr=opt.lr * c, betas=(opt.beta1 ** c, opt.beta2 ** c)
        )
    def set_requires_grad(self, params, requires_grad):
        """ For more efficient optimization, turn on and off
            recording of gradients for |params|.
        """
        for p in params:
            p.requires_grad_(requires_grad)

    def prepare_images(self, data_i):
        return data_i["real_A"], data_i["mask_A"]

    def toggle_training_mode(self):
        modes = ["discriminator", "generator"]#,
        self.train_mode_counter = (self.train_mode_counter + 1) % len(modes)
        return modes[self.train_mode_counter]

    def train_one_step(self, data_i, total_steps_so_far):
        images_minibatch, mask_minibatch = self.prepare_images(data_i)
        if self.toggle_training_mode() == "generator":
            losses = self.train_discriminator_one_step(images_minibatch, mask_minibatch)
        else:
            losses = self.train_generator_one_step(images_minibatch, mask_minibatch)
        return util.to_numpy(losses)

    def train_generator_one_step(self, images, mask):
        self.set_requires_grad(self.Dparams, False)
        self.set_requires_grad(self.Gparams, True)
        self.set_requires_grad(self.E1params, True)
        self.set_requires_grad(self.E2params, True)

        sp_ma, gl_ma = None, None
        self.optimizer_G.zero_grad()
        self.optimizer_E1.zero_grad()
        self.optimizer_E2.zero_grad()
  
        g_losses, g_metrics = self.model(
            images, sp_ma, gl_ma, mask, command="compute_generator_losses"
        )        
        g_loss = sum([v.mean() for v in g_losses.values()])
        g_loss.backward()
        self.optimizer_G.step()
        self.optimizer_E2.step()
        self.optimizer_E1.step() 

        g_losses.update(g_metrics)
        return {**g_losses}

    def train_discriminator_one_step(self, images, mask):
        if self.opt.lambda_GAN == 0.0:
            return {}
        self.set_requires_grad(self.Dparams, True)
        self.set_requires_grad(self.Gparams, False)
        self.set_requires_grad(self.E1params, False)
        self.set_requires_grad(self.E2params, False)
        self.discriminator_iter_counter += 1
        self.optimizer_D.zero_grad()
        d_losses, d_metrics, sp, gl = self.model(
            images, mask, command="compute_discriminator_losses"
        )
        self.previous_sp = sp.detach()
        for x in range(len(gl)):
            gl[x] = gl[x].detach()
        self.previous_gl = gl
        d_loss = sum([v.mean() for v in d_losses.values()])
        d_loss.backward()
        self.optimizer_D.step()

        needs_R1 = self.opt.lambda_R1 > 0.0 or self.opt.lambda_patch_R1 > 0.0
        needs_R1_at_current_iter = needs_R1 and \
            self.discriminator_iter_counter % self.opt.R1_once_every == 0
        if needs_R1_at_current_iter:
            self.optimizer_D.zero_grad()
            r1_losses = self.model(images, command="compute_R1_loss")
            d_losses.update(r1_losses)
            r1_loss = sum([v.mean() for v in r1_losses.values()])
            r1_loss = r1_loss * self.opt.R1_once_every
            r1_loss.backward()
            self.optimizer_D.step()

        d_losses["D_total"] = sum([v.mean() for v in d_losses.values()])
        d_losses.update(d_metrics)
        return d_losses

    def get_visuals_for_snapshot(self, data_i):
        images, _ = self.prepare_images(data_i)
        with torch.no_grad():
            return self.model(images, command="get_visuals_for_snapshot")

    def save(self, total_steps_so_far):
        self.model.save(total_steps_so_far)
