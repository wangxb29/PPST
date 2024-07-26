import lpips
import sys
import torch
import util
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from models import BaseModel
import models.networks as networks
import models.networks.loss as loss
from models.networks.rscl import rsclLoss
from torchvision import models,transforms
from PIL import Image

class PPSTModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        BaseModel.modify_commandline_options(parser, is_train)
        parser.add_argument("--spatial_code_ch", default=256, type=int)
        parser.add_argument("--global_code_ch", default=2048, type=int)
        parser.add_argument("--lambda_R1", default=10.0, type=float)
        parser.add_argument("--lambda_L1", default=3.0, type=float)
        parser.add_argument("--lambda_GAN", default=1.0, type=float)
        parser.add_argument("--training_stage", default=2, type=int)
        parser.add_argument("--lambda_StyleCon", default=1.0, type=float)
        parser.add_argument("--lambda_Maskwarp", default=10.0, type=float)
        parser.add_argument("--lambda_Cycwarp", default=5.0, type=float)
        parser.add_argument("--match_kernel", default=1, type=int)
        parser.add_argument("--lambda_triplet", default=0.0, type=float)
        parser.add_argument("--lambda_hist", default=0.0, type=float)
        parser.add_argument('--num_patches', type=int, default=128, help='number of patches per layer')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=True)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        return parser

    def initialize(self):
        self.E1 = networks.create_network(self.opt, self.opt.netE1, "encoder_con")
        self.E2 = networks.create_network(self.opt, self.opt.netE2, "encoder_col")
        self.G = networks.create_network(self.opt, self.opt.netG, "generator")
        if self.opt.lambda_GAN > 0.0:
            self.D = networks.create_network(
                self.opt, self.opt.netD, "discriminator")
        self.register_buffer(
            "num_discriminator_iters", torch.zeros(1, dtype=torch.long)
        )
        self.l1_loss = torch.nn.L1Loss()
        self.loss_fn_alex = lpips.LPIPS(net='alex')
        self.criterionNCE = rsclLoss(self.opt)
        if (not self.opt.isTrain) or self.opt.continue_train:
            self.load()
        if self.opt.num_gpus > 0:
            self.to("cuda:0")
        self.to_tensor = transforms.Compose([transforms.ToTensor()])

    def per_gpu_initialize(self):
        pass

    def swap(self, x):
        """ Swaps (or mixes) the ordering of the minibatch """
        shape = x.shape
        assert shape[0] % 2 == 0, "Minibatch size must be a multiple of 2"
        new_shape = [shape[0] // 2, 2] + list(shape[1:])
        x = x.view(*new_shape)
        x = torch.flip(x, [1])
        return x.view(*shape)

    def compute_image_discriminator_losses(self, real, rec, mix, cyc):
        if self.opt.lambda_GAN == 0.0:
            return {}

        pred_real = self.D(real)
        pred_rec = self.D(rec)
        losses = {}
        losses["D_real"] = loss.gan_loss(
            pred_real, should_be_classified_as_real=True
        ) * self.opt.lambda_GAN
        losses["D_rec"] = loss.gan_loss(
            pred_rec, should_be_classified_as_real=False
        ) * (0.5 * self.opt.lambda_GAN)
        if self.opt.training_stage ==2:
            if mix!=None:
                pred_mix = self.D(mix)
                losses["D_mix"] = loss.gan_loss(
                    pred_mix, should_be_classified_as_real=False
                ) * (0.5 * self.opt.lambda_GAN)
            if cyc != None:
                pred_cyc = self.D(cyc)
                losses["D_cyc"] = loss.gan_loss(
                    pred_cyc, should_be_classified_as_real=False
                ) * (0.5 * self.opt.lambda_GAN)
        return losses

    def get_random_crops(self, x, mask, crop_window=None):
        """ Make random crops.
            Corresponds to the yellow and blue random crops of Figure 2.
        """
        crops, labels = util.apply_random_crop(
            x, mask, self.opt.patch_size,
            (self.opt.patch_min_scale, self.opt.patch_max_scale),
            num_crops=self.opt.patch_num_crops
        )
        return crops, labels
    
    def compute_discriminator_losses(self, real, mask):
        self.num_discriminator_iters.add_(1)
        sp = self.E1(real)
        gl, _ = self.E2(real)
        if self.opt.training_stage ==1:
            rec = self.G(sp, gl)
            mix = None
            cyc = None
        else:
            mix = None
            _, feas, feas1 = self.G(sp, gl, extract_features=True)
            selfatt = self.Rselfcorr(feas1)
            sps = torch.cat((feas,selfatt),dim=1)
            corrms = self.corrm(sps,self.swap(sps))
            corr_self = self.corrm(sps,sps)            
            if self.opt.lambda_StyleCon > 0.0:
                _, gl_w = self.E2(real,corrmatrix=corrms)     
                mix = self.G(self.swap(sp), gl_w)                
            _, gl = self.E2(real,corrmatrix=corr_self)  
            cyc = None      
        B = real.size(0)
        assert B % 2 == 0, "Batch size must be even on each GPU."
        # To save memory, compute the GAN loss on only
        # half of the reconstructed images
        gl_d = []
        for sgl in gl:
            gl_d.append(sgl[:B // 2])
        rec = self.G(sp[:B // 2], gl_d)
        losses = self.compute_image_discriminator_losses(real, rec, mix, cyc)

        metrics = {}  # no metrics to report for the Discriminator iteration
        for x in range(len(gl)):
            gl[x] = gl[x].detach()
        return losses, metrics, sp.detach(), gl
    
    def compute_R1_loss(self, real):
        losses = {}
        if self.opt.lambda_R1 > 0.0:
            real.requires_grad_()
            pred_real = self.D(real).sum()
            grad_real, = torch.autograd.grad(
                outputs=pred_real,
                inputs=[real],
                create_graph=True,
                retain_graph=True,
            )
            grad_real2 = grad_real.pow(2)
            dims = list(range(1, grad_real2.ndim))
            grad_penalty = grad_real2.sum(dims) * (self.opt.lambda_R1 * 0.5)
        else:
            grad_penalty = 0.0

        grad_crop_penalty = 0.0
        losses["D_R1"] = grad_penalty + grad_crop_penalty
        return losses

    def compute_generator_losses(self, real, sp_ma, gl_ma, mask):
        losses, metrics = {}, {}
        B, c, h, w = real.size()       
        sp = self.E1(real)  
        gl, _ = self.E2(real)
        if self.opt.training_stage ==2:
            _, feas, feas1 = self.G(sp, gl, extract_features=True)
            selfatt = self.Rselfcorr(feas1)
            sps = torch.cat((feas,selfatt),dim=1)
            corrm = self.corrm(sps,self.swap(sps))
            corrm_self = self.corrm(sps,sps)
            _, gl = self.E2(real,corrmatrix=corrm_self)
            if self.opt.lambda_StyleCon > 0.0:
                _, pro_ms, gl_w, pro_mw = self.E2(real,mask=mask,corrmatrix=corrm) 
            if self.opt.lambda_Cycwarp > 0.0:
                image_warp = self.warp(real,corr=corrm)    
                image_rec = self.warp(image_warp,corr=self.swap(corrm))
                regloss = self.loss_fn_alex(image_rec, real)
                losses["image_warp_reg"] = regloss * self.opt.lambda_Cycwarp 
            if self.opt.lambda_Maskwarp > 0.0:
                mask_warp = self.warp(mask,corrm)
                losses["Mask_warp"] = self.l1_loss(mask_warp, self.swap(mask)) * self.opt.lambda_Maskwarp

        rec = self.G(sp, gl)
        if self.opt.lambda_L1 > 0.0:
            losses["G_L1"] = self.l1_loss(rec, real) * self.opt.lambda_L1 
        if self.opt.crop_size >= 1024:
            # another momery-saving trick: reduce #outputs to save memory
            real = real[B // 2:]
            gl = gl[B // 2:]
            sp_mix = sp_mix[B // 2:]

        if self.opt.lambda_StyleCon > 0.0:
            mix = self.G(self.swap(sp), gl_w) 
            _, pro_3m, _, _ = self.E2(mix,mask=self.swap(mask))
            _, pro_2m, _, _ = self.E2(rec,mask=mask)
            sp_3 = self.E1(mix)#+0.5*self.swap(rec)) 
            gl_d = []
            for sgl in gl:
                gl_d.append(sgl[:B // 2])
            cyc = self.G(self.swap(sp_3)[:B // 2], gl_d)    
            metrics["L1_dist"] = self.l1_loss(cyc, real[:B // 2])
            losses["G_L1_cyc"] = metrics["L1_dist"] * 3
            styleloss = 0.0
            styleloss2 = 0.0
            for layer_id, (key0, keyw, query, query_r) in enumerate(zip(pro_ms, pro_mw, pro_3m, pro_2m)):
                if layer_id % 3==0:
                    key0 = torch.cat(pro_ms[layer_id:layer_id+3],dim=0)
                    keyw = torch.cat(pro_mw[layer_id:layer_id+3],dim=0)
                    query = torch.cat(pro_3m[layer_id:layer_id+3],dim=0)
                    query_r = torch.cat(pro_2m[layer_id:layer_id+3],dim=0)
                    styleloss += self.criterionNCE(query, keyw.detach(), key0.detach(), int(layer_id / 3))
                    styleloss2 += self.criterionNCE(query_r, key0.detach(), keyw.detach(), int(layer_id / 3))   
                    self.criterionNCE.dequeue_and_enqueue(key0[0:1].detach(), int(layer_id / 3))   
                    self.criterionNCE.dequeue_and_enqueue(key0[1:2].detach(), int(layer_id / 3))  
                    self.criterionNCE.dequeue_and_enqueue(key0[2:3].detach(), int(layer_id / 3))     
                    self.criterionNCE.dequeue_and_enqueue(keyw[0:1].detach(), int(layer_id / 3))   
                    self.criterionNCE.dequeue_and_enqueue(keyw[1:2].detach(), int(layer_id / 3))  
                    self.criterionNCE.dequeue_and_enqueue(keyw[2:3].detach(), int(layer_id / 3))     
            losses["G_styleContmix"] = styleloss * self.opt.lambda_StyleCon
            losses["G_styleContrec"] = styleloss2 * self.opt.lambda_StyleCon                             

        if self.opt.lambda_GAN > 0.0:
            losses["G_GAN_rec"] = loss.gan_loss(
                self.D(rec),
                should_be_classified_as_real=True
            ) * (self.opt.lambda_GAN * 0.5)

            if self.opt.lambda_StyleCon > 0.0:
                losses["G_GAN_mix"] = loss.gan_loss(
                    self.D(mix),
                    should_be_classified_as_real=True
                ) * (self.opt.lambda_GAN * 1.0)

        return losses, metrics

    def get_visuals_for_snapshot(self, real):
        if self.opt.isTrain:
            # avoid the overhead of generating too many visuals during training
            real = real[:2] if self.opt.num_gpus > 1 else real[:4]
        sp, gl = self.E(real)
        layout = util.resize2d_tensor(util.visualize_spatial_code(sp), real)
        rec = self.G(sp, gl)
        mix = self.G(sp, self.swap(gl))

        visuals = {"real": real, "layout": layout, "rec": rec, "mix": mix}

        return visuals

    def fix_noise(self, sample_image=None):
        """ The generator architecture is stochastic because of the noise
        input at each layer (StyleGAN2 architecture). It could lead to
        flickering of the outputs even when identical inputs are given.
        Prevent flickering by fixing the noise injection of the generator.
        """
        if sample_image is not None:
            # The generator should be run at least once,
            # so that the noise dimensions could be computed
            sp, gl = self.E(sample_image)
            self.G(sp, gl)
        noise_var = self.G.fix_and_gather_noise_parameters()
        return noise_var

    def encode(self, image, extract_features=False, testtime=False):
        return self.E1(image), self.E2(image, extract_features=extract_features)[0]
    
    def encode2(self, image, corrmatrix):
        return self.E2(image, corrmatrix=corrmatrix)  
    
    def smooth(self, out, target):
        if target!=None:
            from photo_gif import GIFSmoothing
            p_pro = GIFSmoothing(r=30, eps=(0.02 * 255) ** 2)
            new_out = torch.zeros_like(out,device=out.device)                
            out = util.tensor2im(out,tile=False)              
            target = util.tensor2im(target,tile=False)
            for i,(ori, wai) in enumerate(zip(target, out)):
                wai = Image.fromarray(wai)
                ori = Image.fromarray(ori)
                wai = p_pro.process(wai, ori)                    
                wai = self.to_tensor(wai)
                ori = self.to_tensor(ori)
                wai = (wai-0.5)*2
                ori = (ori-0.5)*2
                new_out[i] = wai
            return new_out
        
    def decode(self, spatial_code, global_code, target=None):
        out = self.G(spatial_code, global_code)
        if target!=None:
            from photo_gif import GIFSmoothing
            p_pro = GIFSmoothing(r=30, eps=(0.02 * 255) ** 2)
            new_out = torch.zeros_like(out,device=out.device)                
            out = util.tensor2im(out,tile=False)              
            target = util.tensor2im(target,tile=False)
            for i,(ori, wai) in enumerate(zip(target, out)):
                wai = Image.fromarray(wai)
                ori = Image.fromarray(ori)
                wai = p_pro.process(wai, ori)      
                wai = self.to_tensor(wai)
                ori = self.to_tensor(ori)
                wai = (wai-0.5)*2
                ori = (ori-0.5)*2
                new_out[i] = wai
            return new_out
        return out
    
    def extract_feat(self, spatial_code, global_code):
        return self.G(spatial_code, global_code, extract_features=True)
    
    def extract_feat_from_image(self, img):
        sp = self.E1(img)
        gl = self.E2(img)[0]
        _, fea, fea1 = self.G(sp, gl, extract_features=True)
        return fea, fea1
    
    def get_parameters_for_mode(self, mode):
        if mode == "generator":
            return list(self.G.parameters()) 
        elif mode == "contentencoder":
            return list(self.E1.parameters())      
        elif mode == "colorencoder":
            return list(self.E2.parameters())
        elif mode == "discriminator":
            Dparams = []
            if self.opt.lambda_GAN > 0.0:
                Dparams = Dparams + list(self.D.parameters())
            return Dparams
        
    def Rselfcorr(self, fea):
        fea1 = F.unfold(fea, kernel_size=4, stride=4).permute(0,2,1).reshape(fea.size(0),-1,fea.size(1),16).permute(0,2,1,3)
        fea1 = fea1 - fea1.mean(dim=1,keepdim=True)
        fea1_norm = (torch.norm(fea1, 2, 1, keepdim=True)+sys.float_info.epsilon)     
        fea1 = fea1 / fea1_norm
        fea1 = fea1.unsqueeze(4)
        fea0 = fea1.permute(0,1,2,4,3)
        corr = torch.sum(torch.matmul(fea1,fea0).reshape(fea1.size(0),fea1.size(1),fea1.size(2),256),dim=1)
        corr = corr.permute(0,2,1).reshape(fea1.size(0),256,64,64)
        return corr       

    def corrm(self, fea, fea0):
        if self.opt.match_kernel == 1:
            fea0 = fea0.view(fea0.size(0), fea0.size(1), -1).contiguous()  # 2*256*(feature_height*feature_width)
            fea = fea.view(fea.size(0), fea.size(1), -1).contiguous()  # 2*256*(feature_height*feature_width)            
        else:
            fea0 = F.unfold(fea0, kernel_size=self.opt.match_kernel, padding=int(self.opt.match_kernel // 2))
            fea = F.unfold(fea, kernel_size=self.opt.match_kernel, padding=int(self.opt.match_kernel // 2))

        feah1, feah2 = fea[:,0:256], fea[:,256:]
        feah1 = feah1 - feah1.mean(dim=1,keepdim=True)
        fea = torch.cat((feah1,feah2),dim=1)

        fea0h1, fea0h2 = fea0[:,0:256], fea0[:,256:]
        fea0h1 = fea0h1 - fea0h1.mean(dim=1,keepdim=True)
        fea0 = torch.cat((fea0h1,fea0h2),dim=1).permute(0,2,1).contiguous()            

        fea_norm = (torch.norm(fea, 2, 1, keepdim=True)+sys.float_info.epsilon)
        fea = fea / fea_norm
        
        fea0_norm = (torch.norm(fea0, 2, 2, keepdim=True)+sys.float_info.epsilon)
        fea0 = fea0 / fea0_norm

        corr = F.softmax(torch.matmul(fea0, fea) / 0.01, dim=-1)
        return corr
    
    def warp(self, fea, corr):
        b,c,h,w = fea.size()
        l = h * w
        B,H,W = corr.size()
        if H != l:
            s = int((l/H) ** 0.5)
            feas = F.unfold(fea, s, stride=s)
            # feas = F.avg_pool2d(fea,(s,s),s)
            # feas = feas.view(b,c,-1)
            feas = feas.permute(0,2,1).contiguous()

            warp_fea = torch.matmul(corr, feas)
            warp_fea = warp_fea.permute(0,2,1).contiguous()
            # warp_fea = warp_fea.view(b,c,int(h/s),int(w/s))
            # warp_fea = F.interpolate(warp_fea,scale_factor=s,mode='bilinear')
            warp_fea = F.fold(warp_fea, (h,w) ,s, stride=s)
            #
            return warp_fea
        fea = fea.view(b,c,-1).permute(0,2,1).contiguous()
        warp_feat_f = torch.matmul(corr, fea)
        warp_feat = warp_feat_f.permute(0, 2, 1).view(b,c,h,w).contiguous()
        return warp_feat 
     