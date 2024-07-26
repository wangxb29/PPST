import os
import torchvision.transforms as transforms
from PIL import Image
from evaluation import BaseEvaluator
from data.base_dataset import get_transform
import util
import torch


class SimpleSwappingEvaluator(BaseEvaluator):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--input_structure_image", required=True, type=str)
        parser.add_argument("--input_texture_image", required=True, type=str)
        parser.add_argument("--texture_mix_alphas", type=float, nargs='+',
                            default=[1.0],
                            help="Performs interpolation of the texture image."
                            "If set to 1.0, it performs full swapping."
                            "If set to 0.0, it performs direct reconstruction"
                            )
        
        opt, _ = parser.parse_known_args()
        dataroot = os.path.dirname(opt.input_structure_image)
        
        # dataroot and dataset_mode are ignored in SimpleSwapplingEvaluator.
        # Just set it to the directory that contains the input structure image.
        parser.set_defaults(dataroot=dataroot, dataset_mode="imagefolder")
        
        return parser
    
    def load_image(self, path):
        path = os.path.expanduser(path)
        img = Image.open(path).convert('RGB')
        transform = get_transform(self.opt)
        tensor = transform(img).unsqueeze(0)
        return tensor
    
    def evaluate(self, model, dataset, nsteps=None):
        structure_image = self.load_image(self.opt.input_structure_image)
        texture_image = self.load_image(self.opt.input_texture_image)
        os.makedirs(self.output_dir(), exist_ok=True)
        
        #model(sample_image=structure_image, command="fix_noise")
        structure_code, source_texture_code = model(
            structure_image, command="encode")
        #_, target_texture_code = model(texture_image, command="encode")
        fea_c, fea_c1 = model(structure_image, command="extract_feat_from_image")
        fea_s, fea_s1 = model(texture_image, command="extract_feat_from_image")
        fea_c = torch.cat((fea_c,model(fea_c1,command="Rselfcorr")),dim=1)  
        fea_s = torch.cat((fea_s,model(fea_s1,command="Rselfcorr")),dim=1)  

        corrmatrix = model(fea_s, fea_c, command="corrm")
        _, gl_w = model(texture_image.cuda(), [corrmatrix], command="encode2")
        alphas = self.opt.texture_mix_alphas
        for alpha in alphas:
            texture_code = util.lerp(
                source_texture_code, gl_w, alpha)

            output_image = model(structure_code, texture_code, target=None, command="decode")
            #output_image = model(texture_image, corrmatrix, command="warp")
            output_image = transforms.ToPILImage()(
                (output_image[0].clamp(-1.0, 1.0) + 1.0) * 0.5)

            output_name = "%s_%s_%.2f.png" % (
                os.path.splitext(os.path.basename(self.opt.input_structure_image))[0],
                os.path.splitext(os.path.basename(self.opt.input_texture_image))[0],
                alpha
            )
            corrmatrix_name = "corrmatrix.png"
            output_path = os.path.join(self.output_dir(), output_name)
            outcorr_path = os.path.join(self.output_dir(), corrmatrix_name)
            output_image.save(output_path)
            output_image.save(outcorr_path)
            print("Saved at " + output_path)

        return {}
