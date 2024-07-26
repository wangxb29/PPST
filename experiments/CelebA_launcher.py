from .tmux_launcher import Options, TmuxLauncher


class Launcher(TmuxLauncher):
    def options(self):
        opt = Options()
        opt.set(
            # I exported FFHQ dataset to 70,000 image files
            # and load them as images files.
            # Alternatively, the dataset can be prepared as
            # an LMDB dataset (like LSUN), and set dataset_mode = "lmdb".
            dataroot="~/datasets/CelebAMaskHQ/img/",
            dataroot2="~/datasets/CelebAMaskHQ/label/",
            dataset_mode="CelebAMask",
            checkpoints_dir="./checkpoints/",
            num_gpus=1, batch_size=2,
            preprocess="resize",
            load_size=512, crop_size=512,
        )

        return [
            opt.specify(
                name="CelebAMaskHQ_selfatt",
                model="ppst",
                optimizer="ppst",
            ),
        ]

    def train_options(self):
        common_options = self.options()
        return [opt.specify(
            continue_train=True,
            evaluation_metrics="swap_visualization",
            evaluation_freq=50000) for opt in common_options]
        
    def test_options(self):
        opts = self.options()[0]
        return [
            opts.tag("swapping_grid").specify(
                num_gpus=1,
                batch_size=1,
                dataroot="your test data",
                dataname = "test1",
                dataset_mode="CelebAMask",
                preprocess="scale_width",
                evaluation_metrics="content_style_1t1_generation"
            ),
        ]