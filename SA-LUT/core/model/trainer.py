# to resume training:
# python scripts/main.py fit --config configs/xxx.yaml --ckpt_path logs/xxx.ckpt

import os
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
import lightning as L  # type: ignore
from torchvision import transforms  # type: ignore
from torchvision.utils import make_grid, save_image  # type: ignore
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR  # type: ignore
import lpips  # type: ignore

from core.module.discriminator import TAGANDiscriminator
from core.module.loss import PerceptualLoss, HistLoss, LABLoss
from core.module.style2vlognet import Style2VLogImage2ImageNet
from core.module.model import VLog2StyleNet4D


class VLog2StyleTrainer4D(L.LightningModule):
    def __init__(
        self, train_args=None, optim_args=None, model_args=None, name="Testing"
    ):
        super().__init__()
        self.automatic_optimization = (
            False  # because we use GAN, need to manually optimize
        )

        # Set default values
        train_args = train_args or {}
        optim_args = optim_args or {}
        model_args = model_args or {}

        # Get backbone type from model_args
        discriminator_model_name = model_args.get(
            "discriminator_model_name", "TAGANDiscriminator"
        )
        self.temperature = model_args.get("temperature", 34)
        self.dim = model_args.get("dim", 33)
        self.num_basis = model_args.get("num_basis", 64)

        self.experiment_name = name

        losses = train_args.get("losses", {})
        self.tv_weight = losses.get("tv_weight", 2e6)
        self.mn_weight = losses.get("mn_weight", 2e7)
        self.perceptual_content_weight = losses.get("perceptual_content_weight", 1.0)
        self.perceptual_style_weight = losses.get("perceptual_style_weight", 1.0)
        self.paired_diversity_weight = losses.get("paired_diversity_weight", 0.05)
        self.unpaired_diversity_weight = losses.get("unpaired_diversity_weight", 0.05)
        # New LPIPS weight parameter:
        self.lpips_weight = losses.get("lpips_weight", 1.0)

        self.hist_loss_weight = losses.get("hist_loss_weight", 1.0)
        self.lab_loss_weight = losses.get("lab_loss_weight", 1.0)

        gan_losses = losses.get("gan", {})
        paired_gan_losses = gan_losses.get("paired", {})
        unpaired_gan_losses = gan_losses.get("unpaired", {})

        self.paired_gan_generator_weight = paired_gan_losses.get(
            "generator_weight", 1.0
        )
        self.paired_gan_discriminator_weight = paired_gan_losses.get(
            "discriminator_weight", 1.0
        )
        self.unpaired_gan_generator_weight = unpaired_gan_losses.get(
            "generator_weight", 1.0
        )
        self.unpaired_gan_discriminator_weight = unpaired_gan_losses.get(
            "discriminator_weight", 1.0
        )

        self.learning_rate = optim_args.get("learning_rate", 1e-4)
        self.weight_decay = optim_args.get("weight_decay", 0.01)

        self.batch_size = train_args.get("batch_size", 4)

        # Save hyperparameters (excluding _class_path)
        hparams = {k: v for k, v in locals().items() if k != "_class_path"}
        self.save_hyperparameters(hparams)

        # Initialize and load pretrained style2vlog network
        self.style2vlognet = Style2VLogImage2ImageNet(in_channel=6)
        checkpoint = torch.load(
            "logs/fit_dual_style2vlog_v2/version_0/checkpoints/epoch=400-step=423857.ckpt",
            weights_only=True,
        )
        self.style2vlognet.load_state_dict(
            {
                k.replace("style2vlognet.", ""): v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith("style2vlognet")
            }
        )
        self.style2vlognet.requires_grad_(False)  # Freeze all parameters

        self.vlog2stylenet = VLog2StyleNet4D(dim=self.dim, num_basis=self.num_basis)
        if discriminator_model_name == "TAGANDiscriminator":
            self.style_discriminator = TAGANDiscriminator()

        # Visualization frequency parameter
        self.train_vis_freq = train_args.get(
            "train_vis_freq", 500
        )  # Visualize every 500 steps by default

        # Initialize LPIPS loss (using the VGG backbone)
        self.lpips_loss = lpips.LPIPS(net="vgg")
        self.hist_loss = HistLoss()
        self.lab_loss = LABLoss()

    def setup(self, stage=None):
        if stage == "fit":
            self.vgg_loss = PerceptualLoss()

        # Use the configured experiment name for visualization
        self.visual_dir = os.path.join("logs", "visualizations", self.experiment_name)
        try:
            os.makedirs(self.visual_dir, exist_ok=True)
            print(f"Visualization directory created at: {self.visual_dir}")
        except OSError as e:
            print(f"Warning: Error creating visualization directory: {e}")
            self.visual_dir = os.path.join("logs", "visualizations", "debug")
            os.makedirs(self.visual_dir, exist_ok=True)
            print(f"Falling back to debug directory: {self.visual_dir}")

    def forward(self, batch):
        """
        Forward pass of the model.

        Args:
            batch: Dictionary containing:
                inputs: {
                    paired: {
                        vlog: Content image in VLOG space
                        style: Target style image
                        style_pos: Positive style example (same style, different content)
                        style_neg: Negative style example (same content, different style)
                    }
                    unpaired: {
                        vlog: Content image in VLOG space, not used in unpaired
                        content: Target content image, need to convert to VLOG space
                        style_pos: Positive style example (a style reference image)
                        style_neg: Negative style example (same content, different style)
                        other_style: Additional style reference image
                    }
                }
        """
        inputs = batch["inputs"]
        outputs = {}
        if "paired" in inputs and "unpaired" in inputs:
            for name, input in inputs.items():
                vlog = input["vlog"]
                style = input["style"]
                style_pos = input["style_pos"]
                style_neg = input["style_neg"]

                # Generate fake_vlog using style2vlognet (frozen)
                with torch.no_grad():
                    fake_vlog = self.style2vlognet(style, style_neg)

                if self.training:
                    if name == "unpaired":
                        # Get model outputs for the unpaired branch
                        fake_style, fused_lut, feature, tvmn, context_map = (
                            self.vlog2stylenet(style_pos, fake_vlog)
                        )
                        paired_fake_style = None
                    else:  # name == "paired"
                        # Get model outputs for the paired branch
                        paired_fake_style, fused_lut, feature, tvmn, context_map = (
                            self.vlog2stylenet(style, vlog)
                        )
                        fake_style = None

                    outputs[name] = {
                        "vlog": vlog,
                        "style": style,
                        "fake_vlog": fake_vlog,
                        "unpaired_fake_style": fake_style,
                        "paired_fake_style": paired_fake_style,
                        "tvmn": tvmn if self.training else None,
                        "context_map": context_map if self.training else None,
                    }

            batch["outputs"] = outputs
            return batch
        else:
            # Validation/inference mode: only vlog and style are provided
            vlog = inputs["vlog"]
            style = inputs["style"]

            # Get model outputs in evaluation mode
            stylized, fused_lut, context_map = self.vlog2stylenet(style, vlog)
            batch["outputs"] = {
                "stylized": stylized,
                "fused_lut": fused_lut,
                "context_map": context_map,
            }
            return batch

    def generator_step(self, batch):
        """Generator optimization step."""
        outputs = batch["outputs"]
        inputs = batch["inputs"]

        loss = 0
        loss_dict = {}

        # TVMN Loss (Total Variation + Monotonicity)
        if self.tv_weight > 1e-8 or self.mn_weight > 1e-8:
            tvmn_loss = sum(
                self._compute_tvmn_loss(out["tvmn"]) for out in outputs.values()
            )
            loss += tvmn_loss
            loss_dict["train/tvmn_loss"] = tvmn_loss

        # Paired Domain Losses #####################################################
        if outputs["paired"]["paired_fake_style"] is not None:
            # Perceptual Loss using VGG features
            if (
                self.perceptual_content_weight > 1e-8
                or self.perceptual_style_weight > 1e-8
            ):
                content_loss, style_loss = self._compute_perceptual_loss(
                    inputs["paired"]["vlog"],
                    inputs["paired"]["style"],
                    outputs["paired"]["paired_fake_style"],
                )
                perceptual_loss = (
                    self.perceptual_content_weight * content_loss
                    + self.perceptual_style_weight * style_loss
                )
                loss += perceptual_loss
                loss_dict["train/paired_perceptual_loss"] = perceptual_loss

            # Generator GAN Loss (paired branch)
            if self.paired_gan_generator_weight > 1e-8:
                style_paired = inputs["paired"]["style"]
                pred_paired = outputs["paired"]["paired_fake_style"]
                g_loss = self._gan_loss(
                    self.style_discriminator(style_paired, pred_paired), "real"
                )
                loss += self.paired_gan_generator_weight * g_loss
                loss_dict["train/paired_gen_loss"] = g_loss

            # LPIPS Loss for paired branch
            if self.lpips_weight > 1e-8:
                lpips_loss_paired = self.lpips_loss(
                    inputs["paired"]["style"], outputs["paired"]["paired_fake_style"]
                ).mean()
                lpips_loss_paired = self.lpips_weight * lpips_loss_paired
                loss += lpips_loss_paired
                loss_dict["train/paired_lpips_loss"] = lpips_loss_paired

            # Hist Loss for paired branch
            if self.hist_loss_weight > 1e-8:
                hist_loss_paired = self.hist_loss(
                    inputs["paired"]["style"], outputs["paired"]["paired_fake_style"]
                )
                hist_loss_paired = self.hist_loss_weight * hist_loss_paired
                loss += hist_loss_paired
                loss_dict["train/paired_hist_loss"] = hist_loss_paired

            # Lab loss for paired branch
            if self.lab_loss_weight > 1e-8:
                lab_loss_paired = self.lab_loss(
                    inputs["paired"]["style"], outputs["paired"]["paired_fake_style"]
                )
                lab_loss_paired = self.lab_loss_weight * lab_loss_paired
                loss += lab_loss_paired
                loss_dict["train/paired_lab_loss"] = lab_loss_paired

        # Unpaired Domain Losses ###################################################
        if outputs["unpaired"]["unpaired_fake_style"] is not None:
            # Generator GAN Loss (unpaired branch)
            if self.unpaired_gan_generator_weight > 1e-8:
                style_unpaired = inputs["unpaired"]["style_pos"]
                pred_unpaired = outputs["unpaired"]["unpaired_fake_style"]
                g_loss_unpaired = self._gan_loss(
                    self.style_discriminator(style_unpaired, pred_unpaired), "real"
                )
                loss += self.unpaired_gan_generator_weight * g_loss_unpaired
                loss_dict["train/unpaired_gen_loss"] = g_loss_unpaired

            # Hist loss for unpaired branch
            if self.hist_loss_weight > 1e-8:
                hist_loss_unpaired = self.hist_loss(
                    inputs["unpaired"]["style_pos"],
                    outputs["unpaired"]["unpaired_fake_style"],
                )
                hist_loss_unpaired = self.hist_loss_weight * hist_loss_unpaired
                loss += hist_loss_unpaired
                loss_dict["train/unpaired_hist_loss"] = hist_loss_unpaired

        self.log_dict(
            loss_dict, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )

        # Visualization every train_vis_freq steps
        if self.global_step % self.train_vis_freq == 0:
            # Paired branch visualization
            if outputs["paired"]["paired_fake_style"] is not None:
                paired_vis = [
                    inputs["paired"]["vlog"],
                    inputs["paired"]["style"],
                    outputs["paired"]["paired_fake_style"],
                    outputs["paired"]["context_map"].repeat(1, 3, 1, 1),
                ]
                paired_grid = make_grid(
                    torch.cat(paired_vis, dim=0), nrow=self.batch_size
                )

            # Unpaired branch visualization
            if outputs["unpaired"]["unpaired_fake_style"] is not None:
                unpaired_vis = [
                    outputs["unpaired"]["fake_vlog"],
                    inputs["unpaired"]["style_pos"],
                    outputs["unpaired"]["unpaired_fake_style"],
                    outputs["unpaired"]["context_map"].repeat(1, 3, 1, 1),
                ]
                unpaired_grid = make_grid(
                    torch.cat(unpaired_vis, dim=0), nrow=self.batch_size
                )

            # Save images
            rank = self.global_rank if self.trainer else 0
            step_dir = os.path.join(self.visual_dir, f"{self.global_step}")
            train_dir = os.path.join(step_dir, "train")
            os.makedirs(train_dir, exist_ok=True)

            if outputs["paired"]["paired_fake_style"] is not None:
                save_image(
                    paired_grid, os.path.join(train_dir, f"rank{rank}_paired.jpg")
                )
            if outputs["unpaired"]["unpaired_fake_style"] is not None:
                save_image(
                    unpaired_grid, os.path.join(train_dir, f"rank{rank}_unpaired.jpg")
                )

        return loss

    def training_step(self, batch, batch_idx):
        """Training step handling both generator and discriminator updates."""
        batch = self(batch)
        optimizer_generator, optimizer_discriminator = self.optimizers()
        scheduler_generator, scheduler_discriminator = self.lr_schedulers()

        # Generator step with gradient clipping
        self.toggle_optimizer(optimizer_generator)
        loss_generator = self.generator_step(batch)
        optimizer_generator.zero_grad()
        self.manual_backward(loss_generator)
        self.clip_gradients(
            optimizer_generator, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
        )
        optimizer_generator.step()
        self.untoggle_optimizer(optimizer_generator)

        # Discriminator step with gradient clipping
        self.toggle_optimizer(optimizer_discriminator)
        loss_discriminator = self.discriminator_step(batch)
        optimizer_discriminator.zero_grad()
        self.manual_backward(loss_discriminator)
        self.clip_gradients(
            optimizer_discriminator,
            gradient_clip_val=0.5,
            gradient_clip_algorithm="norm",
        )
        optimizer_discriminator.step()
        self.untoggle_optimizer(optimizer_discriminator)

        # Update schedulers if last batch
        if self.trainer.is_last_batch:
            scheduler_generator.step()
            scheduler_discriminator.step()

        return loss_generator

    def validation_step(self, batch, batch_idx):
        """Validation step visualizing stylized outputs."""
        batch = self(batch)
        outputs = batch["outputs"]
        inputs = batch["inputs"]

        # Compute validation perceptual loss (content + style)
        val_loss_dict = {}
        content_loss, style_loss = self._compute_perceptual_loss(
            inputs["vlog"], inputs["style"], outputs["stylized"]
        )
        perceptual_loss = content_loss + style_loss
        val_loss_dict["val/perceptual_loss"] = perceptual_loss

        self.log_dict(val_loss_dict, on_epoch=True, prog_bar=True, sync_dist=True)

        # Visualization: original vlog, target style, stylized output, and context map
        visualization_items = [
            inputs["vlog"],
            inputs["style"],
            outputs["stylized"],
            outputs["context_map"].repeat(1, 3, 1, 1),
        ]
        if visualization_items:
            grid = make_grid(torch.cat(visualization_items), nrow=4)
            rank = (
                self.trainer.global_rank
                if self.trainer is not None and self.trainer.world_size > 1
                else 0
            )
            step_dir = os.path.join(self.visual_dir, f"{self.global_step}")
            os.makedirs(step_dir, exist_ok=True)
            save_path = os.path.join(step_dir, f"rank{rank}_batch{batch_idx}.jpg")
            save_image(grid, save_path)

    def test_step(self, batch, batch_idx):
        """Test step. Here we reuse the validation step."""
        self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        """Configure optimizers with warmup scheduling."""
        opt_g = torch.optim.AdamW(
            self.vlog2stylenet.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        opt_d = torch.optim.AdamW(
            self.style_discriminator.parameters(),
            lr=self.learning_rate
            * 0.1,  # Reduced to 1/10th of generator's learning rate
            weight_decay=self.weight_decay,
        )

        # Warmup scheduling for the generator optimizer
        warmup_epochs = 5
        warmup_scheduler_g = LambdaLR(
            opt_g, lr_lambda=lambda epoch: (epoch + 1) / max(1, warmup_epochs)
        )
        main_scheduler_g = CosineAnnealingLR(
            opt_g, T_max=self.trainer.max_epochs, eta_min=1e-7
        )
        sch_g = SequentialLR(
            opt_g, [warmup_scheduler_g, main_scheduler_g], [warmup_epochs]
        )
        sch_d = CosineAnnealingLR(opt_d, T_max=self.trainer.max_epochs, eta_min=1e-7)

        return [opt_g, opt_d], [sch_g, sch_d]

    def _compute_tvmn_loss(self, tvmn):
        """Compute Total Variation and Monotonicity Loss.
        Args:
            tvmn: Tensor containing (tv_loss, mn_loss)
        """
        tv_loss = tvmn[0]  # First element is TV loss
        mn_loss = tvmn[1]  # Second element is MN loss
        return self.tv_weight * tv_loss + self.mn_weight * mn_loss

    def _compute_perceptual_loss(self, content_img, style_img, input_img):
        """Compute perceptual loss using VGG features.
        Args:
            content_img: Content reference image
            style_img: Style reference image
            input_img: Generated image to evaluate
        """
        content_loss, style_loss = self.vgg_loss(content_img, style_img, input_img)
        return content_loss, style_loss

    def _gan_loss(self, pred, label):
        """Compute GAN loss.
        Args:
            pred: Discriminator predictions
            label: 'real' or 'fake'
        """
        target = (
            torch.ones_like(pred) if label.lower() == "real" else torch.zeros_like(pred)
        )
        return F.binary_cross_entropy(pred, target)

    def discriminator_step(self, batch):
        """Optimization step for discriminator."""
        outputs = batch["outputs"]
        inputs = batch["inputs"]
        loss_dict = {}
        total_loss = 0

        # Paired domain discriminator loss
        if (
            outputs["paired"]["paired_fake_style"]
            is not None
            # and self.paired_gan_discriminator_weight > 1e-8
        ):
            style_paired = inputs["paired"]["style"]
            pred_paired = outputs["paired"]["paired_fake_style"].detach()

            # Real pairs loss
            loss_real = self._gan_loss(
                self.style_discriminator(
                    inputs["paired"]["style"], inputs["paired"]["style_pos"]
                ),
                "real",
            )
            # Fake pairs loss
            loss_fake = self._gan_loss(
                self.style_discriminator(pred_paired, style_paired), "fake"
            )

            loss_d_paired = (loss_real + loss_fake) / 2
            total_loss += self.paired_gan_discriminator_weight * loss_d_paired
            loss_dict["train/disc_loss"] = loss_d_paired

        # Unpaired domain discriminator loss
        if (
            outputs["unpaired"]["unpaired_fake_style"]
            is not None
            # and self.unpaired_gan_discriminator_weight > 1e-8
        ):
            style_unpaired = inputs["unpaired"]["style_pos"]
            pred_unpaired = outputs["unpaired"]["unpaired_fake_style"].detach()

            # Real pairs loss
            loss_real = self._gan_loss(
                self.style_discriminator(
                    inputs["unpaired"]["style_pos"], inputs["unpaired"]["style_pos"]
                ),
                "real",
            )
            # Fake pairs loss
            loss_fake = self._gan_loss(
                self.style_discriminator(pred_unpaired, style_unpaired), "fake"
            )

            loss_d_unpaired = (loss_real + loss_fake) / 2
            total_loss += self.unpaired_gan_discriminator_weight * loss_d_unpaired
            loss_dict["train/disc_loss_unpaired"] = loss_d_unpaired

        self.log_dict(
            loss_dict, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return total_loss
