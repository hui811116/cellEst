import argparse
import logging
from pathlib import Path
from typing import cast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from cellest.classifiers import get_device, get_transforms, setup_seed
from cellest.classifiers.cell import PreFc
from cellest.classifiers.pretrained import HFTransformerGradCAM, TransformerCla
from cellest.config import InferenceConfig, InferenceModelConfig


logger = logging.getLogger(__name__)


class InferenceDataset(Dataset):
    """ImageFolder dataset that preserves image paths for output naming."""

    def __init__(self, data_path: Path, transform=None):
        self.images = ImageFolder(str(data_path))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path, label = self.images.samples[index]
        image = self.images.loader(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, image_path


def collate_images(batch):
    images, labels, paths = zip(*batch)
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images)
    return images, torch.tensor(labels), paths


def build_dataloader(config: InferenceConfig, model_info: InferenceModelConfig):
    transform = None
    if model_info.model.arch == "cell":
        _, transform = get_transforms(model_info.model.premodel)

    dataset = InferenceDataset(config.data.data_path, transform)
    if dataset.images.classes != model_info.class_names:
        raise ValueError(
            "Dataset class order does not match the model summary: "
            f"{dataset.images.classes} != {model_info.class_names}"
        )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_images,
    )
    return dataset, loader


def build_gradcam(model_info: InferenceModelConfig, state_dict, device):
    model_config = model_info.model
    if model_config.arch == "cell":
        model = PreFc(
            nclasses=model_info.num_classes,
            premodel=model_config.premodel,
            classifier_type=model_config.classifier_type,
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        if model_config.premodel == "resnet":
            target_layer = model.extract_net.layer4[-1]
        elif model_config.premodel == "inception":
            target_layer = model.extract_net.Mixed_7c
        else:
            raise ValueError(f"Unsupported cell model: {model_config.premodel}")

        cam = GradCAM(model=model, target_layers=[target_layer])
        return model, cam

    classifier = TransformerCla(
        model_name=cast(str, model_config.pretrained_path),
        num_classes=model_info.num_classes,
    ).to(device)
    classifier.load_state_dict(state_dict)
    model = HFTransformerGradCAM(classifier).to(device)
    model.eval()
    cam = GradCAM(model=model, target_layers=[model.target_layer], reshape_transform=model.reshape_transform)
    return model, cam


def denormalize_cell_images(images):
    mean = np.array([0.485, 0.456, 0.406])[None, :, None, None]
    std = np.array([0.229, 0.224, 0.225])[None, :, None, None]
    images = images.detach().cpu().numpy()
    return np.clip((images * std + mean).transpose(0, 2, 3, 1), 0, 1)


def denormalize_transformer_images(images, processor):
    mean = torch.tensor(processor.image_mean, device=images.device)[None, :, None, None]
    std = torch.tensor(processor.image_std, device=images.device)[None, :, None, None]
    images = images.detach() * std + mean
    return images.clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)


def resize_cam(cam_map, image):
    height, width = image.shape[:2]
    cam_tensor = torch.from_numpy(cam_map).unsqueeze(0).unsqueeze(0)
    resized = torch.nn.functional.interpolate(
        cam_tensor,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze().numpy()


def save_gradcam_image(
    image, cam_map, image_path, true_label, dataset, prediction, output_dir, threshold
):
    cam_map = resize_cam(cam_map, image)
    overlay = show_cam_on_image(image.astype(np.float32), cam_map, use_rgb=True)
    relative_path = Path(image_path).relative_to(dataset.images.root)
    save_path = output_dir / relative_path.parent / f"gradcam_{relative_path.stem}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(overlay)
    axis.contour(cam_map, levels=[threshold], colors="red")
    axis.set_title(
        f"True: {dataset.images.classes[true_label]} | "
        f"Pred: {dataset.images.classes[prediction]}"
    )
    axis.axis("off")
    figure.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(figure)


def run_inference(dataset, loader, model, cam, config: InferenceConfig, device):
    config.output_dir.mkdir(parents=True, exist_ok=True)
    target_index = dataset.images.class_to_idx[config.target_class]
    target = ClassifierOutputTarget(target_index)

    try:
        for images, labels, paths in tqdm.tqdm(loader, desc="Inference"):
            if isinstance(model, HFTransformerGradCAM):
                input_batch = model.preprocess(images).requires_grad_(True)
                display_images = denormalize_transformer_images(
                    input_batch, model.processor
                )
            else:
                input_batch = images.to(device).requires_grad_(True)
                display_images = denormalize_cell_images(images)

            cam_maps = cam(
                input_tensor=input_batch,
                targets=[target] * input_batch.shape[0],
            )
            with torch.no_grad():
                logits = model(input_batch)

            predictions = logits.argmax(dim=1).detach().cpu().tolist()
            for index, image_path in enumerate(paths):
                save_gradcam_image(
                    display_images[index],
                    cam_maps[index],
                    image_path,
                    labels[index].item(),
                    dataset,
                    predictions[index],
                    config.output_dir,
                    config.threshold,
                )
    finally:
        if cam is not None:
            cam.activations_and_grads.release()

    logger.info("GradCAM images saved to %s", config.output_dir.resolve())


def main(args):
    config = InferenceConfig.from_yaml(Path(args.config))
    model_info = InferenceModelConfig.from_summary(
        config.model_path.parent / "summary.json"
    )
    device = get_device(force_cpu=args.force_cpu)
    setup_seed(args.seed)

    checkpoint = torch.load(config.model_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise ValueError(f"Checkpoint does not contain model weights: {config.model_path}")

    dataset, loader = build_dataloader(config, model_info)
    if config.target_class not in dataset.images.class_to_idx:
        raise ValueError(
            f"Target class '{config.target_class}' not found in dataset classes: "
            f"{dataset.images.classes}"
        )

    model, cam = build_gradcam(model_info, state_dict, device)
    run_inference(dataset, loader, model, cam, config, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GradCAM visualizations.")
    parser.add_argument("--config", type=str, required=True, help="Path to inference YAML.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    main(args)