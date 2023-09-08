""" 
Given 25 satellite images of size 256 x 256 and their segmentation maps masking the
roof tops of houses, train a model to predict semantic segmentation maps of roofs.
Evaluate your model on the remaining 5 label-less images.

       Data:  dida-test-task
 Base model:  https://huggingface.co/facebook/maskformer-swin-base-ade
Performance:  mean iou: 0.889600184420778
              mean accuracy: 0.9403406767691679 
              per class iou: [0.9682773  0.81092307]

Usage:

    python dida_maskformer.py
"""
import datetime
import evaluate
import numpy as np
import os
import random
import torch

from PIL import Image
import shutil
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from transformers import (
    # MaskFormerConfig,
    MaskFormerImageProcessor,
    MaskFormerForInstanceSegmentation,
)


class RooftopDataset:
    """Custom dataset class for the dida roof segmentation task.

    Parameters:
    -----------

    root_dir : str
        The root directory of the dataset, i.e. the directory containing the image 
        and label subdirectories.

    filenames : list
        The names of the samples to include in this dataset; e.g., 417.png. This assumes
        that images and labels are named the same, and that they are in the same order.
        This can be used to split the dataset into train, validation, and test sets.

    transform : torchvision.transforms
        The transforms to be applied to the images.

    target_transform : torchvision.transforms
        The transforms to be applied to the labels.

    has_labels : bool
        The test dataset does not have labels, hence labels will only be returned 
        if has_labels=True.
    """

    def __init__(
        self,
        root_dir,
        filenames,
        transform=None,
        target_transform=None,
        has_labels=True,
    ):
        self.threshold = 128  # aka 0.5

        self.root_dir = root_dir
        self.filenames = filenames
        self.has_labels = has_labels

        self.images = []
        self.labels = []
        for fn in filenames:
            self.images.append(
                Image.open(os.path.join(root_dir, "images", fn)).convert("RGB")
            )
            if self.has_labels:
                self.labels.append(
                    Image.open(os.path.join(root_dir, "labels", fn)).convert("L")
                )

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        if not self.has_labels:
            return image
        # else: we can haz labels
        seg_map = self.labels[idx]
        if self.target_transform:
            seg_map = self.target_transform(seg_map)
        else:
            seg_map = np.array(seg_map)
            # apply Gaussian smoothing to the segmentation map before thresholding
            # import cv2
            # seg_map = cv2.GaussianBlur(seg_map, (3, 3), 1.)  # kernel, sigma
            seg_map[seg_map < self.threshold] = 0
            seg_map[seg_map >= self.threshold] = 1

        return image, seg_map


if __name__ == "__main__":
    # custom specs
    ROOT_DIR = "../data/dida_test_task"
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")
    LABEL_DIR = os.path.join(ROOT_DIR, "labels")
    RESULTS_DIR = os.path.join(ROOT_DIR, "outputs")
    base_model_id = "facebook/maskformer-swin-base-ade"
    val_size = 0.2
    batch_size = 4
    num_epochs = 50
    # num_epochs = 25
    # num_epochs = 1  # for testing
    learning_rate = 5e-5
    ignore_index = 255

    id2label = {"0": "other", "1": "roof"}
    label2id = {v: int(k) for k, v in id2label.items()}

    # stats for image normalization, here: ImageNet mean and std
    inet_mean = [0.485, 0.456, 0.406]
    inet_std = [0.229, 0.224, 0.225]

    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():  # commented out because this model
    #      device = torch.device("mps")        # uses code not implemented for mps yet
    else:
        device = torch.device("cpu")

    preprocessor = MaskFormerImageProcessor(
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
        ignore_index=ignore_index,
        do_reduce_labels=False,
    )
    # preprocessor = MaskFormerImageProcessor.from_pretrained(base_model_id)

    def collate_fn(batch):
        images, labels = zip(*batch)
        batch = preprocessor(
            images=images, segmentation_maps=labels, return_tensors="pt"
        )
        batch["images"] = images
        return batch

    def collate_fn_no_labels(batch):
        return preprocessor(images=batch, return_tensors="pt")

    ##### DATA #####

    image_files = os.listdir(IMAGE_DIR)
    label_files = os.listdir(LABEL_DIR)

    # filter out test images (i.e. images without labels) from the rest
    test_files = list(set(image_files) - set(label_files))
    train_files = label_files
    # split the training data into train and validation sets
    # for model selection, early stopping, etc.
    val_idx = int(val_size * len(label_files))
    random.shuffle(train_files)  # in place
    val_files = train_files[:val_idx]
    train_files = train_files[val_idx:]

    # define some image transforms, for training and evaluation
    # see also https://pytorch.org/vision/main/auto_examples/plot_transforms.html
    train_transform = T.Compose(
        [
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.25),
            T.RandomPosterize(bits=2, p=0.2),
            T.RandomAdjustSharpness(sharpness_factor=3, p=0.2),
            T.RandomAutocontrast(p=0.3),
            T.RandomEqualize(p=0.3),
            T.ToTensor(),  # [0, 255] -> [0., 1.] and (W, H, C) -> (C, W, H)
            T.Normalize(mean=inet_mean, std=inet_std),
        ]
    )
    test_transform = T.Compose(
        [T.ToTensor(), T.Normalize(mean=inet_mean, std=inet_std)]
    )

    train_dataset = RooftopDataset(
        ROOT_DIR, train_files, transform=train_transform, has_labels=True
    )
    val_dataset = RooftopDataset(
        ROOT_DIR, val_files, transform=test_transform, has_labels=True
    )
    test_dataset = RooftopDataset(
        ROOT_DIR, test_files, transform=test_transform, has_labels=False
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=5, shuffle=False, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=5, shuffle=False, collate_fn=collate_fn_no_labels
    )

    # # inspect predictions of original pre-trained model
    # model = MaskFormerForInstanceSegmentation.from_pretrained(base_model_id)
    # model.to(device)
    # for batch in train_dataloader:
    #     model.eval()
    #     outputs = model(pixel_values=batch["pixel_values"].to(device))
    #     target_sizes = [(256, 256), (256, 256), (256, 256), (256, 256)]
    #    results = preprocessor.post_process_semantic_segmentation(
    #        outputs, target_sizes=target_sizes)
    #    pred_seg_maps = [res.cpu().detach().numpy().astype(np.uint8) * 255 
    #                     for res in results]
    #    for i in range(len(pred_seg_maps)):
    #        image = (batch['pixel_values'][i].cpu().detach().numpy() * 255)
    #        image.astype(np.uint8).transpose(1, 2, 0)
    #        map = pred_seg_maps[i]
    #         image[map == 255] = np.array([255, 0, 0])  # red
    #         T.ToPILImage()(image).show()

    ##### MODEL #####

    # https://huggingface.co/docs/transformers/main/en/model_doc/maskformer
    # a MaskFormer model for instance segmentation based on mask classification
    # config = MaskFormerConfig.from_pretrained(base_model_id)
    model = MaskFormerForInstanceSegmentation.from_pretrained(
        base_model_id,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # Make a directory to save model parameters, predictions, etc. to
    now = datetime.datetime.now().strftime("%Y%m%d%H%M")
    os.makedirs(os.path.join(RESULTS_DIR, now), exist_ok=True)
    # also save the training script in its current state
    shutil.copy2(os.path.abspath(__file__), os.path.join(RESULTS_DIR, now))

    ##### TRAINING #####

    # criterion = the loss to optimize; here provided by the model:
    # classification loss + segmentation loss (binary cross-entropy + dice loss)
    metric = evaluate.load("mean_iou")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    warmup_phase = 5 * len(train_dataloader)  # 5 epochs, updated once per epoch
    # warmup_phase = 1 * len(train_dataloader)  # 1 epoch, updated every batch
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_phase, num_epochs * len(train_dataloader)
    )

    # variables for saving the best model parameters
    best_val_loss = float("inf")
    best_params = None
    best_epoch = -1

    for epoch in range(num_epochs):
        # train
        train_loss = 0.0
        num_train_batches = len(train_dataloader)
        model.train()
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()  # zero the parameter gradients

            # forward + backward + optimize
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                class_labels=[labels.to(device) for labels in batch["class_labels"]],
            )
            loss = outputs.loss  # calculate loss
            loss.backward()  # compute gradients
            optimizer.step()  # update parameters
            # lr_scheduler.step()    # update learning rate  # TODO: move to epoch level

            train_loss += loss.item()
        train_loss /= (
            num_train_batches * batch["pixel_values"].shape[0]
        )  # assumes constant batch size

        lr_scheduler.step()  # update learning rate

        # evaluate on validation set
        val_loss = 0.0
        num_val_batches = len(val_dataloader)
        model.eval()  # batchnorm and dropout behave differently in train and eval mode
        for batch in val_dataloader:
            # deactivates autograd engine, reduces memory usage, speeds up computations
            with torch.no_grad():
                outputs = model(
                    pixel_values=batch["pixel_values"].to(device),
                    mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                    class_labels=[
                        labels.to(device) for labels in batch["class_labels"]
                    ],
                )
            val_loss += outputs.loss.item()
        val_loss /= num_val_batches * batch["pixel_values"].shape[0]

        # save model weights with lowest validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = model.state_dict()
            best_epoch = epoch + 1
            print(f"Saving best model parameters at (val_loss={val_loss:.4f})")

        # print loss stats once per epoch
        # Note: here each epoch has 5 minibatches,
        # for larger dataset more frequent reporting is recommended
        print(
            f"Epoch {epoch + 1} train loss: {train_loss:.4f} val loss: {val_loss:.4f}"
        )

    # save the best model parameters to disk
    model_name = base_model_id.split("/")[-1]
    MODEL_PATH = os.path.join(
        RESULTS_DIR, now, f"{model_name}_epoch-{best_epoch + 1}.pth"
    )
    # torch.save(best_params, MODEL_PATH)

    ##### EVALUATION #####

    model.load_state_dict(best_params)  # load the best model parameters
    # given that the validation set is so tiny, consider using the last model parameters

    # 1. Compute standard metrics like mean intersection-over-union and mean accuracy
    # use the validation set because don't have labels for the test set ~ heresy
    model.eval()
    for batch in val_dataloader:
        with torch.no_grad():
            outputs = model(pixel_values=batch["pixel_values"].to(device))

        images = [
            np.array(Image.open(os.path.join(IMAGE_DIR, fn)).convert("RGB"))
            for fn in val_files
        ]
        target_sizes = [(image.shape[0], image.shape[1]) for image in images]

        # post-process results to retrieve semantic segmentation maps
        results = preprocessor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes
        )
        pred_seg_maps = [res.cpu().detach().numpy().astype(np.uint8) for res in results]
        # true_seg_maps = [mask.squeeze().to(torch.uint8) 
        #                  for mask in batch["mask_labels"]]
        true_seg_maps = [
            mask[label2id["roof"]].squeeze().to(torch.uint8)
            for mask in batch["mask_labels"]
        ]

        metrics = metric.compute(  # use _compute(..) ?
            predictions=pred_seg_maps,
            references=true_seg_maps,
            num_labels=len(id2label),
            ignore_index=ignore_index,
        )
        print(
            f"mean iou: {metrics['mean_iou']} | \
                mean accuracy: {metrics['mean_accuracy']} | \
                per class iou: {metrics['per_category_iou']}"
        )

        original_labels = [
            np.array(Image.open(os.path.join(LABEL_DIR, fn)).convert("L"))
            for fn in val_files
        ]
        # now, evaluate against the original labels
        metrics2 = metric.compute(
            predictions=pred_seg_maps,
            references=original_labels,
            num_labels=len(id2label),
            ignore_index=ignore_index,
        )
        print(
            f"mean iou: {metrics2['mean_iou']} | \
                mean accuracy: {metrics2['mean_accuracy']} | \
                per class iou: {metrics2['per_category_iou']}"
        )

        # mean intersection over union ~ the area of overlap between the predicted 
        # segmentation of an image
        # and the ground truth divided by the area of union between the predicted 
        # segmentation and the ground truth.
        # https://huggingface.co/docs/evaluate/types_of_evaluations

    # 2. Save the predicted labels and
    # plot predictions onto test images for qualitative inspection
    model.eval()
    for batch in test_dataloader:
        with torch.no_grad():
            outputs = model(pixel_values=batch["pixel_values"].to(device))

        images = [
            np.array(Image.open(os.path.join(IMAGE_DIR, fn)).convert("RGB"))
            for fn in test_files
        ]
        target_sizes = [(image.shape[0], image.shape[1]) for image in images]

        # compute semantic segmentation maps (as opposed to instance segmentation maps)
        segmaps = preprocessor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes
        )

        # save the predicted segmentation maps as images
        for i, map in enumerate(segmaps):
            map = T.ToPILImage()((map * 255).to(torch.uint8))
            map.save(os.path.join(RESULTS_DIR, now, f"predicted_roof_{i + 1}.png"))

        # plot predicted segmentation maps on top of the test images
        for i in range(len(images)):
            image = images[i]
            map = segmaps[i]
            image[map == 1] = np.array([255, 0, 0])
            image = T.ToPILImage()(image)
            image.save(os.path.join(RESULTS_DIR, now, f"roof_on_pic_{i + 1}.png"))
            image.show()

    # import ipdb; ipdb.set_trace()
