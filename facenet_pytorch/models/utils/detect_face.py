from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn.functional import interpolate
from torchvision.ops.boxes import batched_nms
from torchvision.transforms import functional as F

if TYPE_CHECKING:
    from facenet_pytorch.models.mtcnn import ONet, PNet, RNet


def normalize_uint8_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor with values inside uint8 range to [-1, 1]."""
    return (tensor - 127.5) * 0.0078125  # 1/128


def fixed_batch_process(
    im_data: torch.Tensor, model: torch.nn.Module, batch_size: int = 512
) -> tuple[torch.Tensor, ...]:
    """Process a batch of images through the model.

    Args:
        im_data: Batch of images.
        model: Model to process the images with.
        batch_size: Batch size. (default: {512})

    Returns:
        Tuple of processed images results.

    """
    out: list[torch.Tensor] = []
    for i in range(0, len(im_data), batch_size):
        batch = im_data[i : i + batch_size]
        out.append(model(batch))

    return tuple(torch.cat(v, dim=0) for v in zip(*out))


def get_image_boxes(bboxes: torch.Tensor, imgs: torch.Tensor, image_idxs: torch.Tensor, size: int = 24) -> torch.Tensor:
    """Cut out boxes from the images.

    Arguments:
        bboxes: a float numpy array of shape [n, 5].
        imgs: the images.
        image_idxs: array of image indexes for each bbox, shape [n].
        size: an integer, size of cutouts.

    Returns:
        a float numpy array of shape [n, 3, size, size].
    """
    height, width = imgs.shape[2:4]

    y, ey, x, ex = pad(bboxes, width, height)
    im_datas = []
    for k in range(len(bboxes)):
        if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
            img_k = imgs[image_idxs[k], :, (y[k] - 1) : ey[k], (x[k] - 1) : ex[k]].unsqueeze(0)
            im_datas.append(image_resample(img_k, (size, size)))

    return normalize_uint8_tensor(torch.cat(im_datas, dim=0))


def calibrate_box(bboxes: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Transform bounding boxes to be more like true bounding boxes.

    Arguments:
        bboxes: shape [n, 5].
        offsets: shape [n, 4].

    Returns:
        shape [n, 5].
    """
    if offsets.shape[1] == 1:
        offsets = torch.reshape(offsets, (offsets.shape[2], offsets.shape[3]))
        raise RuntimeError("Shape of offsets is not correct.")

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    w = x2 - x1 + 1
    h = y2 - y1 + 1
    # w = np.expand_dims(w, 1)
    # h = np.expand_dims(h, 1)

    # Are offsets always such that x1 < x2 and y1 < y2 ?
    b1 = x1 + offsets[:, 0] * w
    b2 = y1 + offsets[:, 1] * h
    b3 = x2 + offsets[:, 2] * w
    b4 = y2 + offsets[:, 3] * h
    bboxes = torch.stack([b1, b2, b3, b4, bboxes[:, 4]]).permute(1, 0)

    # translation = torch.hstack([w, h, w, h]) * offsets.permute(1, 0)
    # bboxes[:, :4] += bboxes[:, :4] + translation
    return bboxes


def detect_face(
    imgs: torch.Tensor, min_size: int, pnet: PNet, rnet: RNet, onet: ONet, threshold: Sequence[float], factor: float
):
    # Scales the image so that
    # minimum size that we can detect equals to
    # minimum face size that we want to detect
    min_detection_size = 12
    h, w = imgs.shape[2:4]
    m = min_detection_size / min_size
    min_length = min(h, w)
    min_length *= m

    # Create scale pyramid
    scale_i = m
    scales = []
    while min_length > min_detection_size:
        scales.append(scale_i)
        scale_i = scale_i * factor
        min_length = min_length * factor

    # First stage
    bboxes = []
    image_idxs = []

    scale_picks = []

    offset = 0
    for scale in scales:
        im_data: torch.Tensor = image_resample(imgs, (int(h * scale + 1), int(w * scale + 1)))
        im_data = normalize_uint8_tensor(im_data)
        reg, probs = pnet(im_data)

        boxes_scale, image_idxs_scale = generate_bbox(reg, probs[:, 1], scale, threshold[0])
        bboxes.append(boxes_scale)
        image_idxs.append(image_idxs_scale)

        pick = batched_nms(boxes_scale[:, :4], boxes_scale[:, 4], image_idxs_scale, 0.5)
        scale_picks.append(pick + offset)
        offset += boxes_scale.shape[0]

    bboxes = torch.cat(bboxes, dim=0)
    image_idxs = torch.cat(image_idxs, dim=0)
    scale_picks = torch.cat(scale_picks, dim=0)

    # NMS within each scale + image
    bboxes, image_idxs = bboxes[scale_picks], image_idxs[scale_picks]

    # NMS within each image
    pick = batched_nms(bboxes[:, :4], bboxes[:, 4], image_idxs, 0.7)
    bboxes, image_idxs = bboxes[pick], image_idxs[pick]

    # use offsets predicted by pnet to transform bounding boxes
    bboxes = calibrate_box(bboxes[:, 0:5], bboxes[:, 5:])
    bboxes = convert_to_square(bboxes)

    # Second stage
    if len(bboxes) > 0:
        im_data = get_image_boxes(bboxes, imgs, image_idxs, 24)

        # This is equivalent to out = rnet(im_data) to avoid GPU out of memory.
        out = fixed_batch_process(im_data, rnet)

        offsets = out[0].permute(1, 0)  # offsets, shape [4, n]
        probs = out[1].permute(1, 0)  # probs, shape [2, n]
        prob = probs[1, :]
        ipass = prob > threshold[1]
        bboxes = torch.cat((bboxes[ipass, :4], prob[ipass].unsqueeze(1)), dim=1)
        image_idxs = image_idxs[ipass]
        offsets = offsets[:, ipass].permute(1, 0)

        # NMS within each image
        pick = batched_nms(bboxes[:, :4], bboxes[:, 4], image_idxs, 0.7)
        bboxes, image_idxs, offsets = bboxes[pick], image_idxs[pick], offsets[pick]
        bboxes = calibrate_box(bboxes, offsets)
        bboxes = convert_to_square(bboxes)

    # Third stage
    if len(bboxes) > 0:
        im_data = get_image_boxes(bboxes, imgs, image_idxs, 48)

        # This is equivalent to out = onet(im_data) to avoid GPU out of memory.
        out = fixed_batch_process(im_data, onet)

        landmarks = out[0].permute(1, 0)  # landmarks, shape [10, n]
        offsets = out[1].permute(1, 0)  # offsets, shape [4, n]
        probs = out[2].permute(1, 0)  # probs, shape [2, n]
        score = probs[1, :]

        ipass = score > threshold[2]
        bboxes = torch.cat((bboxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
        image_idxs = image_idxs[ipass]
        landmarks = landmarks[:, ipass]
        offsets = offsets[:, ipass].permute(1, 0)

        # compute landmark points
        w_i = bboxes[:, 2] - bboxes[:, 0] + 1
        h_i = bboxes[:, 3] - bboxes[:, 1] + 1
        x_min, y_min = bboxes[:, 0], bboxes[:, 1]

        landmarks_x = w_i.repeat(5, 1) * landmarks[:5, :] + x_min.repeat(5, 1) - 1
        landmarks_y = h_i.repeat(5, 1) * landmarks[5:10, :] + y_min.repeat(5, 1) - 1
        landmarks = torch.stack((landmarks_x, landmarks_y)).permute(2, 1, 0)

        bboxes = calibrate_box(bboxes, offsets)
        # NMS within each image using "min" strategy
        pick = batched_nms_numpy(bboxes[:, :4], bboxes[:, 4], image_idxs, 0.7, "min")
        bboxes, image_idxs, landmarks = bboxes[pick], image_idxs[pick], landmarks[pick]

    bboxes_np = bboxes.cpu().numpy()
    del bboxes
    landmarks_np = landmarks.cpu().numpy()
    del landmarks

    image_idxs = image_idxs.cpu()

    batch_boxes = []
    batch_landmarks = []
    for b_i in range(len(imgs)):
        b_i_inds = np.where(image_idxs == b_i)
        batch_boxes.append(bboxes_np[b_i_inds])
        batch_landmarks.append(landmarks_np[b_i_inds])

    batch_boxes, batch_landmarks = np.array(batch_boxes, dtype=object), np.array(batch_landmarks, dtype=object)

    return batch_boxes, batch_landmarks


def generate_bbox(
    reg: torch.Tensor, probs: torch.Tensor, scale: float, threshold: float, stride: int = 2, cell_size: int = 12
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate bounding boxes at places where there is probably a face.

    Arguments:
        reg: a float numpy array of shape [batch_size, 4, n, m].
        probs: a float numpy array of shape [batch_size, n, m].
        scale: this number scaled a float number, width and height of the image.
        threshold: a float number.
        stride: an integer.
        cell_size: an integer.

    Returns:
        Bounding boxes and image indices.
    """
    reg = reg.permute(1, 0, 2, 3)

    mask: torch.Tensor = probs >= threshold
    mask_idxs = mask.nonzero()
    image_idxs = mask_idxs[:, 0]
    scores = probs[mask]
    reg = reg[:, mask].permute(1, 0)
    bb = mask_idxs[:, 1:].type(reg.dtype).flip(1)
    q1 = ((stride * bb + 1) / scale).floor()
    q2 = ((stride * bb + cell_size - 1 + 1) / scale).floor()
    bbox = torch.cat([q1, q2, scores.unsqueeze(1), reg], dim=1)
    return bbox, image_idxs


def nms_numpy(bboxes: np.ndarray, scores: np.ndarray, threshold: float, method: str) -> np.ndarray:
    """Non-maximum suppression.

    Arguments:
        bboxes: a float numpy array of shape [n, 4],
            where each row is (xmin, ymin, xmax, ymax, score).
        scores: a float numpy array of shape [n].
        threshold: the threshold for deciding whether boxes overlap too much with respect to IOU.
        mode: 'union' or 'min'.

    Returns:
        List with indices of the selected boxes
    """
    if bboxes.size == 0:
        return np.empty((0, 3))

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    scores_idx = np.argsort(scores)
    # list of picked indices
    pick = np.zeros_like(scores, dtype=np.int16)
    counter = 0
    while scores_idx.size > 0:
        i = scores_idx[-1]
        pick[counter] = i
        counter += 1
        idx = scores_idx[0:-1]

        # Compute intersections of the box with the largest score with the rest of the boxes

        # Left top corner of intersection boxes
        ix1 = np.maximum(x1[i], x1[idx])
        iy1 = np.maximum(y1[i], y1[idx])

        # Right bottom corner of intersection boxes
        ix2 = np.minimum(x2[i], x2[idx])
        iy2 = np.minimum(y2[i], y2[idx])

        # Width and height of intersection boxes
        w = np.maximum(0.0, ix2 - ix1 + 1)
        h = np.maximum(0.0, iy2 - iy1 + 1)

        inter = w * h
        if method.lower() == "min":
            overlap = inter / np.minimum(area[i], area[i])
        elif method.lower() == "union":
            # intersection over union (IoU)
            overlap = inter / (area[i] + area[i] - inter)
        else:
            msg = f"Unknown NMS method: {method}"
            raise ValueError(msg)

        scores_idx = scores_idx[np.where(overlap <= threshold)]

    return pick[:counter]


def batched_nms_numpy(
    bboxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, threshold: float, method: str
) -> torch.Tensor:
    device = bboxes.device
    if bboxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    # Strategy: to perform NMS independently per class.
    # We add an offset to all the boxes.
    # The offset is dependent only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = bboxes.max()
    offsets = idxs.to(bboxes) * (max_coordinate + 1)
    boxes_for_nms = bboxes + offsets[:, None]
    boxes_for_nms_np = boxes_for_nms.cpu().numpy()
    del boxes_for_nms
    scores = scores.cpu().numpy()
    keep = nms_numpy(boxes_for_nms_np, scores, threshold, method)
    return torch.as_tensor(keep, dtype=torch.long, device=device)


def pad(bboxes: torch.Tensor, w: int, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the padding coordinates (pad the bounding boxes to square).

    Arguments:
        bboxes: a float numpy array of shape [n, 5].
        w: width of the original image.
        h: height of the original image.

    Returns:
        y, ey, x, ex:
    """
    bboxes = bboxes.trunc().int().cpu().numpy()
    x = bboxes[:, 0]
    y = bboxes[:, 1]
    ex = bboxes[:, 2]
    ey = bboxes[:, 3]

    x[x < 1] = 1
    y[y < 1] = 1
    ex[ex > w] = w
    ey[ey > h] = h

    return y, ey, x, ex


def convert_to_square(bboxes: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes to a square form.

    Arguments:
        bboxes: shape [n, 5].

    Returns:
        squared bounding boxes, shape [n, 5]
    """
    h = bboxes[:, 3] - bboxes[:, 1]
    w = bboxes[:, 2] - bboxes[:, 0]

    l = torch.max(w, h)
    bboxes[:, 0] = bboxes[:, 0] + w * 0.5 - l * 0.5
    bboxes[:, 1] = bboxes[:, 1] + h * 0.5 - l * 0.5
    bboxes[:, 2:4] = bboxes[:, :2] + l.repeat(2, 1).permute(1, 0)
    return bboxes


def image_resample(img: torch.Tensor, sz: tuple[int, int] | int | None) -> torch.Tensor:
    """Resample image to size using torch.nn.functional.interpolate.

    Arguments:
        img: Image tensor to be resampled.
        sz: Output size (sz x sz).

    Returns:
        Resampled image tensor.
    """
    return interpolate(img, size=sz, mode="area")


def crop_resize(
    img: np.ndarray | torch.Tensor, box: tuple[int, int, int, int], image_size: int | None
) -> np.ndarray | torch.Tensor:
    """Crop and resize face image.

    Arguments:
        img: Image to be cropped and resized.
        box: Bounding box around face.
        image_size: Image size, both height and width are the same.

    Returns:
        Cropped and resized image.
    """
    if isinstance(img, Image.Image):
        return img.crop(box).copy().resize((image_size, image_size), Image.Resampling.BILINEAR)

    img = img[:, box[1] : box[3], box[0] : box[2]]
    if isinstance(img, np.ndarray):
        return cv2.resize(img.copy(), (image_size, image_size), interpolation=cv2.INTER_AREA)

    # torch.Tensor
    return (
        image_resample(img.unsqueeze(0).float(), (image_size, image_size))
        .byte()
        .squeeze(0)
        # .permute(1, 2, 0)
    )


def save_img(img: Image | np.ndarray | torch.Tensor, path: os.PathLike) -> bool:
    """Save image to file."""
    if isinstance(img, Image.Image):
        try:
            img.save(path)
        except Exception:  # noqa: BLE001
            return False

        return True

    return cv2.imwrite(str(path), img[:, :, ::-1])  # BGR to RGB


def get_size(img: Image | np.ndarray | torch.Tensor) -> tuple[int, int]:
    if isinstance(img, Image.Image):
        return img.size

    return img.shape[1:]


def extract_face(
    img: np.ndarray | torch.Tensor,
    box: np.ndarray,
    image_size: int = 160,
    margin: int = 0,
    save_path: os.PathLike | str | None = None,
) -> torch.Tensor:
    """Extract face + margin from PIL Image given bounding box.

    Arguments:
        img: A PIL Image.
        box: Four-element bounding box.
        image_size: Output image size in pixels. The image will be square.
        margin: Margin to add to bounding box, in terms of pixels in the final image.
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size.
        save_path: Save path for extracted face image. (default: {None})

    Returns:
        Tensor representing the extracted face.
    """
    margin = [margin * (box[2] - box[0]) / (image_size - margin), margin * (box[3] - box[1]) / (image_size - margin)]
    raw_image_size = get_size(img)
    box = (
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, raw_image_size[0])),
        int(min(box[3] + margin[1] / 2, raw_image_size[1])),
    )

    face = crop_resize(img, box, image_size)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_img(face.permute(1, 2, 0).cpu().numpy(), save_path)

    return face.to(torch.float32)
