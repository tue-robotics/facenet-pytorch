from __future__ import annotations

import importlib.resources
import os
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

import facenet_pytorch.data
from facenet_pytorch.models.utils.detect_face import detect_face, extract_face, get_size

if TYPE_CHECKING:
    from PIL.Image import Image


class PNet(nn.Module):
    """MTCNN PNet.

    Keyword Arguments:
        pretrained {bool} -- Whether to load saved pretrained weights (default: {True})
    """

    def __init__(self, *, pretrained: bool = True) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3)
        self.prelu2 = nn.PReLU(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.prelu3 = nn.PReLU(32)
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
        self.softmax4_1 = nn.Softmax(dim=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)

        self.training = False

        if pretrained:
            with importlib.resources.path(facenet_pytorch.data, "pnet.pt") as state_dict_path:
                state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of PNet.

        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].

        Returns:
            b: a float tensor with shape [batch_size, 4, h', w'].
            a: a float tensor with shape [batch_size, 2, h', w'].
        """
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        a = self.conv4_1(x)
        a = self.softmax4_1(a)
        b = self.conv4_2(x)
        return b, a


class RNet(nn.Module):
    """MTCNN RNet.

    Keyword Arguments:
        pretrained {bool} -- Whether to load saved pretrained weights (default: {True})
    """

    def __init__(self, *, pretrained: bool = True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 28, kernel_size=3)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2)
        self.prelu3 = nn.PReLU(64)
        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU(128)
        self.dense5_1 = nn.Linear(128, 2)
        self.softmax5_1 = nn.Softmax(dim=1)
        self.dense5_2 = nn.Linear(128, 4)

        self.training = False

        if pretrained:
            with importlib.resources.path(facenet_pytorch.data, "rnet.pt") as state_dict_path:
                state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of RNet.

        Args:
            x: a float tensor with shape [batch_size, 3, h, w].

        Returns:
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense4(x.view(x.shape[0], -1))
        x = self.prelu4(x)
        a = self.dense5_1(x)
        a = self.softmax5_1(a)
        b = self.dense5_2(x)
        return b, a


class ONet(nn.Module):
    """MTCNN ONet.

    Keyword Arguments:
        pretrained {bool} -- Whether to load saved pretrained weights (default: {True})
    """

    def __init__(self, *, pretrained: bool = True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        self.prelu4 = nn.PReLU(128)
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)
        self.dense6_1 = nn.Linear(256, 2)
        self.softmax6_1 = nn.Softmax(dim=1)
        self.dense6_2 = nn.Linear(256, 4)
        self.dense6_3 = nn.Linear(256, 10)

        self.training = False

        if pretrained:
            with importlib.resources.path(facenet_pytorch.data, "onet.pt") as state_dict_path:
                state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of ONet.

        Args:
            x: a float tensor with shape [batch_size, 3, h, w].

        Returns:
            c: a float tensor with shape [batch_size, 10].
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense5(x.view(x.shape[0], -1))
        x = self.prelu5(x)
        a = self.dense6_1(x)
        a = self.softmax6_1(a)
        b = self.dense6_2(x)
        c = self.dense6_3(x)
        return b, c, a


class MTCNN(nn.Module):
    """MTCNN face detection module.

    This class loads pretrained P-, R-, and O-nets and returns images cropped to include the face
    only, given raw input images of one of the following types:
        - PIL image or list of PIL images
        - numpy.ndarray (uint8) representing either a single image (3D) or a batch of images (4D).
    Cropped faces can optionally be saved to file
    also.

    Keyword Arguments:
        image_size {int} -- Output image size in pixels. The image will be square. (default: {160})
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image.
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size (this is a bug in davidsandberg/facenet).
            (default: {0})
        min_face_size {int} -- Minimum face size to search for. (default: {20})
        thresholds {list} -- MTCNN face detection thresholds (default: {[0.6, 0.7, 0.7]})
        factor {float} -- Factor used to create a scaling pyramid of face sizes. (default: {0.709})
        post_process {bool} -- Whether to post process images tensors before returning.
            (default: {True})
        select_largest {bool} -- If True, if multiple faces are detected, the largest is returned.
            If False, the face with the highest detection probability is returned.
            (default: {True})
        selection_method {string} -- Which heuristic to use for selection. Default None. If
            specified, will override select_largest:
                    "probability": highest probability selected
                    "largest": largest box selected
                    "largest_over_threshold": largest box over a certain probability selected
                    "center_weighted_size": box size minus weighted squared offset from image center
                (default: {None})
        keep_all {bool} -- If True, all detected faces are returned, in the order dictated by the
            select_largest parameter. If a save_path is specified, the first face is saved to that
            path and the remaining faces are saved to <save_path>1, <save_path>2 etc.
            (default: {False})
        device {torch.device} -- The device on which to run neural net passes. Image tensors and
            models are copied to this device before running forward passes. (default: {None})
    """

    def __init__(
        self,
        image_size: int = 160,
        margin: int = 0,
        min_face_size: int = 20,
        thresholds: list[float] | None = None,
        factor: float = 0.709,
        post_process: bool = True,
        select_largest: bool = True,
        selection_method: str = "largest",
        keep_all: bool = False,
        device: torch.device | str | None = None,
    ):
        super().__init__()

        if thresholds is None:
            thresholds = [0.6, 0.7, 0.7]
        self.image_size: int = image_size
        self.margin: int = margin
        self.min_face_size: int = min_face_size
        self.thresholds: list[float] = thresholds
        self.factor: float = factor
        self.post_process: bool = post_process
        self.select_largest: bool = select_largest
        self.keep_all: bool = keep_all
        self.selection_method: str = selection_method

        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()

        self.device = torch.device("cpu")
        if device is not None:
            self.device = device
            self.to(device)

        if not self.selection_method:
            self.selection_method = "largest" if self.select_largest else "probability"

    def forward(
        self,
        img: Image | np.ndarray | torch.Tensor | list[Image | np.ndarray | torch.Tensor],
        save_path: os.PathLike | str | list[os.PathLike | str] | None = None,
    ):
        """Run MTCNN face detection on a list of images.

        This method performs both detection and extraction of faces, returning tensors representing detected faces rather
        than the bounding boxes. To access bounding boxes, see the MTCNN.detect() method below.

        Arguments:
            img {PIL.Image, np.ndarray, or list} -- A PIL image, np.ndarray, torch.Tensor, or list.

        Keyword Arguments:
            save_path {str} -- An optional save path for the cropped image. Note that when
                self.post_process=True, although the returned tensor is post processed, the saved
                face image is not, so it is a true representation of the face in the input image.
                If `img` is a list of images, `save_path` should be a list of equal length.
                (default: {None})
            return_prob {bool} -- Whether to return the detection probability.
                (default: {False})

        Returns:
            Union[torch.Tensor, tuple(torch.tensor, float)] -- If detected, cropped image of a face
                with dimensions 3 x image_size x image_size. Optionally, the probability that a
                face was detected. If self.keep_all is True, n detected faces are returned in an
                n x 3 x image_size x image_size tensor with an optional list of detection
                probabilities. If `img` is a list of images, the item(s) returned have an extra
                dimension (batch) as the first dimension.

        Example:
        >>> from facenet_pytorch import MTCNN
        >>> mtcnn = MTCNN()
        >>> face_tensor, prob = mtcnn(img, save_path="face.png")
        """
        # Detect faces
        batch_boxes, batch_probs, batch_points = self.detect(img)
        # Select faces
        if not self.keep_all:
            batch_boxes, batch_probs, batch_points = self.select_boxes(
                batch_boxes, batch_probs, batch_points, img, method=self.selection_method
            )
        # Extract faces
        faces = self.extract(img, batch_boxes, save_path)

        return faces, batch_probs

    def detect(
        self, img: Image | np.ndarray | torch.Tensor | list[Image | np.ndarray | torch.Tensor]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect all faces in PIL image and return bounding boxes and optional facial landmarks.

        This method is used by the forward method and is also useful for face detection tasks
        that require lower-level handling of bounding boxes and facial landmarks (e.g., face
        tracking). The functionality of the forward function can be emulated by using this method
        followed by the extract_face() function.

        Args:
            img: A PIL image, np.ndarray, torch.Tensor, or list.

        Returns:
            For N detected faces, a tuple containing a Nx4 array of bounding boxes and a length
            N list of detection probabilities. Returned boxes will be sorted in descending order
            by detection probability if self.select_largest=False, otherwise the largest face will
            be returned first. If `img` is a list of images, the items returned have an extra dimension
            (batch) as the first dimension. The third item is the facial landmarks.

        Example:
        >>> from PIL import Image, ImageDraw
        >>> from facenet_pytorch import MTCNN, extract_face
        >>> mtcnn = MTCNN(keep_all=True)
        >>> boxes, probs, points = mtcnn.detect(img, landmarks=True)
        >>> # Draw boxes and save faces
        >>> img_draw = img.copy()
        >>> draw = ImageDraw.Draw(img_draw)
        >>> for i, (box, point) in enumerate(zip(boxes, points)):
        ...     draw.rectangle(box.tolist(), width=5)
        ...     for p in point:
        ...         draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
        ...     extract_face(img, box, save_path=f"detected_face_{i}.png")
        >>> img_draw.save("annotated_faces.png")
        """
        with torch.no_grad():
            batch_boxes, batch_points = detect_face(
                img, self.min_face_size, self.pnet, self.rnet, self.onet, self.thresholds, self.factor, self.device
            )

        boxes, probs, points = [], [], []
        for box, point in zip(batch_boxes, batch_points):
            box = np.array(box)
            point = np.array(point)
            if len(box) == 0:
                boxes.append(None)
                probs.append([None])
                points.append(None)
            elif self.select_largest:
                box_order = np.argsort((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]))[::-1]
                box = box[box_order]
                point = point[box_order]
                boxes.append(box[:, :4])
                probs.append(box[:, 4])
                points.append(point)
            else:
                boxes.append(box[:, :4])
                probs.append(box[:, 4])
                points.append(point)

        boxes = np.array(boxes, dtype=object)
        probs = np.array(probs, dtype=object)
        points = np.array(points, dtype=object)

        if (
            not isinstance(img, (list, tuple))
            and not (isinstance(img, np.ndarray) and len(img.shape) == 4)
            and not (isinstance(img, torch.Tensor) and len(img.shape) == 4)
        ):
            boxes = boxes[0]
            probs = probs[0]
            points = points[0]

        return boxes, probs, points

    def select_boxes(
        self,
        all_boxes: np.ndarray,
        all_probs: np.ndarray,
        all_points: np.ndarray,
        imgs: list[Image | np.ndarray | torch.Tensor],
        method: str = "probability",
        threshold: float = 0.9,
        center_weight: float = 2.0,
    ):
        """Selects a single box from multiple for a given image using one of multiple heuristics.

        Arguments:
                all_boxes {np.ndarray} -- Ix0 ndarray where each element is a Nx4 ndarry of
                    bounding boxes for N detected faces in I images (output from self.detect).
                all_probs {np.ndarray} -- Ix0 ndarray where each element is a Nx0 ndarry of
                    probabilities for N detected faces in I images (output from self.detect).
                all_points {np.ndarray} -- Ix0 ndarray where each element is a Nx5x2 array of
                    points for N detected faces. (output from self.detect).
                imgs {PIL.Image, np.ndarray, or list} -- A PIL image, np.ndarray, torch.Tensor, or list.

        Keyword Arguments:
                method {str} -- Which heuristic to use for selection:
                    "probability": highest probability selected
                    "largest": largest box selected
                    "largest_over_theshold": largest box over a certain probability selected
                    "center_weighted_size": box size minus weighted squared offset from image center
                    (default: {'probability'})
                threshold {float} -- theshold for "largest_over_threshold" method. (default: {0.9})
                center_weight {float} -- weight for squared offset in center weighted size method.
                    (default: {2.0})

        Returns:
                tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray) -- nx4 ndarray of bounding boxes
                    for n images. Ix0 array of probabilities for each box, array of landmark points.
        """
        # copying batch detection from extract, but would be easier to ensure detect creates consistent output.
        batch_mode = True
        if (
            not isinstance(imgs, (list, tuple))
            and not (isinstance(imgs, np.ndarray) and len(imgs.shape) == 4)
            and not (isinstance(imgs, torch.Tensor) and len(imgs.shape) == 4)
        ):
            imgs = [imgs]
            all_boxes = [all_boxes]
            all_probs = [all_probs]
            all_points = [all_points]
            batch_mode = False

        selected_boxes, selected_probs, selected_points = [], [], []
        for boxes, points, probs, img in zip(all_boxes, all_points, all_probs, imgs):
            if boxes is None:
                selected_boxes.append(None)
                selected_probs.append([None])
                selected_points.append(None)
                continue

            # If at least 1 box found
            boxes = np.array(boxes)
            probs = np.array(probs)
            points = np.array(points)

            if method == "largest":
                box_order = np.argsort((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[::-1]
            elif method == "probability":
                box_order = np.argsort(probs)[::-1]
            elif method == "center_weighted_size":
                box_sizes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                height, width = get_size(img)
                img_center = (width / 2, height / 2)
                box_centers = np.asarray(list(zip((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2)))
                offsets = box_centers - img_center
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 1)
                box_order = np.argsort(box_sizes - offset_dist_squared * center_weight)[::-1]
            elif method == "largest_over_threshold":
                box_mask = probs > threshold
                boxes = boxes[box_mask]
                box_order = np.argsort((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[::-1]
                if sum(box_mask) == 0:
                    selected_boxes.append(None)
                    selected_probs.append([None])
                    selected_points.append(None)
                    continue
            else:
                msg = f"Invalid selection method: {method}"
                raise ValueError(msg)

            box = boxes[box_order][[0]]
            prob = probs[box_order][[0]]
            point = points[box_order][[0]]
            selected_boxes.append(box)
            selected_probs.append(prob)
            selected_points.append(point)

        if batch_mode:
            selected_boxes = np.array(selected_boxes)
            selected_probs = np.array(selected_probs)
            selected_points = np.array(selected_points)
        else:
            selected_boxes = selected_boxes[0]
            selected_probs = selected_probs[0][0]
            selected_points = selected_points[0]

        return selected_boxes, selected_probs, selected_points

    def extract(
        self,
        img: Image | np.ndarray | torch.Tensor | list[Image | np.ndarray | torch.Tensor],
        batch_boxes: torch.Tensor,
        save_path: os.PathLike | str | list[os.PathLike | str] | None,
    ):
        # Determine if a batch or single image was passed
        batch_mode = True
        if (
            not isinstance(img, (list, tuple))
            and not (isinstance(img, np.ndarray) and len(img.shape) == 4)
            and not (isinstance(img, torch.Tensor) and len(img.shape) == 4)
        ):
            img = [img]
            batch_boxes = [batch_boxes]
            batch_mode = False

        # Parse save path(s)
        if save_path is not None:
            if isinstance(save_path, str):
                save_path = [save_path]
        else:
            save_path = [None] * len(img)

        # Process all bounding boxes
        faces = []
        for im, box_im, path_im in zip(img, batch_boxes, save_path):
            if box_im is None:
                faces.append(None)
                continue

            if not self.keep_all:
                box_im = box_im[[0]]

            # If box contains a single face, convert to list for handling
            if isinstance(box_im, (np.ndarray)) and len(box_im) == 4:  # noqa: PLR2004
                box_im = [box_im]

            faces_im = []
            for i, box in enumerate(box_im):
                face_path = path_im
                if path_im is not None and i > 0:
                    save_name, ext = os.path.splitext(path_im)
                    face_path = save_name + "_" + str(i + 1) + ext
                face = extract_face(im, box, self.image_size, self.margin, face_path)
                if self.post_process:
                    face = fixed_image_standardization(face)
                faces_im.append(face)

            faces_im = torch.stack(faces_im) if self.keep_all else faces_im[0]

            faces.append(faces_im.cpu())

        if not batch_mode:
            faces = faces[0]

        return faces


def fixed_image_standardization(image_tensor: torch.Tensor) -> torch.Tensor:
    """Fixed image standardization."""
    return (image_tensor - 127.5) / 128.0


def prewhiten(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Pre-whiten image data."""
    mean = x.mean()
    std = x.std()
    std_adj = std.clamp(min=1.0 / (float(x.numel()) ** 0.5))
    return (x - mean) / std_adj  # y
