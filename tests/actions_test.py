"""
The following code is intended to be run only by GitHub actions for continuous integration and
testing purposes.
For implementation examples, see notebooks in the examples folder.
"""

import glob
import os
from pathlib import Path
import sys
from time import time

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torchvision import datasets, transforms

from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1, get_torch_home
from facenet_pytorch.models.mtcnn import MTCNN, fixed_image_standardization

#### CLEAR ALL OUTPUT FILES ####

checkpoints = glob.glob(os.path.join(get_torch_home(), "checkpoints/*"))
for c in checkpoints:
    print(f"Removing {c}")
    os.remove(c)

crop_files = glob.glob("data/test_images_aligned/**/*.png")
for c in crop_files:
    print(f"Removing {c}")
    os.remove(c)


#### TEST EXAMPLE IPYNB'S ####
test_dir = Path(__file__).parent
root_dir = test_dir.parent
example_dir = root_dir / "examples"

os.system(
    f"jupyter nbconvert --to python --stdout {example_dir}/infer.ipynb {example_dir}/finetune.ipynb > {example_dir}/tmptest.py"
)
sys.path.append(str(example_dir))
import tmptest  # noqa: F401
os.chdir(test_dir)


#### TEST MTCNN ####
def get_image(path, trans):
    img = Image.open(path)
    img = trans(img)
    return img


trans = transforms.Compose([transforms.Resize(512)])

trans_cropped = transforms.Compose([np.float32, transforms.ToTensor(), fixed_image_standardization])

dataset = datasets.ImageFolder("data/test_images", transform=trans)
dataset.idx_to_class = {k: v for v, k in dataset.class_to_idx.items()}

mtcnn_pt = MTCNN(device=torch.device("cpu"))

names = []
aligned = []
aligned_fromfile = []
for img, idx in dataset:
    name = dataset.idx_to_class[idx]
    start = time()
    img_align, _ = mtcnn_pt(img, save_path=[f"data/test_images_aligned/{name}/1.png"])
    print(f"MTCNN time: {time() - start:6f} seconds")

    # Comparison between types
    img_box = mtcnn_pt.detect([img])[0][0]
    assert (img_box - mtcnn_pt.detect(np.array(img))[0]).sum() < 1e-2
    assert (img_box - mtcnn_pt.detect(torch.as_tensor(np.array(img)))[0]).sum() < 1e-2

    # Batching test
    assert (img_box - mtcnn_pt.detect([img, img])[0]).sum() < 1e-2
    assert (img_box - mtcnn_pt.detect(np.array([np.array(img), np.array(img)]))[0]).sum() < 1e-2
    assert (img_box - mtcnn_pt.detect(torch.as_tensor([np.array(img), np.array(img)]))[0]).sum() < 1e-2

    # Box selection
    mtcnn_pt.selection_method = "probability"
    print("\nprobability - ", mtcnn_pt.detect(img))
    mtcnn_pt.selection_method = "largest"
    print("largest - ", mtcnn_pt.detect(img))
    mtcnn_pt.selection_method = "largest_over_threshold"
    print("largest_over_threshold - ", mtcnn_pt.detect(img))
    mtcnn_pt.selection_method = "center_weighted_size"
    print("center_weighted_size - ", mtcnn_pt.detect(img))

    if img_align is not None:
        names.append(name)
        aligned.append(img_align)
        aligned_fromfile.append(get_image(f"data/test_images_aligned/{name}/1.png", trans_cropped))

aligned = torch.stack(aligned)
aligned_fromfile = torch.stack(aligned_fromfile)


#### TEST EMBEDDINGS ####

expected = [
    [
        [0.000000, 1.482895, 0.886342, 1.438450, 1.437583],
        [1.482895, 0.000000, 1.345686, 1.029880, 1.061939],
        [0.886342, 1.345686, 0.000000, 1.363125, 1.338803],
        [1.438450, 1.029880, 1.363125, 0.000000, 1.066040],
        [1.437583, 1.061939, 1.338803, 1.066040, 0.000000],
    ],
    [
        [0.000000, 1.430769, 0.992931, 1.414197, 1.329544],
        [1.430769, 0.000000, 1.253911, 1.144899, 1.079755],
        [0.992931, 1.253911, 0.000000, 1.358875, 1.337322],
        [1.414197, 1.144899, 1.358875, 0.000000, 1.204118],
        [1.329544, 1.079755, 1.337322, 1.204118, 0.000000],
    ],
]

for i, ds in enumerate(["vggface2", "casia-webface"]):
    resnet_pt = InceptionResnetV1(pretrained=ds).eval()

    start = time()
    embs = resnet_pt(aligned)
    print(f"\nResnet time: {time() - start:6f} seconds\n")

    embs_fromfile = resnet_pt(aligned_fromfile)

    dists = [[(emb - e).norm().item() for e in embs] for emb in embs]
    dists_fromfile = [[(emb - e).norm().item() for e in embs_fromfile] for emb in embs_fromfile]

    print("\nOutput:")
    print(pd.DataFrame(dists, columns=names, index=names))
    print("\nOutput (from file):")
    print(pd.DataFrame(dists_fromfile, columns=names, index=names))
    print("\nExpected:")
    print(pd.DataFrame(expected[i], columns=names, index=names))

    total_error = (torch.tensor(dists) - torch.tensor(expected[i])).norm()
    total_error_fromfile = (torch.tensor(dists_fromfile) - torch.tensor(expected[i])).norm()

    print(f"\nTotal error: {total_error}, {total_error_fromfile}")

    if sys.platform != "win32":
        assert total_error < 1e-2
        assert total_error_fromfile < 1e-2

    #### TEST CLASSIFICATION ####
    resnet_pt = InceptionResnetV1(pretrained=ds, classify=True).eval()
    prob = resnet_pt(aligned)


#### MULTI-FACE TEST ####

mtcnn = MTCNN(keep_all=True)
img = Image.open("data/multiface.jpg")
boxes, probs, _ = mtcnn.detect(img)

draw = ImageDraw.Draw(img)
for i, box in enumerate(boxes):
    draw.rectangle(box.tolist())

mtcnn(img, save_path="data/tmp.png")


#### MTCNN TYPES TEST ####

img = Image.open("data/multiface.jpg")

mtcnn = MTCNN(keep_all=True)
boxes_ref, _, _ = mtcnn.detect(img)
mtcnn(img)

mtcnn = MTCNN(keep_all=True).double()
boxes_test, _, _ = mtcnn.detect(img)
mtcnn(img)

box_diff = boxes_ref[np.argsort(boxes_ref[:, 1])] - boxes_test[np.argsort(boxes_test[:, 1])]
total_error = np.sum(np.abs(box_diff))
print(f"\nfp64 Total box error: {total_error}")

assert total_error < 1e-2


# half is not supported on CPUs, only GPUs
if torch.cuda.is_available():
    mtcnn = MTCNN(keep_all=True, device="cuda").half()
    boxes_test, _, _ = mtcnn.detect(img)
    mtcnn(img)

    box_diff = boxes_ref[np.argsort(boxes_ref[:, 1])] - boxes_test[np.argsort(boxes_test[:, 1])]
    print(f"fp16 Total box error: {np.sum(np.abs(box_diff))}")

    # test new automatic multi precision to compare
    if hasattr(torch.cuda, "amp"):
        with torch.cuda.amp.autocast():
            mtcnn = MTCNN(keep_all=True, device="cuda")
            boxes_test, _, _ = mtcnn.detect(img)
            mtcnn(img)

        box_diff = boxes_ref[np.argsort(boxes_ref[:, 1])] - boxes_test[np.argsort(boxes_test[:, 1])]
        print(f"AMP total box error: {np.sum(np.abs(box_diff))}")


#### MULTI-IMAGE TEST ####

mtcnn = MTCNN(keep_all=True)
img = [Image.open("data/multiface.jpg"), Image.open("data/multiface.jpg")]
batch_boxes, batch_probs, _ = mtcnn.detect(img)

mtcnn(img, save_path=["data/tmp1.png", "data/tmp1.png"])
tmp_files = glob.glob("data/tmp*")
for f in tmp_files:
    os.remove(f)


#### NO-FACE TEST ####

img = Image.new("RGB", (512, 512))
mtcnn(img)
