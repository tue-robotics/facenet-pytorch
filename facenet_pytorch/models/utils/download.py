from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    import _hashlib

try:
    from tqdm.auto import tqdm  # automatically select proper tqdm submodule if available
except ImportError:
    from tqdm import tqdm


def download_url_to_file(url: str, dst: Path | str, hash_prefix: str | None = None, *, progress: bool = True) -> None:
    """Download object at the given URL to a local path.

    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
        hash_prefix (string, optional): If not None, the SHA256 downloaded file should start with `hash_prefix`.
            Default: None
        progress (bool, optional): whether to display a progress bar to stderr
            Default: True
    Example:
        >>> torch.hub.download_url_to_file("https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth", "/tmp/temporary_file")
    """
    file_size = None
    # We use a different API for python2 since urllib(2) doesn't recognize the CA
    # certificates in older Python
    req = Request(url, headers={"User-Agent": "torch.hub"})  # noqa: S310
    with urlopen(req) as response:  # noqa: S310
        meta = response.info()
        if hasattr(meta, "getheaders"):
            content_length = meta.getheaders("Content-Length")
        else:
            content_length = meta.get_all("Content-Length")
        if content_length is not None and len(content_length) > 0:
            file_size = int(content_length[0])

        # We deliberately save it in a temp file and move it after
        # download is complete. This prevents a local working checkpoint
        # being overridden by a broken download.
        dst = dst.expanduser()
        dst_dir = dst.parent
        f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

        sha256: _hashlib.HASH | None = None
        if hash_prefix is not None:
            sha256 = hashlib.sha256()

        try:
            with tqdm(total=file_size, disable=not progress, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
                while True:
                    buffer = response.read(8192)
                    if len(buffer) == 0:
                        break
                    f.write(buffer)
                    if sha256 is not None:
                        sha256.update(buffer)
                    pbar.update(len(buffer))

            f.close()
            if sha256 is not None:
                digest = sha256.hexdigest()
                if digest[: len(hash_prefix)] != hash_prefix:
                    msg = f"invalid hash value (expected '{hash_prefix}', got '{digest}')"
                    raise RuntimeError(msg)
            shutil.move(f.name, dst)
        finally:
            f.close()
            if Path(f.name).exists():
                Path(f.name).unlink()
