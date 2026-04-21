"""Unit tests for CsvDataset and get_csv_dataset."""
import types

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from open_clip_train.data import CsvDataset, get_csv_dataset


def _make_csv(tmp_path, n=3):
    """Create a tiny TSV dataset with n image/caption pairs."""
    rows = []
    for i in range(n):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (4, 4), color=(i, i, i)).save(p)
        rows.append({"filepath": str(p), "caption": f"cap_{i}"})
    csv_path = tmp_path / "data.tsv"
    pd.DataFrame(rows).to_csv(csv_path, sep="\t", index=False)
    return csv_path


# ---------------------------------------------------------------------------
# CsvDataset internals
# ---------------------------------------------------------------------------


def test_csvdataset_stores_pandas_series(tmp_path):
    """images and captions are pandas Series, not Python lists."""
    csv_path = _make_csv(tmp_path)
    tok = lambda xs: [x.upper() for x in xs]
    ds = CsvDataset(csv_path, transforms.ToTensor(), "filepath", "caption", sep="\t", tokenizer=tok)
    assert isinstance(ds.images, pd.Series)
    assert isinstance(ds.captions, pd.Series)


def test_csvdataset_len(tmp_path):
    csv_path = _make_csv(tmp_path, n=5)
    tok = lambda xs: xs
    ds = CsvDataset(csv_path, transforms.ToTensor(), "filepath", "caption", sep="\t", tokenizer=tok)
    assert len(ds) == 5


def test_csvdataset_getitem_returns_dict(tmp_path):
    csv_path = _make_csv(tmp_path)
    tok = lambda xs: [f"T::{x}" for x in xs]
    ds = CsvDataset(csv_path, transforms.ToTensor(), "filepath", "caption", sep="\t", tokenizer=tok)
    sample = ds[0]
    assert isinstance(sample, dict)
    assert set(sample.keys()) == {"image", "text"}
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["text"] == "T::cap_0"


def test_csvdataset_getitem_stringifies_numeric_caption(tmp_path):
    """Numeric captions are converted to strings before tokenizing."""
    rows = []
    for i in range(2):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (4, 4)).save(p)
        rows.append({"filepath": str(p), "caption": i})  # integer caption
    csv_path = tmp_path / "numeric.tsv"
    pd.DataFrame(rows).to_csv(csv_path, sep="\t", index=False)

    received = []
    tok = lambda xs: [received.append(xs[0]) or xs[0]]
    ds = CsvDataset(csv_path, transforms.ToTensor(), "filepath", "caption", sep="\t", tokenizer=tok)
    ds[0]
    # The tokenizer should receive the stringified caption
    assert received[0] == "0"


# ---------------------------------------------------------------------------
# get_csv_dataset
# ---------------------------------------------------------------------------


def test_get_csv_dataset_train_drop_last(tmp_path):
    """Training dataloader uses drop_last=True."""
    csv_path = _make_csv(tmp_path, n=3)
    args = types.SimpleNamespace(
        train_data=str(csv_path),
        val_data=str(csv_path),
        csv_img_key="filepath",
        csv_caption_key="caption",
        csv_separator="\t",
        distributed=False,
        batch_size=2,
        workers=0,
    )
    tok = lambda xs: xs
    train_info = get_csv_dataset(args, transforms.ToTensor(), is_train=True, tokenizer=tok)
    # 3 samples, batch_size=2, drop_last=True => 1 batch
    assert train_info.dataloader.num_batches == 1


def test_get_csv_dataset_val_keeps_partial(tmp_path):
    """Val dataloader uses drop_last=False."""
    csv_path = _make_csv(tmp_path, n=3)
    args = types.SimpleNamespace(
        train_data=str(csv_path),
        val_data=str(csv_path),
        csv_img_key="filepath",
        csv_caption_key="caption",
        csv_separator="\t",
        distributed=False,
        batch_size=2,
        workers=0,
    )
    tok = lambda xs: xs
    val_info = get_csv_dataset(args, transforms.ToTensor(), is_train=False, tokenizer=tok)
    # 3 samples, batch_size=2, drop_last=False => 2 batches
    assert val_info.dataloader.num_batches == 2


def test_get_csv_dataset_batches_are_dicts(tmp_path):
    """Batches from the dataloader should be dicts with 'image' and 'text'."""
    csv_path = _make_csv(tmp_path, n=4)
    args = types.SimpleNamespace(
        train_data=str(csv_path),
        val_data=str(csv_path),
        csv_img_key="filepath",
        csv_caption_key="caption",
        csv_separator="\t",
        distributed=False,
        batch_size=2,
        workers=0,
    )
    tok = lambda xs: [torch.tensor([ord(c) for c in x[:5]]) for x in xs]
    info = get_csv_dataset(args, transforms.ToTensor(), is_train=False, tokenizer=tok)
    batch = next(iter(info.dataloader))
    assert isinstance(batch, dict)
    assert "image" in batch and "text" in batch
    assert batch["image"].shape[0] == 2
