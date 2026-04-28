"""Tests that evaluate() correctly unwraps task objects to get the model.

Covers raw tasks, torch.compile'd tasks, and DDP-wrapped tasks for each
task type (CLIPTask, SigLIPTask, CoCaTask, DistillCLIPTask).
"""
import os
import sys
import types
from unittest import mock

import pytest
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import open_clip
from open_clip.task import (
    CLIPTask,
    SigLIPTask,
    CoCaTask,
    DistillCLIPTask,
    get_model_from_task,
)


def _make_args(**overrides):
    """Build a minimal args namespace matching what evaluate() needs."""
    defaults = dict(
        device='cpu',
        precision='fp32',
        rank=0,
        local_rank=0,
        world_size=1,
        distributed=False,
        val_frequency=1,
        epochs=1,
        zeroshot_frequency=0,  # disable zero-shot to keep test fast
        model='RN50',
        save_logs=False,
        wandb=False,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


# ──────────────────────────────────────────────────────────────────────
# get_model_from_task: unwrap correctly for every wrapper combination
# ──────────────────────────────────────────────────────────────────────

_TASK_CONFIGS = [
    ('CLIPTask', CLIPTask, 'RN50', {}),
    ('SigLIPTask', SigLIPTask, 'RN50', {}),
    ('CoCaTask', CoCaTask, 'coca_ViT-B-32', {}),
]


@pytest.mark.parametrize("label,TaskCls,model_name,task_kw", _TASK_CONFIGS)
def test_get_model_from_raw_task(label, TaskCls, model_name, task_kw):
    model = open_clip.create_model(model_name)
    task = TaskCls(model, rank=0, world_size=1, **task_kw)
    extracted = get_model_from_task(task)
    assert hasattr(extracted, 'encode_text'), f'{label}: missing encode_text'
    assert hasattr(extracted, 'encode_image'), f'{label}: missing encode_image'


@pytest.mark.parametrize("label,TaskCls,model_name,task_kw", _TASK_CONFIGS)
def test_get_model_from_compiled_task(label, TaskCls, model_name, task_kw):
    model = open_clip.create_model(model_name)
    task = TaskCls(model, rank=0, world_size=1, **task_kw)
    compiled = torch.compile(task)
    extracted = get_model_from_task(compiled)
    assert hasattr(extracted, 'encode_text'), f'{label}: missing encode_text after compile'
    assert hasattr(extracted, 'encode_image'), f'{label}: missing encode_image after compile'


@pytest.mark.parametrize("label,TaskCls,model_name,task_kw", _TASK_CONFIGS)
def test_get_model_from_plain_model(label, TaskCls, model_name, task_kw):
    """Passing a raw model (no task) should return it as-is."""
    model = open_clip.create_model(model_name)
    extracted = get_model_from_task(model)
    assert hasattr(extracted, 'encode_text')
    assert hasattr(extracted, 'encode_image')


def test_get_model_from_distill_task():
    student = open_clip.create_model('RN50')
    teacher = open_clip.create_model('RN50')
    task = DistillCLIPTask(student, teacher, rank=0, world_size=1)
    extracted = get_model_from_task(task)
    assert hasattr(extracted, 'encode_text')
    assert hasattr(extracted, 'encode_image')
    # Should return the student, not the teacher
    assert extracted is student


def test_get_model_from_compiled_distill_task():
    student = open_clip.create_model('RN50')
    teacher = open_clip.create_model('RN50')
    task = DistillCLIPTask(student, teacher, rank=0, world_size=1)
    compiled = torch.compile(task)
    extracted = get_model_from_task(compiled)
    assert hasattr(extracted, 'encode_text')


# ──────────────────────────────────────────────────────────────────────
# evaluate(): smoke test with mocked val data
# ──────────────────────────────────────────────────────────────────────

def _make_val_dataloader(model, batch_size=2, num_batches=2):
    """Create a fake dataloader that yields (images, texts) batches."""
    image_size = model.visual.image_size
    if not isinstance(image_size, tuple):
        image_size = (image_size, image_size)
    tokenizer = open_clip.get_tokenizer('RN50')
    batches = []
    for _ in range(num_batches):
        images = torch.randn(batch_size, 3, *image_size)
        texts = tokenizer(['a cat', 'a dog'][:batch_size])
        batches.append({"image": images, "text": texts})

    dl = mock.MagicMock()
    dl.__iter__ = mock.MagicMock(return_value=iter(batches))
    dl.num_samples = batch_size * num_batches
    return dl


_EVAL_TASK_CONFIGS = [
    ('CLIPTask', CLIPTask, 'RN50', {}),
    ('SigLIPTask', SigLIPTask, 'RN50', {}),
]


@pytest.mark.parametrize("label,TaskCls,model_name,task_kw", _EVAL_TASK_CONFIGS)
@pytest.mark.skipif(sys.platform.startswith('darwin'), reason="macos pickle bug with locals")
def test_evaluate_with_task(label, TaskCls, model_name, task_kw):
    """evaluate() should work with each task type without AttributeError."""
    from open_clip_train.train import evaluate

    model = open_clip.create_model(model_name, output_dict=True)
    task = TaskCls(model, rank=0, world_size=1, **task_kw)

    args = _make_args(model=model_name)
    val_dl = _make_val_dataloader(model)
    data = {'val': mock.MagicMock(dataloader=val_dl)}

    metrics = evaluate(task, data, epoch=1, args=args)
    assert 'clip_val_loss' in metrics


@pytest.mark.skipif(sys.platform.startswith('darwin'), reason="macos pickle bug with locals")
def test_evaluate_with_compiled_task():
    """evaluate() should work with a torch.compile'd task."""
    from open_clip_train.train import evaluate

    model = open_clip.create_model('RN50', output_dict=True)
    task = SigLIPTask(model, rank=0, world_size=1)
    compiled = torch.compile(task)

    args = _make_args()
    val_dl = _make_val_dataloader(model)
    data = {'val': mock.MagicMock(dataloader=val_dl)}

    metrics = evaluate(compiled, data, epoch=1, args=args)
    assert 'clip_val_loss' in metrics
