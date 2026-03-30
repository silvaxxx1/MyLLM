"""Tests for trainers, configs, datasets, and factory."""
import pytest
import torch
from torch.utils.data import DataLoader

from myllm.Configs import ModelConfig
from myllm.Train.configs.TrainerConfig import TrainerConfig
from myllm.Train.configs.SFTConfig import SFTTrainerConfig
from myllm.Train.configs.ClassiferConfig import ClassifierConfig
from myllm.Train.trainer import PretrainTrainer
from myllm.Train.sft_trainer import SFTTrainer
from myllm.Train.sft_classifer import SFTClassifierTrainer
from myllm.Train.factory import create_trainer
from myllm.Train.datasets.toy_dataset import (
    ToyPretrainDataset,
    ToySFTDataset,
    ToyClassificationDataset,
    get_toy_dataloader,
)
from myllm.tests.conftest import make_trainer_config


# ---------------------------------------------------------------------------
# Trainer config helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def base_cfg(tmp_path):
    return make_trainer_config(TrainerConfig, tmp_path)


@pytest.fixture
def sft_cfg(tmp_path):
    return make_trainer_config(SFTTrainerConfig, tmp_path)


@pytest.fixture
def clf_cfg(tmp_path):
    return make_trainer_config(
        ClassifierConfig, tmp_path,
        num_labels=3,
        class_names=["neg", "neu", "pos"],
    )


# ---------------------------------------------------------------------------
# Trainer configs
# ---------------------------------------------------------------------------

class TestTrainerConfig:

    def test_defaults_set(self, tmp_path):
        cfg = make_trainer_config(TrainerConfig, tmp_path)
        assert cfg.num_epochs == 1
        assert cfg.batch_size == 2
        assert cfg.seed == 42

    def test_output_dir_created(self, tmp_path):
        import os
        cfg = make_trainer_config(TrainerConfig, tmp_path)
        assert os.path.isdir(cfg.output_dir)

    def test_wandb_skipped_when_report_to_empty(self, tmp_path):
        # Should not raise even without wandb_project
        cfg = make_trainer_config(TrainerConfig, tmp_path, report_to=[])
        assert cfg.report_to == []


class TestSFTConfig:

    def test_task_type_default(self, sft_cfg):
        assert sft_cfg.task_type == "instruction"

    def test_invalid_task_type_raises(self, tmp_path):
        cfg = make_trainer_config(SFTTrainerConfig, tmp_path)
        cfg.task_type = "invalid"
        with pytest.raises(Exception):
            cfg.validate()


class TestClassifierConfig:

    def test_num_labels(self, clf_cfg):
        assert clf_cfg.num_labels == 3

    def test_class_names(self, clf_cfg):
        assert clf_cfg.class_names == ["neg", "neu", "pos"]

    def test_pooling_strategy_default(self, clf_cfg):
        from myllm.Train.configs.ClassiferConfig import PoolingStrategy
        assert clf_cfg.pooling_strategy == PoolingStrategy.LAST


# ---------------------------------------------------------------------------
# Toy datasets
# ---------------------------------------------------------------------------

class TestToyDatasets:

    def test_pretrain_length(self):
        ds = ToyPretrainDataset(num_samples=20, max_length=16, vocab_size=1000)
        assert len(ds) == 20

    def test_pretrain_item_has_input_ids(self):
        ds = ToyPretrainDataset(num_samples=5, max_length=16, vocab_size=1000)
        item = ds[0]
        assert "input_ids" in item
        assert len(item["input_ids"]) == 16

    def test_sft_length(self):
        ds = ToySFTDataset(num_samples=10, max_length=32)
        assert len(ds) == 10

    def test_sft_item_has_instruction_and_response(self):
        ds = ToySFTDataset(num_samples=5, max_length=32)
        item = ds[0]
        assert "instruction" in item
        assert "response" in item

    def test_classification_length(self):
        ds = ToyClassificationDataset(num_samples=10, max_length=32, num_classes=3)
        assert len(ds) == 10

    def test_classification_labels_in_range(self):
        ds = ToyClassificationDataset(num_samples=20, max_length=16, num_classes=3)
        for i in range(len(ds)):
            assert 0 <= ds[i]["label"] < 3

    def test_get_toy_dataloader_pretrain(self):
        dl = get_toy_dataloader("pretrain", batch_size=2, num_samples=6, max_length=16, vocab_size=1000)
        assert isinstance(dl, DataLoader)

    def test_unknown_dataset_type_raises(self):
        with pytest.raises(ValueError):
            get_toy_dataloader("unknown_type", batch_size=2)


# ---------------------------------------------------------------------------
# PretrainTrainer
# ---------------------------------------------------------------------------

class TestPretrainTrainer:

    def test_setup_model_returns_model(self, base_cfg, tiny_model_config):
        trainer = PretrainTrainer(base_cfg, model_config=tiny_model_config)
        model = trainer.setup_model()
        assert model is not None

    def test_setup_data_accepts_dataloader(self, base_cfg, tiny_model_config):
        trainer = PretrainTrainer(base_cfg, model_config=tiny_model_config)
        trainer.setup_model()
        dl = get_toy_dataloader("pretrain", batch_size=2, num_samples=4, max_length=16, vocab_size=1000)
        result = trainer.setup_data(train_dataloader=dl)
        assert result is not None

    def test_training_step_returns_loss(self, base_cfg, tiny_model_config, pretrain_batch):
        trainer = PretrainTrainer(base_cfg, model_config=tiny_model_config)
        trainer.setup_model()
        trainer.setup_optimizer()
        batch = trainer._prepare_batch(pretrain_batch)
        result = trainer.training_step(batch)
        assert "loss" in result
        assert isinstance(result["loss"], float)
        assert result["loss"] > 0

    def test_evaluation_step_returns_loss(self, base_cfg, tiny_model_config, pretrain_batch):
        trainer = PretrainTrainer(base_cfg, model_config=tiny_model_config)
        trainer.setup_model()
        batch = trainer._prepare_batch(pretrain_batch)
        result = trainer.evaluation_step(batch)
        assert "loss" in result

    def test_compute_loss_finite(self, base_cfg, tiny_model_config, pretrain_batch):
        trainer = PretrainTrainer(base_cfg, model_config=tiny_model_config)
        trainer.setup_model()
        batch = trainer._prepare_batch(pretrain_batch)
        labels = trainer._get_labels(batch)
        x = batch["input_ids"]
        logits = trainer.model(x)
        loss = trainer.compute_loss(logits, labels)
        assert torch.isfinite(loss)

    def test_checkpoint_save(self, base_cfg, tiny_model_config, tmp_path):
        trainer = PretrainTrainer(base_cfg, model_config=tiny_model_config)
        trainer.setup_model()
        ckpt = trainer.save_checkpoint(str(tmp_path / "ckpt"))
        import os
        assert os.path.exists(ckpt) or os.path.isdir(ckpt)


# ---------------------------------------------------------------------------
# SFTTrainer
# ---------------------------------------------------------------------------

class TestSFTTrainer:

    def test_setup_model_returns_model(self, sft_cfg, tiny_model_config):
        trainer = SFTTrainer(sft_cfg, model_config=tiny_model_config)
        model = trainer.setup_model()
        assert model is not None

    def test_training_step_returns_loss(self, sft_cfg, tiny_model_config, sft_batch):
        trainer = SFTTrainer(sft_cfg, model_config=tiny_model_config)
        trainer.setup_model()
        trainer.setup_optimizer()
        batch = trainer._prepare_batch(sft_batch)
        result = trainer.training_step(batch)
        assert "loss" in result
        assert isinstance(result["loss"], float)

    def test_response_masking_labels(self, sft_cfg, tiny_model_config, sft_batch):
        trainer = SFTTrainer(sft_cfg, model_config=tiny_model_config)
        trainer.setup_model()
        batch = trainer._prepare_batch(sft_batch)
        labels = trainer._get_labels(batch)
        # instruction portion should be masked (-100)
        assert (labels == -100).any()


# ---------------------------------------------------------------------------
# SFTClassifierTrainer
# ---------------------------------------------------------------------------

class TestSFTClassifierTrainer:

    def test_setup_model_attaches_classifier(self, clf_cfg, tiny_model_config):
        trainer = SFTClassifierTrainer(clf_cfg, model_config=tiny_model_config)
        trainer.setup_model()
        assert trainer.classifier is not None

    def test_forward_output_shape(self, clf_cfg, tiny_model_config, classification_batch):
        trainer = SFTClassifierTrainer(clf_cfg, model_config=tiny_model_config)
        trainer.setup_model()
        batch = trainer._prepare_batch(classification_batch)
        logits, _ = trainer.forward(batch)
        assert logits.shape == (2, clf_cfg.num_labels)

    def test_logits_are_finite(self, clf_cfg, tiny_model_config, classification_batch):
        trainer = SFTClassifierTrainer(clf_cfg, model_config=tiny_model_config)
        trainer.setup_model()
        batch = trainer._prepare_batch(classification_batch)
        logits, _ = trainer.forward(batch)
        assert torch.isfinite(logits).all()

    def test_training_step_returns_loss_and_accuracy(self, clf_cfg, tiny_model_config, classification_batch):
        trainer = SFTClassifierTrainer(clf_cfg, model_config=tiny_model_config)
        trainer.setup_model()
        trainer.setup_optimizer()
        batch = trainer._prepare_batch(classification_batch)
        result = trainer.training_step(batch)
        assert "loss" in result
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_evaluation_step_returns_metrics(self, clf_cfg, tiny_model_config, classification_batch):
        trainer = SFTClassifierTrainer(clf_cfg, model_config=tiny_model_config)
        trainer.setup_model()
        batch = trainer._prepare_batch(classification_batch)
        result = trainer.evaluation_step(batch)
        assert "loss" in result

    def test_pooling_hidden_states(self, clf_cfg, tiny_model_config, classification_batch):
        trainer = SFTClassifierTrainer(clf_cfg, model_config=tiny_model_config)
        trainer.setup_model()
        batch = trainer._prepare_batch(classification_batch)
        hidden = trainer._get_hidden_states(batch["input_ids"])
        pooled = trainer._pool_hidden_states(hidden)
        assert pooled.shape == (2, tiny_model_config.n_embd)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestTrainerFactory:

    def test_creates_pretrain_trainer(self, base_cfg, tiny_model_config):
        trainer = create_trainer("pretrain", base_cfg, model_config=tiny_model_config)
        assert isinstance(trainer, PretrainTrainer)

    def test_creates_sft_trainer(self, sft_cfg, tiny_model_config):
        trainer = create_trainer("sft", sft_cfg, model_config=tiny_model_config)
        assert isinstance(trainer, SFTTrainer)

    def test_creates_classifier_trainer(self, clf_cfg, tiny_model_config):
        trainer = create_trainer("sft_classifier", clf_cfg, model_config=tiny_model_config)
        assert isinstance(trainer, SFTClassifierTrainer)

    def test_invalid_type_raises(self, base_cfg):
        with pytest.raises(Exception):
            create_trainer("does_not_exist", base_cfg)
