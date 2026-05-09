"""Config parsing and validation tests."""

import pytest
import yaml

from tts_auto_eval.config import (
    ABTestConfig,
    Config,
    DatasetConfig,
    EvaluationConfig,
    ModelConfig,
    ReportConfig,
    load_ab_config,
    load_config,
)


class TestDefaultConfig:
    """test_default_config — Config() with defaults, check all default values."""

    def test_default_config(self):
        config = Config()

        # ModelConfig defaults
        assert config.model.type == "custom"
        assert config.model.module is None
        assert config.model.params == {}

        # EvaluationConfig defaults
        assert config.evaluation.language == "ko"
        assert config.evaluation.metrics == ["utmos", "wer"]
        assert config.evaluation.whisper_model == "large-v3"
        assert config.evaluation.whisper_backend == "openai-whisper"
        assert config.evaluation.device == "cpu"
        assert config.evaluation.reference_audio is None
        assert config.evaluation.model_cache_dir == ".cache"

        # DatasetConfig defaults
        assert config.dataset.path == "./datasets/ko/"
        assert config.dataset.categories is None
        assert config.dataset.max_samples is None

        # ReportConfig defaults
        assert config.report.formats == ["html", "csv"]
        assert config.report.output_dir == "./results/"
        assert config.report.compare_with is None
        assert config.report.include_audio is True


class TestLoadConfigFromYaml:
    """test_load_config_from_yaml — write a temp YAML file, load it, verify parsed values."""

    def test_load_config_from_yaml(self, tmp_path):
        yaml_content = {
            "model": {
                "type": "f5-tts",
                "module": "my_module",
                "params": {"speed": 1.2},
            },
            "evaluation": {
                "language": "en",
                "metrics": ["utmos", "wer", "pesq"],
                "whisper_model": "medium",
                "device": "cuda",
            },
            "dataset": {
                "path": "./datasets/en/",
                "categories": ["news", "conversation"],
                "max_samples": 100,
            },
            "report": {
                "formats": ["html"],
                "output_dir": "./output/",
                "compare_with": "baseline",
                "include_audio": False,
            },
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(yaml_content), encoding="utf-8")

        config = load_config(config_file)

        assert config.model.type == "f5-tts"
        assert config.model.module == "my_module"
        assert config.model.params == {"speed": 1.2}
        assert config.evaluation.language == "en"
        assert config.evaluation.metrics == ["utmos", "wer", "pesq"]
        assert config.evaluation.whisper_model == "medium"
        assert config.evaluation.device == "cuda"
        assert config.dataset.path == "./datasets/en/"
        assert config.dataset.categories == ["news", "conversation"]
        assert config.dataset.max_samples == 100
        assert config.report.formats == ["html"]
        assert config.report.output_dir == "./output/"
        assert config.report.compare_with == "baseline"
        assert config.report.include_audio is False


class TestModelConfigBuiltin:
    """test_model_config_builtin — ModelConfig(type='f5-tts')."""

    def test_model_config_builtin(self):
        model = ModelConfig(type="f5-tts")
        assert model.type == "f5-tts"
        assert model.module is None
        assert model.params == {}


class TestEvaluationConfigCustom:
    """test_evaluation_config_custom — EvaluationConfig with custom metrics and faster-whisper."""

    def test_evaluation_config_custom(self):
        eval_config = EvaluationConfig(
            metrics=["utmos", "pesq", "stoi"],
            whisper_backend="faster-whisper",
            device="cuda",
            language="en",
        )
        assert eval_config.metrics == ["utmos", "pesq", "stoi"]
        assert eval_config.whisper_backend == "faster-whisper"
        assert eval_config.device == "cuda"
        assert eval_config.language == "en"
        # Defaults still hold for unset fields
        assert eval_config.whisper_model == "large-v3"
        assert eval_config.reference_audio is None


class TestLoadConfigEmptyYaml:
    """test_load_config_empty_yaml — empty YAML file should return defaults."""

    def test_load_config_empty_yaml(self, tmp_path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("", encoding="utf-8")

        config = load_config(config_file)

        assert config.model.type == "custom"
        assert config.evaluation.language == "ko"
        assert config.evaluation.metrics == ["utmos", "wer"]
        assert config.dataset.path == "./datasets/ko/"
        assert config.report.formats == ["html", "csv"]


class TestABTestConfig:
    """test_ab_test_config — ABTestConfig with two different models."""

    def test_ab_test_config(self):
        ab_config = ABTestConfig(
            model_a=ModelConfig(type="f5-tts"),
            model_b=ModelConfig(type="custom", module="my_tts", params={"voice": "male"}),
        )
        assert ab_config.model_a.type == "f5-tts"
        assert ab_config.model_b.type == "custom"
        assert ab_config.model_b.module == "my_tts"
        assert ab_config.model_b.params == {"voice": "male"}
        # Defaults for other fields
        assert ab_config.evaluation.language == "ko"
        assert ab_config.dataset.path == "./datasets/ko/"
        assert ab_config.report.formats == ["html", "csv"]


class TestLoadABConfigFromYaml:
    """test_load_ab_config_from_yaml — write temp YAML, load ABTestConfig, verify both models."""

    def test_load_ab_config_from_yaml(self, tmp_path):
        yaml_content = {
            "model_a": {
                "type": "f5-tts",
                "params": {"speed": 1.0},
            },
            "model_b": {
                "type": "custom",
                "module": "competitor_tts",
                "params": {"quality": "high"},
            },
            "evaluation": {
                "language": "en",
                "metrics": ["utmos"],
            },
            "dataset": {
                "path": "./datasets/en/",
                "max_samples": 50,
            },
        }
        config_file = tmp_path / "ab_config.yaml"
        config_file.write_text(yaml.dump(yaml_content), encoding="utf-8")

        ab_config = load_ab_config(config_file)

        assert ab_config.model_a.type == "f5-tts"
        assert ab_config.model_a.params == {"speed": 1.0}
        assert ab_config.model_b.type == "custom"
        assert ab_config.model_b.module == "competitor_tts"
        assert ab_config.model_b.params == {"quality": "high"}
        assert ab_config.evaluation.language == "en"
        assert ab_config.evaluation.metrics == ["utmos"]
        assert ab_config.dataset.path == "./datasets/en/"
        assert ab_config.dataset.max_samples == 50
        # Report defaults
        assert ab_config.report.formats == ["html", "csv"]


class TestConfigWhisperBackendDefault:
    """test_config_whisper_backend_default — verify default is 'openai-whisper'."""

    def test_config_whisper_backend_default(self):
        eval_config = EvaluationConfig()
        assert eval_config.whisper_backend == "openai-whisper"

    def test_config_whisper_backend_default_via_config(self):
        config = Config()
        assert config.evaluation.whisper_backend == "openai-whisper"
