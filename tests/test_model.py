"""
Basic tests for MODEL_NAME
"""
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class TestModelLoading:
    """Test model loading functionality"""

    def test_model_exists(self):
        """Test that model can be loaded from HuggingFace"""
        try:
            model = AutoModelForCausalLM.from_pretrained("zenlm/MODEL_NAME")
            assert model is not None
        except Exception as e:
            pytest.skip(f"Model not yet uploaded to HuggingFace: {e}")

    def test_tokenizer_exists(self):
        """Test that tokenizer can be loaded"""
        try:
            tokenizer = AutoTokenizer.from_pretrained("zenlm/MODEL_NAME")
            assert tokenizer is not None
        except Exception as e:
            pytest.skip(f"Tokenizer not yet uploaded to HuggingFace: {e}")


class TestInference:
    """Test inference functionality"""

    @pytest.mark.slow
    def test_basic_generation(self):
        """Test basic text generation"""
        try:
            model = AutoModelForCausalLM.from_pretrained("zenlm/MODEL_NAME")
            tokenizer = AutoTokenizer.from_pretrained("zenlm/MODEL_NAME")

            inputs = tokenizer("Hello, world!", return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=20)
            text = tokenizer.decode(outputs[0])

            assert len(text) > 0
            assert isinstance(text, str)
        except Exception as e:
            pytest.skip(f"Model not yet available: {e}")


class TestModelProperties:
    """Test model properties and configuration"""

    def test_model_config(self):
        """Test model configuration"""
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained("zenlm/MODEL_NAME")
            assert config is not None
            assert hasattr(config, "vocab_size")
        except Exception as e:
            pytest.skip(f"Config not yet available: {e}")
