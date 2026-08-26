"""Plugin settings: declaration, binding, and the errors that make config legible."""
import pytest

from core.plugins import (
    REDACTED,
    Setting,
    SettingsError,
    bind,
    cross_plugin_hint,
    describe,
    env_key,
    redact,
)

SPECS = (
    Setting("model", str, required=True),
    Setting("base_url", str, default="http://localhost:1234/v1"),
    Setting("max_tokens", int, default=8000),
    Setting("no_think", bool, default=False),
    Setting("api_key", str, secret=True),
)


class TestBinding:
    def test_defaults_apply(self):
        got = bind("openai", SPECS, {"model": "m"}, env={})
        assert got["base_url"] == "http://localhost:1234/v1"
        assert got["max_tokens"] == 8000
        assert got["no_think"] is False

    def test_manifest_overrides_default(self):
        got = bind("openai", SPECS, {"model": "m", "max_tokens": 500}, env={})
        assert got["max_tokens"] == 500

    def test_namespaced_env_overrides_manifest(self):
        got = bind("openai", SPECS, {"model": "from-file"},
                   env={"DI_OPENAI_MODEL": "from-env"})
        assert got["model"] == "from-env"

    def test_explicit_override_beats_everything(self):
        got = bind("openai", SPECS, {"model": "from-file"},
                   env={"DI_OPENAI_MODEL": "from-env"}, overrides={"model": "from-flag"})
        assert got["model"] == "from-flag"

    def test_env_reference_resolves_from_environment(self):
        got = bind("openai", SPECS, {"model": "m", "api_key": "${MY_TOKEN}"},
                   env={"MY_TOKEN": "secret-value"})
        assert got["api_key"] == "secret-value"

    def test_unset_env_reference_becomes_none_not_the_literal(self):
        """A manifest committed with ${TOKEN} and no token set must not send '${TOKEN}'."""
        got = bind("openai", SPECS, {"model": "m", "api_key": "${MISSING}"}, env={})
        assert got["api_key"] is None

    def test_env_key_naming(self):
        assert env_key("openai", "model") == "DI_OPENAI_MODEL"
        assert env_key("anthropic", "max_tokens") == "DI_ANTHROPIC_MAX_TOKENS"


class TestTypes:
    @pytest.mark.parametrize("value,expected", [
        ("true", True), ("1", True), ("yes", True), ("on", True),
        ("false", False), ("0", False), ("", False),
    ])
    def test_bool_coercion(self, value, expected):
        assert bind("openai", SPECS, {"model": "m", "no_think": value}, env={})["no_think"] is expected

    def test_int_coercion_from_string(self):
        """Environment variables arrive as strings; settings still have types."""
        got = bind("openai", SPECS, {"model": "m"}, env={"DI_OPENAI_MAX_TOKENS": "1234"})
        assert got["max_tokens"] == 1234

    def test_bad_number_is_an_error_naming_the_setting(self):
        with pytest.raises(SettingsError, match="max_tokens"):
            bind("openai", SPECS, {"model": "m", "max_tokens": "lots"}, env={})

    def test_bad_bool_is_an_error(self):
        with pytest.raises(SettingsError, match="no_think"):
            bind("openai", SPECS, {"model": "m", "no_think": "maybe"}, env={})


class TestErrors:
    def test_unknown_setting_is_rejected(self):
        """The whole point: a setting that is silently ignored is the bug."""
        with pytest.raises(SettingsError) as caught:
            bind("openai", SPECS, {"model": "m", "effort": "high"}, env={})
        assert "no setting 'effort'" in str(caught.value)
        assert "it accepts:" in str(caught.value)

    def test_typo_gets_a_suggestion(self):
        with pytest.raises(SettingsError, match="Did you mean 'max_tokens'"):
            bind("openai", SPECS, {"model": "m", "max_token": 10}, env={})

    def test_missing_required_names_the_plugin_and_the_variable(self):
        with pytest.raises(SettingsError) as caught:
            bind("openai", SPECS, {}, env={})
        message = str(caught.value)
        assert "'openai' requires 'model'" in message
        assert "DI_OPENAI_MODEL" in message

    def test_cross_plugin_hint_identifies_the_right_owner(self):
        """`effort` is a real setting -- just not on this plugin. Say which."""
        others = {"anthropic": (Setting("effort", str),)}
        assert cross_plugin_hint("effort", others) == \
            "'effort' is a setting of the 'anthropic' plugin, not this one"
        assert cross_plugin_hint("nonsense", others) is None


class TestSecrets:
    def test_secrets_are_redacted_for_provenance(self):
        got = bind("openai", SPECS, {"model": "m", "api_key": "${T}"}, env={"T": "shhh"})
        safe = redact(got, SPECS)
        assert safe["api_key"] == REDACTED
        assert safe["model"] == "m"
        assert "shhh" not in str(safe)

    def test_unset_secret_is_not_redacted_into_looking_set(self):
        got = bind("openai", SPECS, {"model": "m"}, env={})
        assert redact(got, SPECS)["api_key"] is None


class TestDescribe:
    def test_describe_marks_required_and_secret(self):
        text = describe("openai", SPECS)
        assert "[openai]" in text
        assert "required" in text
        assert "secret" in text


class TestEndpointCheck:
    """`config --check` asks the endpoint whether it serves the model we named.

    The manifest pinned qwen/qwen3.5-9b -- a model the server had never heard of --
    and a base_url pointing at a workstation, and it went unnoticed because every run
    overrode both from the environment. It worked perfectly for the one person who
    knew the right values and failed on document one for everyone else. Nothing
    offline can catch that, which is the whole reason for a live check.
    """

    def config_naming(self, model):
        from core.config import Config
        return Config(path="test.toml", pipeline={"extractor": "openai"},
                      blocks={"extractor": {"openai": {
                          "model": model, "base_url": "http://x/v1", "api_key": "k"}}})

    def check(self, monkeypatch, model, available):
        from extract import cli
        from extract import backends

        class Backend:
            def __init__(self):
                self.model = model

            def available_models(self):
                return available

        monkeypatch.setattr(backends, "build", lambda **kw: Backend())
        return cli._check_endpoint(self.config_naming(model))

    def test_a_served_model_passes(self, monkeypatch):
        assert self.check(monkeypatch, "qwen/qwen3-vl-8b",
                          ["qwen/qwen3-vl-8b", "other"]) == 0

    def test_a_model_the_endpoint_lacks_fails(self, monkeypatch):
        assert self.check(monkeypatch, "qwen/qwen3.5-9b",
                          ["qwen/qwen3-vl-8b", "qwen/qwen3.8-27b"]) == 1

    def test_an_unreachable_endpoint_fails_rather_than_passing_quietly(self, monkeypatch):
        """An empty list means we learned nothing, which is not the same as ok."""
        assert self.check(monkeypatch, "qwen/qwen3-vl-8b", []) == 1

    def test_a_backend_that_cannot_be_built_fails(self, monkeypatch):
        from extract import cli
        from extract import backends

        def boom(**kw):
            raise ValueError("no api key")

        monkeypatch.setattr(backends, "build", boom)
        assert cli._check_endpoint(self.config_naming("m")) == 1
