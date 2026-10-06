import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import sys
import types

from sim_eval.backends.episode_budget import EpisodeOutputTokenBudgetBackend
from sim_eval.backends.routed import NamedRoleRoutedBackend
from sim_eval.contracts import ChatMessage, ModelRequest, ModelResponse, TokenUsage
from sim_eval.errors import EpisodeTokenBudgetExceeded
from sim_eval.integrations import token_counting
from sim_eval.interfaces import ModelBackend


class UsageBackend(ModelBackend):
    name = "usage"

    def __init__(self, completion_tokens):
        self.completion_tokens = list(completion_tokens)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        tokens = self.completion_tokens.pop(0)
        return ModelResponse(
            text="ok",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=tokens, total_tokens=10 + tokens),
        )


class NoUsageBackend(ModelBackend):
    name = "no_usage"

    def __init__(self, text="three local tokens"):
        self.text = text
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return ModelResponse(text=self.text)


class FakeTokenCounter:
    def __init__(self, *, prompt_tokens=20, response_tokens=3):
        self.prompt_tokens = prompt_tokens
        self.response_tokens = response_tokens
        self.prompt_kwargs = []

    def count_prompt(self, request, *, chat_template_kwargs=None):
        self.prompt_kwargs.append(dict(chat_template_kwargs or {}))
        return self.prompt_tokens

    def count_response(self, response):
        return self.response_tokens

    def identity(self):
        return {"kind": "fake", "device": "cpu"}


class EpisodeBudgetTests(unittest.TestCase):
    @staticmethod
    def request(*, model="adapter-model", role=None, max_tokens=4096):
        metadata = {"route_role": role} if role else {}
        return ModelRequest(
            messages=(ChatMessage("user", "go"),),
            model=model,
            max_tokens=max_tokens,
            metadata=metadata,
        )

    def test_cumulative_budget_clamps_after_role_overrides_and_then_stops(self):
        target = UsageBackend([4000, 1000])
        router = NamedRoleRoutedBackend(
            roles={"evaluated_model": target},
            default_role="evaluated_model",
            request_overrides={
                "evaluated_model": {"model": "served-model", "max_tokens": 4096}
            },
        )
        budget = EpisodeOutputTokenBudgetBackend(
            router,
            max_output_tokens=5000,
            evaluated_role="evaluated_model",
            evaluated_model="adapter-model",
        )
        budget.generate(self.request(role="evaluated_model"))
        budget.generate(self.request(role="evaluated_model"))
        self.assertEqual([request.max_tokens for request in target.requests], [4096, 1000])
        self.assertEqual(budget.used_output_tokens, 5000)
        self.assertEqual(budget.remaining_output_tokens, 0)
        with self.assertRaises(EpisodeTokenBudgetExceeded):
            budget.generate(self.request(role="evaluated_model"))

    def test_support_role_outputs_do_not_consume_evaluated_model_budget(self):
        target = UsageBackend([7])
        judge = UsageBackend([9000])
        router = NamedRoleRoutedBackend(
            roles={"evaluated_user": target, "judge": judge},
            default_role="evaluated_user",
        )
        budget = EpisodeOutputTokenBudgetBackend(
            router,
            max_output_tokens=100,
            evaluated_role="evaluated_user",
            evaluated_model="target-model",
            support_roles=("judge",),
        )
        budget.generate(self.request(model="judge-model", role="judge"))
        self.assertEqual(budget.used_output_tokens, 0)
        budget.generate(self.request(model="target-model", role="evaluated_user"))
        self.assertEqual(budget.used_output_tokens, 7)
        self.assertEqual(budget.usage_summary()["scope"], "evaluated_model_outputs_only")

    def test_missing_provider_usage_uses_local_tokenizer_instead_of_reservation(self):
        backend = NoUsageBackend()
        counter = FakeTokenCounter(response_tokens=3)
        budget = EpisodeOutputTokenBudgetBackend(
            backend,
            max_output_tokens=100,
            evaluated_role="evaluated_model",
            evaluated_model="adapter-model",
            token_counter=counter,
            model_context_tokens=100,
            chat_template_kwargs={"enable_thinking": False},
        )
        budget.generate(self.request(role="evaluated_model", max_tokens=64))
        self.assertEqual(budget.used_output_tokens, 3)
        self.assertEqual(budget.tokenizer_usage_request_count, 1)
        self.assertEqual(budget.fallback_usage_request_count, 0)
        self.assertEqual(counter.prompt_kwargs, [{"enable_thinking": False}])

    def test_context_and_episode_allowances_are_both_applied_after_role_override(self):
        target = UsageBackend([5])
        router = NamedRoleRoutedBackend(
            roles={"evaluated_model": target},
            default_role="evaluated_model",
            request_overrides={"evaluated_model": {"max_tokens": 4096}},
        )
        budget = EpisodeOutputTokenBudgetBackend(
            router,
            max_output_tokens=50,
            evaluated_role="evaluated_model",
            evaluated_model="adapter-model",
            token_counter=FakeTokenCounter(prompt_tokens=75),
            model_context_tokens=100,
        )
        budget.generate(self.request(role="evaluated_model", max_tokens=40))
        self.assertEqual(target.requests[0].max_tokens, 25)
        self.assertEqual(budget.used_output_tokens, 5)
        self.assertEqual(budget.usage_summary()["last_context_remaining_tokens"], 25)

    def test_less_than_ten_remaining_stops_without_calling_backend(self):
        target = UsageBackend([])
        budget = EpisodeOutputTokenBudgetBackend(
            target,
            max_output_tokens=9,
            evaluated_role="evaluated_model",
            evaluated_model="adapter-model",
        )
        with self.assertRaises(EpisodeTokenBudgetExceeded) as caught:
            budget.generate(self.request(role="evaluated_model"))
        self.assertEqual(caught.exception.scope, "episode_output")
        self.assertEqual(target.requests, [])
        self.assertTrue(budget.usage_summary()["exhausted"])

    def test_less_than_ten_context_tokens_stops_without_calling_backend(self):
        target = UsageBackend([])
        budget = EpisodeOutputTokenBudgetBackend(
            target,
            max_output_tokens=100,
            evaluated_role="evaluated_model",
            evaluated_model="adapter-model",
            token_counter=FakeTokenCounter(prompt_tokens=92),
            model_context_tokens=100,
        )
        with self.assertRaises(EpisodeTokenBudgetExceeded) as caught:
            budget.generate(self.request(role="evaluated_model"))
        self.assertEqual(caught.exception.scope, "model_context")
        self.assertEqual(caught.exception.context_remaining_tokens, 8)
        self.assertEqual(target.requests, [])

    def test_last_returned_response_is_preserved_when_it_consumes_the_budget(self):
        target = UsageBackend([20])
        budget = EpisodeOutputTokenBudgetBackend(
            target,
            max_output_tokens=20,
            evaluated_role="evaluated_model",
            evaluated_model="adapter-model",
        )
        response = budget.generate(self.request(role="evaluated_model", max_tokens=20))
        self.assertEqual(response.text, "ok")
        self.assertEqual(budget.remaining_output_tokens, 0)
        with self.assertRaises(EpisodeTokenBudgetExceeded):
            budget.generate(self.request(role="evaluated_model"))

    def test_provider_overrun_is_audited_but_current_response_is_preserved(self):
        target = UsageBackend([25])
        budget = EpisodeOutputTokenBudgetBackend(
            target,
            max_output_tokens=20,
            evaluated_role="evaluated_model",
            evaluated_model="adapter-model",
        )
        response = budget.generate(self.request(role="evaluated_model", max_tokens=20))
        self.assertEqual(response.text, "ok")
        self.assertEqual(budget.used_output_tokens, 25)
        self.assertEqual(budget.usage_summary()["budget_overrun_tokens"], 5)
        with self.assertRaises(EpisodeTokenBudgetExceeded):
            budget.generate(self.request(role="evaluated_model"))

    def test_shared_tokenizer_factory_initializes_once_under_concurrency(self):
        class FakeTokenizer:
            chat_template = "template"

            def apply_chat_template(self, messages, **kwargs):
                return [1, 2, 3]

            def encode(self, text, add_special_tokens=False):
                return text.split()

        loads = []

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(path, **kwargs):
                loads.append((path, kwargs))
                return FakeTokenizer()

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = FakeAutoTokenizer
        with TemporaryDirectory() as directory, patch.dict(
            sys.modules, {"transformers": fake_transformers}
        ):
            key_path = str(Path(directory))
            key_revision = f"test-revision-{id(loads)}"
            with ThreadPoolExecutor(max_workers=16) as pool:
                counters = tuple(
                    pool.map(
                        lambda _: token_counting.get_shared_huggingface_token_counter(
                            model_path=key_path,
                            model_id="fixture-tokenizer",
                            model_revision=key_revision,
                        ),
                        range(64),
                    )
                )
        self.assertEqual(len(loads), 1)
        self.assertTrue(all(counter is counters[0] for counter in counters))

    def test_chat_token_counter_accepts_batch_encoding_and_forces_sequence_output(self):
        template_kwargs = []

        class FakeTokenizer:
            chat_template = "template"

            def apply_chat_template(self, messages, **kwargs):
                template_kwargs.append(kwargs)
                return {"input_ids": [[1, 2, 3, 4]], "attention_mask": [[1, 1, 1, 1]]}

            def encode(self, text, add_special_tokens=False):
                return {"input_ids": [1, 2]}

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(path, **kwargs):
                return FakeTokenizer()

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = FakeAutoTokenizer
        with TemporaryDirectory() as directory, patch.dict(
            sys.modules, {"transformers": fake_transformers}
        ):
            counter = token_counting.HuggingFaceChatTokenCounter(
                model_path=directory,
                model_id="fixture-tokenizer",
                model_revision="batch-encoding-test-v1",
            )
            request = self.request(role="evaluated_model")
            self.assertEqual(
                counter.count_prompt(
                    request,
                    chat_template_kwargs={"enable_thinking": False, "return_dict": True},
                ),
                4,
            )
            self.assertEqual(counter.count_response(ModelResponse(text="done")), 2)
            separated = ModelResponse(
                text="done",
                raw={"choices": [{"message": {"content": "done", "reasoning_content": "thinking"}}]},
                usage=TokenUsage(completion_tokens=2000),
            )
            self.assertEqual(counter.count_response(separated), 4)
            self.assertEqual(counter.count_content(separated), 2)
            self.assertEqual(counter.count_content(ModelResponse(text="")), 0)


        self.assertFalse(template_kwargs[0]["return_dict"])
        self.assertFalse(template_kwargs[0]["enable_thinking"])


if __name__ == "__main__":
    unittest.main()
