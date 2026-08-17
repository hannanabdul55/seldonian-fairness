"""Black-box tests for seldonian.parser and the seldonian.algorithm base classes."""

import numpy as np
import pytest

from seldonian.algorithm import Model, SeldonianAlgorithm
from seldonian.parser import Stack, parse_ghat


class TestStack:
    def test_push_peek_pop(self):
        s = Stack()
        s.push(1)
        s.push(2)
        assert s.peek() == 2
        assert s.pop() == 2
        assert s.pop() == 1

    def test_pop_empty_returns_none(self):
        assert Stack().pop() is None

    def test_size_tracks_contents(self):
        s = Stack()
        assert s.size() == 0
        s.push("a")
        s.push("b")
        assert s.size() == 2
        s.pop()
        assert s.size() == 1

    def test_data_exposes_items(self):
        s = Stack()
        s.push(1)
        s.push(2)
        assert s.data() == [1, 2]


class TestParseGhat:
    def test_unfinished_parser_raises(self):
        with pytest.raises(NotImplementedError):
            parse_ghat("a + b", vars={"a": 1, "b": 2})


class TestSeldonianAlgorithmABC:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            SeldonianAlgorithm()

    def test_safety_test_wrapper_semantics(self):
        class Passing(SeldonianAlgorithm):
            def fit(self, **kwargs):
                pass

            def predict(self, X):
                pass

            def data(self):
                pass

            def _safetyTest(self, **kwargs):
                return 0.0

        class Failing(Passing):
            def _safetyTest(self, **kwargs):
                return 0.3

        assert Passing().safetyTest() is True
        assert Failing().safetyTest() is False


class TestModelBase:
    @pytest.mark.parametrize("method,args", [
        ("data", ()),
        ("fit", (np.zeros((2, 2)), np.zeros(2))),
        ("predict", (np.zeros((2, 2)),)),
        ("parameters", ()),
        ("reset", ()),
    ])
    def test_all_methods_must_be_overridden(self, method, args):
        with pytest.raises(NotImplementedError):
            getattr(Model(), method)(*args)
