import pytest

from python_basics.learn_pytest.source.shapes import Rectangle


@pytest.fixture
def my_rectangle():
    return Rectangle(10, 20)


@pytest.fixture
def another_rectangle():
    return Rectangle(30, 20)
