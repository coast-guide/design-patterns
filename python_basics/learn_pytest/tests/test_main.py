import pytest

from python_basics.learn_pytest.source.main import add, divide


def test_add():
    result = add(number1=1, number2=4)
    assert result == 5


def test_add_strings():
    result = add("I like ", "burgers")
    assert result == "I like burgers"


def test_divide():
    result = divide(number1=10, number2=5)
    assert result == 2


def test_divide_by_zero():
    with pytest.raises(ValueError):
        divide(number1=10, number2=0)


@pytest.mark.skip(reason="This feature is currently broken")
def test_add():
    assert add(10, -8) == 2


@pytest.mark.xfail(reason="We know we cannot divide by zero")
def test_divide_by_zero_sample():
    assert divide(90, 0)
