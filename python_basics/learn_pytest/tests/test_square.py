import pytest

from python_basics.learn_pytest.source.shapes import Square


@pytest.mark.parametrize("side_length, expected_area", [(10, 100), (20, 400), (50, 2500), (39, 1521)])
def test_multiple_square_areas(side_length, expected_area):
    assert Square(side_length).area() == expected_area


@pytest.mark.parametrize("side_length, expected_perimeter", [(3, 12), (4, 16)])
def test_multiple_square_perimeters(side_length, expected_perimeter):
    assert Square(side_length).perimeter() == expected_perimeter
