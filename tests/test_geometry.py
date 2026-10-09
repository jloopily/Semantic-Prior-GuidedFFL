import numpy as np

from src.cirlid.geometry import binarize_mask, fractional_frame_index, middle_frame_index, remap_frame_index


def test_fractional_frame_index_uses_half_up_rounding():
    assert fractional_frame_index(10, 0.25) == 2
    assert fractional_frame_index(10, 0.50) == 5
    assert fractional_frame_index(5, 0.125) == 1  # 0.5 rounds upward


def test_remap_frame_index_is_normalized_and_half_up():
    assert remap_frame_index(0, 10, 5) == 0
    assert remap_frame_index(9, 10, 5) == 4
    assert remap_frame_index(5, 10, 5) == 2


def test_nonzero_mask_policy_preserves_gray_regions():
    mask = np.array([[0, 1], [127, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(binarize_mask(mask, policy="nonzero"), [[0, 1], [1, 1]])
    np.testing.assert_array_equal(binarize_mask(mask, policy="threshold"), [[0, 0], [0, 1]])


def test_middle_frame_index_is_zero_based_and_lower_middle_for_even_counts():
    assert middle_frame_index(1) == 0
    assert middle_frame_index(5) == 2
    assert middle_frame_index(6) == 2
