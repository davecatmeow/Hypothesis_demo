import numpy as np
import scipy.linalg
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst


array_st = npst.arrays(
    dtype=float,
    shape=npst.array_shapes(min_dims=2, max_dims=2).filter(lambda s: s[0] == s[1]),
    elements=st.floats(allow_nan=False, allow_infinity=False)
)

@given(array_st)
def test_eigval(arr):
    w, v = np.linalg.eig(arr)
    np.testing.assert_allclose(arr @ v, w * v, atol=10e-7)



test_eigval()
