import numpy as np
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst

array_st = npst.arrays(
    dtype=npst.integer_dtypes(),
    shape=npst.array_shapes(min_dims=2, max_dims=2),
    elements=None
)

#From the idea https://github.com/numpy/numpy/issues/19405

@given(array_st)
def test_unique(ar):
    ar = np.matrix(ar)
    u, inverse, count = np.unique(ar, return_inverse=True, return_counts=True)
    print(ar,u,inverse,count)
    np.testing.assert_array_equal(ar, u[inverse])
    assert len(ar) == np.sum(count)

if __name__ == "__main__":
    test_unique()
