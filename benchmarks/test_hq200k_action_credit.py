import unittest
from types import SimpleNamespace
import numpy as np
from benchmarks.hq200k_action_credit import select_edges,update

class CreditTest(unittest.TestCase):
    def test_backward_trace_excludes_unrelated_active_path(self):
        dst=np.array([[1],[2],[2],[4],[4]],dtype=np.int32)
        traces=[(np.array([0,3]),np.array([0,0])),
                (np.array([1,4]),np.array([0,0]))]
        ids,strength=select_edges(dst,traces,np.array([2]),"traced")
        np.testing.assert_array_equal(ids,[0,1])
        np.testing.assert_allclose(strength,[.0008,.001])
        ids,_=select_edges(dst,traces,np.array([2]),"global")
        np.testing.assert_array_equal(ids,[0,1,3,4])

    def test_signed_update_preserves_polarity_and_budget(self):
        b=SimpleNamespace(w=np.full((2,32),.1,np.float32))
        b.w[1,0]=-.1
        old=b.w.copy()
        delta=update(b,np.array([0,32]),np.array([.001,.001]),1,budget=.001)
        self.assertLessEqual(delta,.001001)
        self.assertGreater(b.w[0,0],old[0,0])
        self.assertGreater(b.w[1,0],old[1,0])
        np.testing.assert_array_equal(np.sign(b.w),np.sign(old))
        np.testing.assert_array_equal(b.w[:,1:],old[:,1:])

if __name__=="__main__":
    unittest.main()
