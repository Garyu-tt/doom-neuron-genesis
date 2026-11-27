import unittest
import numpy as np
from hq200k_backend import HQ200K, N, BOUNDS, ROUTES, build_graph, graph_hash
from genesis_neural_interface import GROUPS, handle_stim, validate_metadata
import udp_protocol as wire

class HQTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = HQ200K()

    def test_topology(self):
        b=self.b
        validate_metadata(b.metadata())
        self.assertEqual(b.dst.shape,(N,32))
        self.assertEqual(b.w.shape,(N,32))
        self.assertEqual(len(b.v),N)
        self.assertEqual(len(np.unique(b.electrodes)),64*32)
        for r,(lo,hi) in enumerate(zip(BOUNDS[:-1],BOUNDS[1:])):
            d=b.dst[lo:hi]
            self.assertTrue(np.all((d[:,:24]>=lo)&(d[:,:24]<hi)))
            t=ROUTES[r]
            self.assertTrue(np.all((d[:,24:]>=BOUNDS[t])&(d[:,24:]<BOUNDS[t+1])))
            self.assertTrue(np.all(np.sign(b.w[lo:hi])==np.sign(b.w[lo:hi,:1])))
        dst,w=build_graph()
        self.assertEqual(graph_hash(dst,w),b.initial_hash)
        del dst,w
        dst,w=build_graph(shuffled=True)
        self.assertTrue(np.array_equal(np.bincount(dst.ravel(),minlength=N),np.bincount(b.dst.ravel(),minlength=N)))

    def test_udp_and_virtual_time(self):
        b=self.b
        t=b.ticks
        packet=wire.pack_stimulation_command(np.full(8,20,np.float32),np.full(8,2,np.float32))
        _,counts=wire.unpack_spike_data(handle_stim(b,packet,50,50))
        self.assertEqual(b.ticks-t,10)
        self.assertEqual(counts.shape,(8,))
        self.assertTrue(np.all(counts>=0))
        self.assertGreater(b.spikes,0)

    def test_local_feedback(self):
        b=self.b
        b.eligibility.clear()
        rows=np.array([0,1]); slots=np.array([0,1])
        old=b.w[rows,slots].copy()
        try:
            b.eligibility.append((rows,slots))
            b.feedback("reward",[19,20,22],20,2,30,False,"test")
            self.assertTrue(np.all(np.abs(b.w[rows,slots])>np.abs(old)))
            self.assertTrue(np.array_equal(np.sign(old),np.sign(b.w[rows,slots])))
            b.plasticity=False
            before=b.w[rows,slots].copy()
            b.feedback("reward",[23,24,26],60,2,90,True,"test")
            np.testing.assert_array_equal(b.w[rows,slots],before)
            b.feedback("interrupt",[19,20,22],0,0,0,False,"")
            self.assertNotIn(19,b.feedback_queue)
        finally:
            b.w[rows,slots]=old
            b.plasticity=True
            b.feedback_queue.clear()
            b.eligibility.clear()

if __name__=="__main__":
    unittest.main()
