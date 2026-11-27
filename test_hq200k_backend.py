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

    def test_rate_encoder_preserves_fractional_phase(self):
        b=HQ200K(stimulus_mode="rate",plasticity=False)
        # Instrument only pulse delivery, not neural output.
        b._tick=lambda injection: np.zeros(64,dtype=np.int64)
        for _ in range(10):
            b.stimulate([4.,40.,0.,10.,20.,30.,5.,15.],[2.]*8,GROUPS,50,50)
        self.assertEqual(b.input_pulses.tolist(),[4,40,0,10,20,30,5,15])
        self.assertEqual(b.steps,10)
        self.assertEqual(b.metadata()["stimulus_mode"],"rate")

    def test_reference_single_pulse_unchanged(self):
        b=HQ200K(plasticity=False)
        b._tick=lambda injection: np.zeros(64,dtype=np.int64)
        b.stimulate([4.]*8,[2.]*8,GROUPS,50,50)
        b.stimulate([40.]*8,[2.]*8,GROUPS,50,50)
        self.assertEqual(b.input_pulses.tolist(),[2]*8)

    def test_adaptation_rises_and_decays(self):
        b=HQ200K(plasticity=False,adaptation_strength=0.5)
        cells=b.electrodes[8]
        b._tick({8:2.})
        np.testing.assert_allclose(b.adaptation[cells],0.5)
        b._tick({})
        self.assertTrue(np.all(b.adaptation[cells]<0.5))
        self.assertTrue(np.all(b.adaptation[cells]>0))
        self.assertEqual(b.metadata()["adaptation_strength"],0.5)

    def test_budgeted_feedback_coalesces_and_bounds_updates(self):
        b=HQ200K(plasticity_mode="budgeted",plasticity_budget=.002)
        rows=np.arange(100,dtype=np.int64)
        slots=np.zeros(100,dtype=np.int64)
        old=b.w[rows,slots].copy()
        for _ in range(10):
            b.eligibility.append((rows,slots))
        b.feedback("reward",[19,20,22],20,2,30,False,"test")
        new=b.w[rows,slots]
        self.assertEqual(b.updates,100)
        self.assertEqual(b.plasticity_events,1)
        self.assertGreater(b.plasticity_l1,0.)
        self.assertLessEqual(b.plasticity_l1,.00201)
        np.testing.assert_array_equal(np.sign(new),np.sign(old))
        b.plasticity=False
        b.feedback("reward",[23,24,26],20,2,30,False,"test")
        np.testing.assert_array_equal(b.w[rows,slots],new)

    def test_electrode_geometry_matches_calibration(self):
        from benchmarks.hq200k_geometry_validation import patch_electrodes
        import hashlib
        reference=HQ200K(plasticity=False)
        for patches in (1,2,4):
            b=HQ200K(plasticity=False,electrode_patches=patches)
            np.testing.assert_array_equal(b.electrodes,patch_electrodes(b.seed,patches))
            self.assertEqual(np.unique(b.electrodes).size,2048)
            self.assertEqual(b.initial_hash,reference.initial_hash)
            self.assertEqual(b.metadata()["electrode_patches"],patches)
            self.assertEqual(b.metadata()["electrode_sha256"],hashlib.sha256(b.electrodes.astype("<i8").tobytes()).hexdigest())
            for i,(lo,hi) in enumerate(zip(BOUNDS[:-1],BOUNDS[1:])):
                mask=(b.electrodes>=lo)&(b.electrodes<hi)
                self.assertTrue(np.all(mask.sum(axis=1)==(5 if i<4 else 4)))
                self.assertEqual(np.unique((b.electrodes[mask]-lo)//1024).size,patches)
            # The geometry RNG must not affect the initial spontaneous-noise sequence.
            np.testing.assert_array_equal(b.rng.integers(N,size=64),HQ200K(plasticity=False).rng.integers(N,size=64))
        with self.assertRaises(ValueError):
            HQ200K(electrode_patches=3)

if __name__=="__main__":
    unittest.main()
