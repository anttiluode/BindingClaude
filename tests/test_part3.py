"""Small invariant tests; not a replacement for registered ten-seed runs."""
import unittest
import numpy as np
from part3 import Binder, world, accuracy
from part3_controls import visible, lookup, hybrid_features

class TestPart3(unittest.TestCase):
    def test_true_world_rules_and_noiseless_one_hot(self):
        r=np.random.default_rng(21)
        z,x,h,ctx=world('chain',250,r)
        self.assertTrue(np.all(x[:,4]==x[:,0]*x[:,1]))
        self.assertTrue(np.all(x[:,5]==x[:,4]*x[:,2]))
        self.assertTrue(np.all(x[:,6]==x[:,5]*x[:,3]))
        z,x,h,ctx=world('context',250,r,flip=True)
        self.assertTrue(np.array_equal(z[:,4:6],x[:,4:6]))
        self.assertTrue(np.all(x[ctx==0,6]==-x[ctx==0,0]*x[ctx==0,1]))
        self.assertTrue(np.all(x[ctx==1,6]==x[ctx==1,2]*x[ctx==1,3]))

    def test_no_hidden_peek_and_sequential_chain(self):
        r=np.random.default_rng(3)
        net=Binder(13,r,m=3)
        net.idx[:]=np.array([[0,1,4,-1],[4,2,5,-1],[5,3,6,-1]])
        net.cons[:]=True; net.s1[:]=200; net.age[:]=200
        z,x,hidden,_=world('chain',120,r)
        shot,_=net.settle(z,hidden,mode='one_shot')
        settled,_=net.settle(z,hidden)
        self.assertTrue(np.all(shot[:,6]==0))
        self.assertGreater(accuracy(settled[:,6],x[:,6])['acc'],.95)
        z2=z.copy(); z2[:,list(hidden)]=100000.
        self.assertTrue(np.array_equal(settled,net.settle(z2,hidden)[0]))

    def test_zero_ties_abstain(self):
        net=Binder(13,np.random.default_rng(1),m=1)
        z,x,hidden,_=world('chain',20,np.random.default_rng(2))
        filled,_=net.settle(z,hidden)
        self.assertTrue(np.all(filled[:,list(hidden)]==0))
        self.assertEqual(accuracy(filled[:,6],x[:,6])['acc'],.5)

    def test_context_knockout(self):
        r=np.random.default_rng(22)
        net=Binder(11,r,m=2)
        net.idx[:]=np.array([[4,0,1,6],[5,2,3,6]])
        net.cons[:]=True; net.s1[:]=200; net.age[:]=200
        z,x,hidden,ctx=world('context',600,r)
        p,_=net.settle(z,hidden)
        self.assertGreater(accuracy(p[:,6],x[:,6])['acc'],.98)
        knocked,_=net.settle(z,hidden,silence=net.containing(0,1,6))
        self.assertEqual(accuracy(knocked[ctx==0,6],x[ctx==0,6])['acc'],.5)
        self.assertGreater(accuracy(knocked[ctx==1,6],x[ctx==1,6])['acc'],.98)

    def test_controls_use_visible_only(self):
        r=np.random.default_rng(40)
        z,x,hidden,_=world('context',100,r)
        v=visible(z,hidden)
        zz=z.copy();zz[:,6]*=-1
        self.assertTrue(np.array_equal(v,visible(zz,hidden)))
        self.assertTrue(np.array_equal(lookup(v,x[:,6],v),lookup(visible(zz,hidden),x[:,6],v)))
        net=Binder(11,np.random.default_rng(3),m=3)
        self.assertTrue(np.array_equal(hybrid_features(net,z,hidden),hybrid_features(net,zz,hidden)))

if __name__=='__main__': unittest.main()
