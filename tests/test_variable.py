from model_operator.variable import Variable
import unittest
import numpy as np

class TestVariable(unittest.TestCase):
    def test_init_fail(self):
        # Tests for failure with unexpected transform or None
        with self.assertRaises(ValueError):
            Variable(name='test', transform='')
        with self.assertRaises(ValueError):
            Variable(name='test', transform=None)

        # Tests for failure with name='custom' but without scale or offset
        with self.assertRaises(ValueError):
            Variable(name='test', transform='custom')
        with self.assertRaises(ValueError):
            Variable(name='test', transform='custom', scale=0.1)
        with self.assertRaises(ValueError):
            Variable(name='test', transform='custom', offset=0.1)
    
    def test_ignore(self):
        # for transrofm == percent or transform == identiry, scale and offset are automatically assigned.
        var = Variable(name='test', transform='percent', scale=10, offset=-2, _fit=True)
        self.assertEqual(var.scale, 100)
        self.assertEqual(var.offset, 50)
        var = Variable(name='test', transform='identity', scale=10, offset=-2, _fit=True)
        self.assertEqual(var.scale, 1)
        self.assertEqual(var.offset, 0)

        # for normalize, it depends on _fit
        # if _fit == True
        var = Variable(name='test', transform='normalize', scale=10, offset=-2, _fit=True)
        self.assertEqual(var.scale, 10)
        self.assertEqual(var.offset, -2)
        # if _fit == False, refulse the scale and offset
        var = Variable(name='test', transform='normalize', scale=10, offset=-2)
        self.assertEqual(var.scale, None)
        self.assertEqual(var.offset, None)

        # if custom, scale and offset have to be given
        var = Variable(name='test', transform='custom', scale=12.34, offset=5.67)
        self.assertEqual(var.scale, 12.34)
        self.assertEqual(var.offset, 5.67)
        

    def test_fit(self):
        random_values = np.random.random(100)
        # Tests for transform='normalize'
        var = Variable(name='test', transform='normalize')
        var.fit(random_values)
        self.assertEqual(var.scale, np.std(random_values))
        self.assertEqual(var.offset, np.mean(random_values))
        # Tests for transform='percent'
        var = Variable(name='test', transform='percent')
        var.fit(random_values)
        self.assertEqual(var.scale, 100)
        self.assertEqual(var.offset, 50)
        # Tests for transform='identiry'
        var = Variable(name='test', transform='identity')
        var.fit(random_values)
        self.assertEqual(var.scale, 1.)
        self.assertEqual(var.offset, 0.)
        # Tests for transform='custom'
        var = Variable(name='test', transform='custom', scale=0.12345, offset=11.234)
        var.fit(random_values)
        self.assertEqual(var.scale, 0.12345)
        self.assertEqual(var.offset, 11.234)

if __name__ == '__main__':
    unittest.main()
