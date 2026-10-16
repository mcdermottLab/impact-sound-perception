import unittest
import sys
sys.path.append('../')
import chcochleagram
from chcochleagram import *
import numpy as np
import torch as ch
import faulthandler
faulthandler.enable()

class CompressionTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self,*args,**kwargs)
        self.all_test_values = np.array([1e-2,1e-3,1e-4,1e-5, 1e-6, 1e-8, 1e-9, 1e-10, 0, -1])
    
    def test_power_compression(self):
        power = 0.3
        offset = 1e-8
        scale = 1
        # For these tests, take the relu but don't calculate it into the gradient
        # A value of -1 should behave the same as a value of 0. 
        expected_grads = power * ((scale * np.maximum(self.all_test_values, 0) + offset)**(power-1))
        expected_power = (scale * np.maximum(self.all_test_values, 0) + offset)**(power)
        for t_idx, test_x in enumerate(self.all_test_values):
            x = ch.autograd.Variable(ch.zeros(1).fill_(test_x), requires_grad=True)
            p = chcochleagram.compression.PowerCompression(scale=scale, offset=offset, power=power)
            y = p(x)
            y.backward()
            x_grad = x.grad.item()
            self.assertTrue(np.allclose(expected_grads[t_idx], x_grad, rtol=1e-8, atol=5), 
                            'Test Value: %f, Expected Grad %f, Measured Grad %f'%(test_x, 
                                                                                  expected_grads[t_idx], 
                                                                                  x_grad))
            print('Test Value: %f, Expected Grad %f, Measured Grad %f'%(test_x, 
                                                                                  expected_grads[t_idx], 
                                                                                  x_grad))
            self.assertTrue(np.allclose(expected_power[t_idx], y.item(), rtol=1e-8, atol=5), 
                            'Test Value: %f'%test_x)
            
    
    def test_clipped_power_compression(self):
        power = 0.3
        offset = 1e-8
        scale = 1
        clip_value = 100
        # For these tests, take the relu but don't calculate it into the gradient
        # A value of -1 should behave the same as a value of 0. 
        expected_grads = np.clip(power * ((scale * np.maximum(self.all_test_values, 0) + offset)**(power-1)), 
                                 -clip_value, clip_value)
        expected_power = (scale * np.maximum(self.all_test_values, 0) + offset)**(power)
        for t_idx, test_x in enumerate(self.all_test_values):
            x =  ch.autograd.Variable(ch.zeros(1).fill_(test_x), requires_grad=True)
            p = chcochleagram.compression.ClippedGradPowerCompression(scale=scale, 
                                                                      offset=offset, 
                                                                      power=power, 
                                                                      clip_value=clip_value)
            y = p(x)
            y.backward()
            x_grad = x.grad.item()
            self.assertTrue(np.allclose(expected_grads[t_idx], x_grad, rtol=1e-8, atol=5), 
                            'Test Value: %f, Expected Grad %f, Measured Grad %f'%(test_x, 
                                                                                  expected_grads[t_idx], 
                                                                                  x_grad))
            print('Test Value: %f, Expected Grad %f, Measured Grad %f'%(test_x, 
                                                                                  expected_grads[t_idx], 
                                                                                  x_grad))
            self.assertTrue(np.allclose(expected_power[t_idx], y.item(), rtol=1e-8, atol=5), 
                            'Test Value: %f'%test_x)


class SubbandsTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self,*args,**kwargs)
    
    def test_subbands_shape_pad_factor_1(self):
        half_cos_filter_kwargs = {
                'n':40,
                'low_lim':50,
                'high_lim':10000,
                'sample_factor':4,
                'include_highpass':False,
                'include_lowpass':False,
                'full_filter':False
            }
        filters = chcochleagram.cochlear_filters.ERBCosFilters(40000,20000, 
                                                               use_rfft=True, 
                                                               filter_kwargs=half_cos_filter_kwargs)
        
        cochleagram = chcochleagram.cochleagram.Cochleagram(filters, None, None)
        x = ch.autograd.Variable(ch.ones([1,40000]), requires_grad=True)
        subband_shape = list(cochleagram.compute_subbands(x).shape)
        self.assertTrue(subband_shape==[1, 171, 20001, 2], 'Expected Subbands '
                         'shape of [1, 171, 20001, 2], got %s'%subband_shape)
        
    def test_subbands_shape_pad_factor_2(self):
        half_cos_filter_kwargs = {
            'n':40,
            'low_lim':50,
            'high_lim':10000,
            'sample_factor':4,
            'include_highpass':False,
            'include_lowpass':False,
            'full_filter':False, 
            
        }
        filters = chcochleagram.cochlear_filters.ERBCosFilters(40000,20000, 
                                                               use_rfft=True, pad_factor=2, 
                                                               filter_kwargs=half_cos_filter_kwargs)
        cochleagram = chcochleagram.cochleagram.Cochleagram(filters, None, None)
        x = ch.autograd.Variable(ch.ones([1,40000]), requires_grad=True)
        subband_shape = list(cochleagram.compute_subbands(x).shape)
        self.assertTrue(subband_shape==[1, 171, 40001, 2], 'Expected Subbands '
                         'shape of [1, 171, 40001, 2], got %s'%subband_shape)


class FFTSEGV(ch.nn.Module):
    def forward(self, input):
        return ch.fft(input, 2)

class SegFaultWithFFTTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self,*args,**kwargs)

    def test_segfault_fft(self):
        model = FFTSEGV()
        tensor = ch.rand(4, 3, 2).cuda()
    
        out = model.cuda(0)(tensor.cuda(0))
    
    
        # This segfaults if cuFFT didn't initialize on cuda:1.
        model = ch.nn.DataParallel(model)
        print(model(tensor))


class FullCochIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self,*args,**kwargs)
        self.signal_size = 40000
        self.sr = 20000

        # Args used for multiple of the cochleagram operations
        signal_size = self.signal_size
        sr = self.sr
        pad_factor = None
        use_rfft = True
        
        # Define cochlear filters
        half_cos_filter_kwargs = {
            'n':40,
            'low_lim':50,
            'high_lim':8000,
            'sample_factor':4,
            'include_highpass':False,
            'include_lowpass':False,
            'full_filter':False, 
        }
        coch_filter_kwargs = {'use_rfft':use_rfft,
                              'pad_factor':pad_factor,
                              'filter_kwargs':half_cos_filter_kwargs}
        
        filters = chcochleagram.cochlear_filters.ERBCosFilters(signal_size,
                                                               sr, 
                                                               **coch_filter_kwargs)
        
        # Define an envelope extraction operation
        envelope_extraction = chcochleagram.envelope_extraction.HilbertEnvelopeExtraction(signal_size,
                                                                                          sr, 
                                                                                          use_rfft, 
                                                                                          pad_factor)
        
        # Define a downsampling operation
        env_sr = 200
        downsampling_kwargs = {'window_size':1001}
        downsampling_op = chcochleagram.downsampling.SincWithKaiserWindow(sr, env_sr, **downsampling_kwargs)
        
        # Define a compression operation
        compression_kwargs = {'power':0.3,
                              'offset':1e-8,
                              'scale':1,
                              'clip_value':100}
        
        compression = chcochleagram.compression.ClippedGradPowerCompression(**compression_kwargs)
        
        cochleagram = chcochleagram.cochleagram.Cochleagram(filters, 
                                                            envelope_extraction,
                                                            downsampling_op,
                                                            compression=compression)


        self.cochleagram = cochleagram

    def test_no_batch_x(self):
        # Build x and test the cochleagram generation with it
        cochleagram = self.cochleagram

        freq = 100
        time_x = np.arange(0, self.signal_size)/20000;
        # Amplitude of the sine wave is sine of a variable like time=
        amplitude = np.sin(2 * np.pi * freq * time_x)

        x = ch.autograd.Variable(ch.Tensor(amplitude), requires_grad=True)
        y = cochleagram(x)

    def test_batch_x(self): 
        cochleagram = self.cochleagram
        freq = 100
        time_x = np.arange(0, self.signal_size)/20000;
        # Amplitude of the sine wave is sine of a variable like time=
        amplitude = np.sin(2 * np.pi * freq * time_x)
        x_batched = ch.autograd.Variable(ch.Tensor(np.stack([amplitude, amplitude, amplitude], axis=0)))
        y = cochleagram(x_batched)

    def test_gpu_cochleagram(self):
        cochleagram = self.cochleagram.cuda()
        freq = 100
        time_x = np.arange(0, self.signal_size)/20000;
        # Amplitude of the sine wave is sine of a variable like time=
        amplitude = np.sin(2 * np.pi * freq * time_x)
        x_batched = ch.autograd.Variable(ch.Tensor(np.stack([amplitude, amplitude, amplitude], axis=0)).cuda())
        y = cochleagram(x_batched)
        print(y.device)
        self.assertTrue(y.device==ch.device("cuda:0"))

    def test_subbands_multi_gpu(self):
        subbands_op = chcochleagram.cochleagram.ComputeSubbands(np.zeros([1,171,int(self.signal_size/2)+1,2]), True, True,
                 None, self.signal_size)
        subbands = ch.nn.DataParallel(subbands_op).cuda()
        freq = 100
        time_x = np.arange(0, self.signal_size)/20000;
        # Amplitude of the sine wave is sine of a variable like time=
        amplitude = np.sin(2 * np.pi * freq * time_x)
        x_batched = ch.autograd.Variable(ch.Tensor(np.stack([amplitude, amplitude, amplitude, amplitude], axis=0)).cuda())
        y = subbands(x_batched)

    def test_multi_gpu_cochleagram(self):
        cochleagram = ch.nn.DataParallel(self.cochleagram).cuda()
        print(cochleagram)
        freq = 100
        time_x = np.arange(0, self.signal_size)/20000;
        # Amplitude of the sine wave is sine of a variable like time=
        amplitude = np.sin(2 * np.pi * freq * time_x)
        x_batched = ch.autograd.Variable(ch.Tensor(np.stack([amplitude, amplitude, amplitude, amplitude], axis=0)).cuda())
        y = cochleagram(x_batched)
        print(y.device)
        self.assertTrue(y.device==ch.device("cuda:0"))


class OuterModule(ch.nn.Module):
    def __init__(self, filts_np):
        super(OuterModule, self).__init__()
        self.filts_np = filts_np
        self.inner_module = InnerModule(self.filts_np)

    def forward(self, x):
        x = self.inner_module(x)

class InnerModule(ch.nn.Module):
    def __init__(self, filts_np):
        super(InnerModule, self).__init__()
# Caching self._mul_internal as self.mul_op places filts_np on the wrong GPU.
        self.register_buffer("filts_np", 
                             ch.from_numpy(filts_np).float())

    def forward(self, x):
# Call self._mul_internal directly in the forward pass.
        x = self._mul_internal(x)
        return x

    def _mul_internal(self, x):
        return self.filts_np * x
   

class DevicePlacementTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self,*args,**kwargs)
        self.numpy_filters = np.ones(10)
        self.outer_module = OuterModule(self.numpy_filters)

    def test_device_placement(self):
        x = np.zeros(10)
        x_batched = ch.autograd.Variable(ch.Tensor(np.stack([x, x, x, x], axis=0)).cuda())
        outer_module = ch.nn.DataParallel(self.outer_module.cuda())
        y = outer_module(x_batched)
        

if __name__ == '__main__':
    unittest.main()
