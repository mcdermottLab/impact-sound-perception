import numpy as np
import torch as ch
import scipy.signal as signal

class DownsampleEnvelopes(ch.nn.Module):
    """
    Base class for downsampling operations for the cochlear envelopes.
    """
    def __init__(self, sr, env_sr):
        """
        Base class for downsampling cochlear envelopes.
      
        Args:
            sr (int): the sampling rate for the input signal.
            env_sr (int): the desired sampling rate for the cochlear
                envelopes. 
        """
        super(DownsampleEnvelopes, self).__init__()
        self.sr = sr
        self.env_sr = env_sr
        self.downsample_factor = self.sr/self.env_sr

    def forward(self, x):
        """
        Define the forward pass within each downsample envelope class. 

        Args:
            x (Tensor): the tensor to perform downsampling. Downsampling is 
                performed on the last dimension of this tensor. 

        Returns: 
            (Tensor): the downsampled envelope of the signal. 
        """
        raise NotImplementedError('Forward Pass is not implemented')


class SincWithKaiserWindow(DownsampleEnvelopes):
    """
    Makes a downsampling window that is a sinc function windowed with a kaiser window.
    """
    def __init__(self, sr, env_sr, window_size=1001):
        super(SincWithKaiserWindow, self).__init__(sr, env_sr)
        self.window_size = window_size
        if self.sr % self.env_sr !=0:
            raise ValueError('SincWithKaiserWindow downsampling is only '
                             'supported with integer downsampling factors')
        self.downsample_factor = int(self.downsample_factor)
        downsample_filter_np = self._make_downsample_filter_response()
        self.register_buffer('downsample_filter', ch.from_numpy(downsample_filter_np).float())
        
    def forward(self, x):
        x_shape = x.shape
        if len(x_shape)>2:
            x = x.view(x_shape[0]*x_shape[-2], 1, -1)
        else: # Handle the case where there is no batch dimension
            x = x.view(x_shape[0], 1, -1)
        x = ch.nn.functional.conv1d(x, self.downsample_filter, 
                                    stride=self.downsample_factor)
        return x.view(x_shape[0:-1] + (-1,))

    def _make_downsample_filter_response(self):
        downsample_filter_times = np.arange(-self.window_size/2,
                                            int(self.window_size/2))
        downsample_filter_response_orig = (np.sinc(downsample_filter_times / 
                                                  self.downsample_factor) /
                                              self.downsample_factor)
        downsample_filter_window = signal.kaiser(self.window_size, 5)
        downsample_filter_response = (downsample_filter_window * 
                                         downsample_filter_response_orig)
        downsample_filter_response = np.expand_dims(np.expand_dims(
                                         downsample_filter_response, 0), 0)

        return downsample_filter_response
        
