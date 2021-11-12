import numpy as np
from adam import adam
from im2col import im2col_indices, col2im_indices,col2im_6d


class ConvLayer(object):
    def __init__(self,
                 input_shape=[3,32,32],
                 num_filters = 32,
                 filter_dims = [3,3,3], 
                 stride = 1,
                 padding = 1,
                 weight_scale = 5e-2):
        self.best_params = None
        self.name = "convolution"
        self.num_filters = num_filters 
        self.C_filter = filter_dims[0] 
        self.F_filter = filter_dims[1] 
        self.S_filter = stride         
        self.padding_input = padding  
        self.C_input = input_shape[0]  
        self.N_input = input_shape[1] + 2* padding  
        self.input_shape = input_shape 
        
        if(self.C_filter != self.C_input):
            print ('Error in Channel for the convolutional layer')
 
        self.weights = weight_scale * np.random.randn(self.num_filters,
                                                     self.C_input,
                                                     self.F_filter,
                                                     self.F_filter)
        self.bias = np.zeros(self.num_filters)
        self.Weight_opt = adam(self.weights)
        self.Bias_opt = adam(self.bias)

        self.O = int((self.N_input-self.F_filter)/self.S_filter) + 1 
        self.output_shape = (self.num_filters,self.O,self.O) 
        
    def get_bias(self):
        return self.bias
    
    def save_params(self):
        self.best_params = (self.weights, self.bias)

    def load_best_params(self):
        if self.best_params is not None:
            self.weights, self.bias = self.best_params
  
    def forward(self,x):
        N, C, H, W = x.shape
        w = self.weights
        b = self.bias
        F, _, HH, WW = w.shape
        stride = self.S_filter
        pad = self.padding_input
        
        # Check dimensions
        assert (W + 2 * pad - WW) % stride == 0, 'width does not work'
        assert (H + 2 * pad - HH) % stride == 0, 'height does not work'
        
        #padding the input
        p = pad
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        
        #output dimensions
        H += 2 * pad
        W += 2 * pad
        out_h =  int((H - HH) / stride) + 1
        out_w =  int((W - WW) / stride) + 1
        
  
        shape = (C, HH, WW, N, out_h, out_w)
        strides = (H * W, W, 1, C * H * W, stride * W, stride)
        strides = x.itemsize * np.array(strides)
        x_stride = np.lib.stride_tricks.as_strided(x_padded,
               shape=shape, strides=strides)
        x_cols = np.ascontiguousarray(x_stride)
        x_cols.shape = (C * HH * WW, N * out_h * out_w)
        
        res = w.reshape(F, -1).dot(x_cols) + b.reshape(-1, 1)
        
        # Reshape the output
        res.shape = (F, N, out_h, out_w)
        out = res.transpose(1, 0, 2, 3)
        out = np.ascontiguousarray(out)
        
        self.cache_input = x
        self.x_cols = x_cols
        return out

    def backward(self,dout):
        
        x = self.cache_input
        w = self.weights
        b = self.bias
        x_cols = self.x_cols
        stride = self.S_filter
        pad = self.padding_input
        
        
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        _, _, out_h, out_w = dout.shape
        
        db = np.sum(dout, axis=(0, 2, 3))
        
        dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(F, -1)
        dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)
        
        dx_cols = w.reshape(F, -1).T.dot(dout_reshaped)
        dx_cols.shape = (C, HH, WW, N, out_h, out_w)
        dx = col2im_6d(dx_cols, N, C, H, W, HH, WW, pad, stride)
        
        self.dw = dw
        self.db = db
        
        self.weights = self.Weight_opt.update(self.weights,self.dw)
        self.bias = self.Bias_opt.update(self.bias,self.db)
        
        return dx