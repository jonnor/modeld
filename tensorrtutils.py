import tensorrt as trt
import numpy as np
import os

import pycuda.driver as cuda
import pycuda.autoinit


# From https://stackoverflow.com/questions/59280745/inference-with-tensorrt-engine-file-on-python

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TrtModel:
    
    def __init__(self,engine_path,max_batch_size=1, dtype=np.float32):
        
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

                
                
    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    def allocate_buffers(self):
        
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        # defined in common.tools.lib.parser
        self.proper_output_order = {
            "path": 0,
            "left_lane": 1,
            "right_lane": 2,
            "lead": 3,
            "long_x": 4,
            "long_v": 5,
            "long_a": 6,
            "snpe_pleaser2": 7, # desire_state ???
            "meta": 8,
            "pose": 9, # desire_pred ???
            "add_3": 10,
        }
        self.output_index_mapping = {}
        output_number = 0

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                proper = self.proper_output_order[str(binding)]
                self.output_index_mapping[output_number] = proper
                output_number += 1
                outputs.append(HostDeviceMem(host_mem, device_mem))

        reordered = [ None ] * len(outputs)
        for idx, out in enumerate(outputs):
            correct = self.output_index_mapping[idx]
            reordered[correct] = out 
        outputs = reordered

        return inputs, outputs, bindings, stream
       
            
    def __call__(self, images:np.ndarray, desire: np.ndarray, state: np.ndarray, batch_size=1):
        
        images = images.astype(self.dtype)
        desire = desire.astype(self.dtype)
        state = state.astype(self.dtype)

        np.copyto(self.inputs[0].host, images.ravel())
        np.copyto(self.inputs[1].host, desire.ravel())
        np.copyto(self.inputs[2].host, state.ravel())
        
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
 
        self.stream.synchronize()

        outs = [out.host.reshape(batch_size,-1) for out in self.outputs]

        return outs

