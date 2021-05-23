from common.transformations.camera import transform_img, eon_intrinsics
from common.transformations.model import medmodel_intrinsics
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from common.lanes_image_space import transform_points
import os
from common.tools.lib.parser import parser
import cv2
import sys
import time

#matplotlib.use('Agg')
#camerafile = sys.argv[1]
camerafile = 'project_video.mp4'

class KerasModel():
    def __init__(self):
        self.model = None

    def load(self):
        #from tensorflow.keras.models import load_model

        self.model = load_model('models/supercombo.keras')

    def run(self, inputs):
        return self.model.predict(inputs)


class OnnxCPUModel():
    def __init__(self):
        self.session = None
        self.outputs_names = None
        self.input_names = [ ]

    def load(self):
        import onnx
        import onnxruntime as rt

        self.session = rt.InferenceSession('models/supercombo.converted.onnx')

        self.output_names = []
        for p in self.session.get_outputs():
            print('output', p)
            self.output_names.append(str(p.name))

        for p in self.session.get_inputs():
            print('input', p)

    def run(self, inputs):
        #print(type(inputs))
        #print(
        images, desire, state = inputs

        named_inputs = {
            'vision_input_imgs': images.astype(np.float32),
            'desire': desire.astype(np.float32),
            'rnn_state': state.astype(np.float32),
        }

        return self.session.run(self.output_names, named_inputs)

# XXX: untested
class OnnxTensorRTModel():
    def __init__(self):
        self.session = None
        self.outputs_names = None
        self.input_names = [ ]

    def load(self):
        import onnx
        import onnx_tensorrt.backend as backend

        self.model = onnx.load('models/supercombo.converted.onnx')
        self.engine = backend.prepare(model, device='CUDA:1')

        self.output_names = []
        for p in self.session.get_outputs():
            print('output', p)
            self.output_names.append(str(p.name))

        for p in self.session.get_inputs():
            print('input', p)

    def run(self, inputs):
        #print(type(inputs))
        #print(
        images, desire, state = inputs

        named_inputs = {
            'vision_input_imgs': images.astype(np.float32),
            'desire': desire.astype(np.float32),
            'rnn_state': state.astype(np.float32),
        }

        return self.engine.run(self.output_names, named_inputs)


class OnnxTensorRTEngine():
    def __init__(self):
        self.session = None
        self.outputs_names = None
        self.input_names = [ ]

    def load(self):
        import tensorrtutils
        self.model = tensorrtutils.TrtModel("models/supercombo.fp16.trt")

    def run(self, inputs):
        images, desire, state = inputs
        outputs = self.model(images, desire, state)

        return outputs 


#model = KerasModel()
#model = OnnxCPUModel() 
#model = OnnxTensorRTModel() 
model = OnnxTensorRTEngine()
model.load()


MAX_DISTANCE = 140.
LANE_OFFSET = 1.8
MAX_REL_V = 10.

LEAD_X_SCALE = 10
LEAD_Y_SCALE = 10

def frames_to_tensor(frames):                                                                                               
  H = (frames.shape[1]*2)//3                                                                                                
  W = frames.shape[2]                                                                                                       
  in_img1 = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.uint8)                                                      
                                                                                                                            
  in_img1[:, 0] = frames[:, 0:H:2, 0::2]                                                                                    
  in_img1[:, 1] = frames[:, 1:H:2, 0::2]                                                                                    
  in_img1[:, 2] = frames[:, 0:H:2, 1::2]                                                                                    
  in_img1[:, 3] = frames[:, 1:H:2, 1::2]                                                                                    
  in_img1[:, 4] = frames[:, H:H+H//4].reshape((-1, H//2,W//2))                                                              
  in_img1[:, 5] = frames[:, H+H//4:H+H//2].reshape((-1, H//2,W//2))
  return in_img1

imgs_med_model = np.zeros((2, 384, 512), dtype=np.uint8)
state = np.zeros((1,512))
desire = np.zeros((1,8))

cap = cv2.VideoCapture(camerafile)

x_left = x_right = x_path = np.linspace(0, 192, 192)
(ret, previous_frame) = cap.read()
if not ret:
   exit()
else:
  img_yuv = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2YUV_I420)
  imgs_med_model[0] = transform_img(img_yuv, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                    output_size=(512,256))

last_time = time.time()

while True:
  read_start = time.time()
  (ret, current_frame) = cap.read()
  if not ret:
       break
  read_end = time.time()

  preproc_start = time.time()
  frame = current_frame.copy()
  img_yuv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2YUV_I420)
  imgs_med_model[1] = transform_img(img_yuv, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                    output_size=(512,256))
  frame_tensors = frames_to_tensor(np.array(imgs_med_model)).astype(np.float32)/128.0 - 1.0

  inputs = [np.vstack(frame_tensors[0:2])[None], desire, state]
  preproc_end = time.time()

  predict_start = time.time()
  outs = model.run(inputs)
  parsed = parser(outs)
  predict_end = time.time()

  # Important to refeed the state
  state = outs[-1]
  pose = outs[-2]   # For 6 DoF Callibration

  visualize = False

  visualize_start = time.time()
  if visualize:
      plt.clf()
      plt.title("lanes and path")
      plt.xlim(0, 1200)
      plt.ylim(800, 0)

      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      plt.imshow(frame)
      #print("frame", frame.shape, parsed)
      
      new_x_left, new_y_left = transform_points(x_left, parsed["lll"][0])
      new_x_right, new_y_right = transform_points(x_left, parsed["rll"][0])
      new_x_path, new_y_path = transform_points(x_left, parsed["path"][0])
      plt.plot(new_x_left, new_y_left, label='transformed', color='w')
      plt.plot(new_x_right, new_y_right, label='transformed', color='w')
      plt.plot(new_x_path, new_y_path, label='transformed', color='green')
      imgs_med_model[0]=imgs_med_model[1]
      plt.pause(0.001)
  visualize_end = time.time()

  if visualize:
      if cv2.waitKey(10) & 0xFF == ord('q'):
            break

  frame_dur = time.time() - last_time
  last_time = time.time()

  print(1/frame_dur, read_end-read_start, visualize_end-visualize_start, predict_end-predict_start, preproc_end-preproc_start)



#plt.show()
  


