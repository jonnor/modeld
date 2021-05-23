
import warnings
warnings.filterwarnings('ignore') # Ignore all the warning messages in this tutorial

import onnx
#from onnx_tf.backend import prepare

import onnxruntime as rt

model_path = 'models/supercombo.converted.onnx'

#model = onnx.load(model_path) # Load the ONNX file
#engine = prepare(model) # Import the ONNX model to Tensorflow

sess = rt.InferenceSession(model_path)

inputs = sess.get_inputs()
for i in inputs:
    print(i)

#out = engine.run(doggy_y)
#print(out)
