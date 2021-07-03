import numpy as np
import cv2 as cv2
import os
import time
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import pycuda.autoinit


img_width, img_height = (224, 224)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)
def build_engine(onnx_path, shape = [1,img_width,img_height,3]):
   with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
       builder.max_workspace_size = (256 << 20)
       with open(onnx_path, 'rb') as model:
           parser.parse(model.read())
       network.get_input(0).shape = shape
       engine = builder.build_cuda_engine(network)
       return engine


def allocate_buffers(engine, batch_size=1, data_type=trt.float32):
   h_input = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(data_type))
   h_output = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(data_type))

   d_input = cuda.mem_alloc(h_input.nbytes)
   d_output = cuda.mem_alloc(h_output.nbytes)

   stream = cuda.Stream()
   return h_input, d_input, h_output, d_output, stream 

def do_inference(engine, pics, h_input, d_input, h_output, d_output, stream, batch_size):
   np.copyto(h_input, np.asarray(pics).ravel()) 

   with engine.create_execution_context() as context:
       cuda.memcpy_htod_async(d_input, h_input, stream)

       context.execute(batch_size, bindings=[int(d_input), int(d_output)])

       cuda.memcpy_dtoh_async(h_output, d_output, stream)
       stream.synchronize()

       return h_output 

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.realpath(__file__))
    input_dir = os.path.join(root_dir, 'preprocessed_data/test')
    save_image_dir = os.path.join(root_dir, 'output')
    engine = build_engine(os.path.join(root_dir, "trained_model.onnx"))
    print('Model loaded...')

    infer_times = []
    for root, dirs, files in os.walk(input_dir, topdown=False):
        print('Processing ' + root + '...')
        for file in files:
            if file.endswith('.png'):
                img = cv2.imread(os.path.join(root, file))
                im = np.array(img, dtype=np.float32, order='C')
                h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)

                local_start = time.perf_counter()
                landmarks = do_inference(engine, im, h_input, d_input, h_output, d_output, stream, 1)
                local_end = time.perf_counter()
                infer_times.append(local_end - local_start)

                landmarks = landmarks.reshape(-1, 2)
                if save_image_dir is not None and len(save_image_dir):
                    pixel_landmark = landmarks * [img_width, img_height]

                    for (x, y) in pixel_landmark.astype(np.int32):
                        img = cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
                    save_image_path = os.path.join(save_image_dir, root, file)
                    save_image_full_dir = os.path.dirname(save_image_path)
                    os.makedirs(save_image_full_dir, exist_ok=True)
                    cv2.imwrite(save_image_path, img)

    print("inference_cost_time: {0:4f}".format(np.mean(infer_times)))
