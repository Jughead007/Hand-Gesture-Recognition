from abc import ABC

import cv2
import numpy as np
import onnxruntime as ort


class OnnxModel(ABC):
    def __init__(self, model_path, image_size):
        self.model_path = model_path
        self.image_size = image_size
        self.mean = np.array([127, 127, 127], dtype=np.float32)
        self.std = np.array([128, 128, 128], dtype=np.float32)

        options, providers = self.get_onnx_provider()
        self.sess = ort.InferenceSession(
            model_path, sess_options=options, providers=providers
        )
        self._get_input_output()

    def preprocess(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = (image - self.mean) / self.std
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        return image

    def _get_input_output(self):
        inputs = self.sess.get_inputs()
        self.inputs = "".join(
            [
                f"\n {i}: {input.name} Shape: ({','.join(map(str, input.shape))}) Dtype: {input.type}"
                for i, input in enumerate(inputs)
            ]
        )

        outputs = self.sess.get_outputs()
        self.outputs = "".join(
            [
                f"\n {i}: {output.name} Shape: ({','.join(map(str, output.shape))}) Dtype: {output.type}"
                for i, output in enumerate(outputs)
            ]
        )

    @staticmethod
    def get_onnx_provider():
        providers = ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        print("Using ONNX Runtime", ort.get_device())

        # Optional: extend support for GPU if desired in future
        if "DML" in ort.get_device():
            providers.append("DmlExecutionProvider")
        elif "GPU" in ort.get_device():
            providers.append("CUDAExecutionProvider")

        return options, providers

    def __repr__(self):
        return (
            f"Providers: {self.sess.get_providers()}\n"
            f"Model: {self.sess.get_modelmeta().description}\n"
            f"Version: {self.sess.get_modelmeta().version}\n"
            f"Inputs: {self.inputs}\n"
            f"Outputs: {self.outputs}"
        )


class HandDetection(OnnxModel):
    def __init__(self, model_path, image_size=(320, 240)):
        super().__init__(model_path, image_size)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [output.name for output in self.sess.get_outputs()]

    def __call__(self, frame):
        input_tensor = self.preprocess(frame)
        boxes, _, probs = self.sess.run(self.output_names, {self.input_name: input_tensor})
        width, height = frame.shape[1], frame.shape[0]
        boxes[:, 0] *= width
        boxes[:, 1] *= height
        boxes[:, 2] *= width
        boxes[:, 3] *= height
        return boxes.astype(np.int32), probs


class HandClassification(OnnxModel):
    def __init__(self, model_path, image_size=(128, 128)):
        super().__init__(model_path, image_size)

    @staticmethod
    def get_square(box, image):
        height, width, _ = image.shape
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        if h < w:
            y0 -= int((w - h) / 2)
            y1 = y0 + w
        elif h > w:
            x0 -= int((h - w) / 2)
            x1 = x0 + h
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(width - 1, x1)
        y1 = min(height - 1, y1)
        return x0, y0, x1, y1

    def get_crops(self, frame, bboxes):
        crops = []
        for bbox in bboxes:
            bbox = self.get_square(bbox, frame)
            crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            crops.append(crop)
        return crops

    def __call__(self, image, bboxes):
        crops = self.get_crops(image, bboxes)
        crops = [self.preprocess(crop) for crop in crops]
        input_name = self.sess.get_inputs()[0].name
        outputs = self.sess.run(None, {input_name: np.concatenate(crops, axis=0)})[0]
        labels = np.argmax(outputs, axis=1)
        return labels
