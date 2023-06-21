# Internal Imports
import argparse
from typing import Union
import os
import logging 

# CLAMS Imports
from clams import ClamsApp, Restifier
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes

# App Imports
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
import numpy as np
import cv2
import traceback

# Constants etc.
FRAME_TYPE = 'slate'
FRAME_RATE = 29.97
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
#=============================================================================|
class SlateTextDetection(ClamsApp):

    def __init__(self):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        num_classes = 2
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model.load_state_dict(torch.load('fasterrcnn_resnet50_fpn.pth', map_location=torch.device('cpu')))  
        self.model.eval()
        self.model = self.model.float()
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
        ])
        logging.info("model loaded")
        super().__init__()

    def _appmetadata(self):
        # see metadata.py
        pass

    def _annotate(self, mmif: Union[str, dict, Mmif], **parameters) -> Mmif:
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._annotate
        if isinstance(mmif, Mmif):
            mmif_obj = mmif
        else:
            mmif_obj = Mmif(mmif) 
        
        new_view = mmif_obj.new_view()
        config = self.get_configuration(**parameters)
        self.sign_view(new_view, config) 
        new_view.new_contain(AnnotationTypes.BoundingBox)
        filename = mmif_obj.get_document_location(DocumentTypes.VideoDocument)[7:]
        logging.debug(f"running document with location {filename}")
        cap = cv2.VideoCapture(filename)
        target_frames = self.get_target_frame_numbers(mmif_obj)
        for frame_number in target_frames:
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                _, slate_image = cap.read()
                slate_image = cv2.cvtColor(slate_image, cv2.COLOR_BGR2RGB)
                slate_boxes = self.process_slate(slate_image)
                for _id, box in enumerate(slate_boxes):
                    box = box.tolist()
                    box = [int(x) for x in box]
                    annotation = new_view.new_annotation(f"{str(frame_number)}_{str(_id)}", AnnotationTypes.BoundingBox)
                    annotation.add_property("coordinates",
                                            [[box[0], box[1]], [box[0],box[3]], [box[2], box[1]], [box[2], box[3]]])
                    annotation.add_property('boxType', 'text')
                    annotation.add_property('frame', frame_number)
            except:
                traceback.print_exc()
        return mmif_obj
    
    @staticmethod
    def get_target_frame_numbers(mmif, frames_per_segment=2):
        def convert_msec(time_msec):
            import math
            return math.floor(time_msec * FRAME_RATE)
        
        views_with_tframe = [
            tf_view
            for tf_view in mmif.get_all_views_contain(AnnotationTypes.TimeFrame)
            if tf_view.get_annotations(AnnotationTypes.TimeFrame, frameType=FRAME_TYPE)
        ]

        frame_number_ranges = [
            (tf_annotation.properties["start"], tf_annotation.properties["end"])
            if tf_view.metadata.get_parameter("unit") == "frame"
            else (convert_msec(tf_annotation.properties["start"]), convert_msec(tf_annotation.properties["end"]))
            for tf_view in views_with_tframe
            for tf_annotation in tf_view.get_annotations(AnnotationTypes.TimeFrame, frameType=FRAME_TYPE)
        ]

        target_frames = list(set([f for start, end in frame_number_ranges
                                   for f in np.linspace(start, end, frames_per_segment, dtype=int)]))
        logging.info(f"target frame list: {target_frames}")
        if not target_frames:
            logging.error(f"No slates for video: {mmif}")
        return target_frames
    
    def process_slate(self, image):
        image = image.transpose((2,0,1)).astype(float)
        image /= 255.0
        image = self.transforms(torch.unsqueeze(torch.tensor(image), dim=0).float())
        output = self.model(image)
        logging.info(f"detected {len(output[0]['boxes'])}")
        output_indices = torchvision.ops.nms(output[0]['boxes'], output[0]['scores'], iou_threshold=0.1)
        boxes = output[0]['boxes'][output_indices]
        logging.info(f"nms resulted in {len(boxes)} regions")
        return boxes 
    
#=============================================================================|
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", action="store", default="5000", help="set port to listen"
    )
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    # more arguments as needed
    # parser.add_argument(more_arg...)

    parsed_args = parser.parse_args()

    # create the app instance
    app = SlateTextDetection()

    http_app = Restifier(app, port=int(parsed_args.port)
    )
    if parsed_args.production:
        http_app.serve_production()
    else:
        http_app.run()
