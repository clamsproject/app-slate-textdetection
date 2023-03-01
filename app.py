from clams.app import ClamsApp
from clams.restify import Restifier
from clams.appmetadata import AppMetadata
from mmif import Mmif, DocumentTypes, AnnotationTypes
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import cv2
import numpy as np
import torch
import traceback
import os
import logging

APP_VERSION = 0.1
FRAME_TYPE = "slate"
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


class Slate_TD(ClamsApp):
    def __init__(self):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
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

    def _appmetadata(self) -> AppMetadata:
        metadata = AppMetadata(
            name="Slate Text Detection",
            description="This tool applies a custom Faster-RCNN model to slates specified in the input mmif.",
            app_version=APP_VERSION,
            app_license='MIT',
            analyzer_license='apache',
            url=f"https://github.com/clamsproject/slate_td/{APP_VERSION}", 
            identifier=f"http://apps.clams.ai/slate_td/{APP_VERSION}",
        )
        metadata.add_input(DocumentTypes.VideoDocument)
        metadata.add_input(AnnotationTypes.TimeFrame, required=True, frameType='slate')

        metadata.add_output(AnnotationTypes.BoundingBox)
        return metadata


    def _annotate(self, mmif: Mmif, **kwargs) -> str:
        new_view = mmif.new_view()
        config = self.get_configuration(**kwargs)
        self.sign_view(new_view, config)
        new_view.new_contain(AnnotationTypes.BoundingBox.value)
        filename = mmif.get_document_location(DocumentTypes.VideoDocument)[7:]
        logging.debug(f"running document with location {filename}")
        cap = cv2.VideoCapture(filename)
        target_frames = self.get_target_frame_numbers(mmif)
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
                                            [[box[0], box[1]], [box[0], box[3]], [box[2], box[1]], [box[2], box[3]]])
                    annotation.add_property('boxType', "text")
                    annotation.add_property("frame", frame_number)
            except:
                traceback.print_exc()
        return mmif

    @staticmethod
    def get_target_frame_numbers(mmif, frames_per_segment=2):
        def convert_msec(time_msec):
            import math
            return math.floor(time_msec * 29.97)  # todo 6/1/21 kelleylynch assuming frame rate

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
            logging.error(f"No Slates for video: {mmif}")
        return target_frames

    def process_slate(self, image):
        image = image.transpose((2, 0, 1)).astype(float)
        image /= 255.0
        image = self.transforms(torch.unsqueeze(torch.tensor(image), dim=0).float())
        output = self.model(image)
        logging.info(f"detected {len(output[0]['boxes'])}")
        output_indices = torchvision.ops.nms(output[0]["boxes"], output[0]["scores"], iou_threshold=.1)
        boxes = output[0]['boxes'][output_indices]
        logging.info(f"nms resulted in {len(boxes)} regions")
        return boxes


if __name__ == "__main__":
    td_tool = Slate_TD()
    td_service = Restifier(td_tool)
    td_service.run()
