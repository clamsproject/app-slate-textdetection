from clams.app import ClamsApp
from clams.restify import Restifier
from mmif import Mmif, View, DocumentTypes, AnnotationTypes
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import cv2
import torch
import traceback
import os
import numpy
APP_VERSION = 0.1


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
        super().__init__()

    def _appmetadata(self):
        return {"name": "Slate Text Detection",
                "description": "This tool applies a custom Faster-RCNN model to slates specified in the input mmif.",
                "vendor": "Team CLAMS",
                "iri": f"http://mmif.clams.ai/apps/slate_td/{APP_VERSION}",
                "app": f"http://mmif.clams.ai/apps/slate_td/{APP_VERSION}",
                "requires": [DocumentTypes.VideoDocument.value, AnnotationTypes.TimeFrame.value],
                # here timeframe is mandatory
                "produces": [AnnotationTypes.BoundingBox.value]}

    def _annotate(self, mmif: Mmif, **kwargs) -> str:
        new_view = mmif.new_view()
        new_view.metadata['app'] = self.metadata["iri"]
        filename = mmif.get_document_location(DocumentTypes.VideoDocument)[7:]
        cap = cv2.VideoCapture(filename)
        file_basename = os.path.basename(filename)
        FRAME_TYPE = "slate"
        views_with_tframe = [
            tf_view
            for tf_view in mmif.get_all_views_contain(AnnotationTypes.TimeFrame)
            if tf_view.get_annotations(AnnotationTypes.TimeFrame, frameType=FRAME_TYPE)
        ]
        frame_number_ranges = [
            (tf_annotation.properties["start"], tf_annotation.properties["end"])
            for tf_view in views_with_tframe
            for tf_annotation in tf_view.get_annotations(AnnotationTypes.TimeFrame, frameType=FRAME_TYPE)
        ]
        target_frames = [(int(start) + int(end)) // 2 for start, end in frame_number_ranges]
        if not target_frames:
            print (f"No Slates for video: {file_basename}")
        for frame_number in target_frames:
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                _, slate_image = cap.read()
                cv2.imwrite(f"tmp/{frame_number}.jpg", slate_image)
                orig = slate_image.copy()
                slate_image = cv2.cvtColor(slate_image, cv2.COLOR_BGR2RGB)
                slate_boxes = self.process_slate(slate_image)
                for _id, box in enumerate(slate_boxes):
                    box = box.tolist()
                    box = [int(x) for x in box]
                    rect = cv2.rectangle(orig, (box[0],box[1]), (box[2],box[3]), (0,0,255), 5)
                    cv2.imwrite(f"tmp/{file_basename}_{_id}_rect.jpg",rect)
                    annotation = new_view.new_annotation(f"{str(frame_number)}_{str(_id)}", AnnotationTypes.BoundingBox)
                    annotation.add_property("coordinates", [[box[0], box[1]], [box[0],box[3]], [box[2], box[1]],[box[2],box[3]]])
                    annotation.add_property('boxType', "slate_text")
                    annotation.add_property("frame", frame_number)
            except:
                traceback.print_exc()
        return mmif

    def process_slate(self, image):
        image = image.transpose((2,0,1)).astype(float)
        image /= 255.0
        image = self.transforms(torch.unsqueeze(torch.tensor(image), dim=0).float())
        output = self.model(image)
        output_indices = torchvision.ops.nms(output[0]["boxes"], output[0]["scores"], iou_threshold=.1)
        print (output_indices)
        print (f"num_boxes:{len(output[0]['boxes'])}")
        boxes = output[0]['boxes'][output_indices]
        print (f"num_filtered_boxes:{len(boxes)}")
        return boxes

if __name__ == "__main__":
    td_tool = Slate_TD()
    td_service = Restifier(td_tool)
    td_service.run()
