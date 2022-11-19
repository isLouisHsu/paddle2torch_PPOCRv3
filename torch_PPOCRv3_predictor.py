import os
import cv2
import torch
import numpy as np
from addict import Dict as AttrDict
from torch_det_infer import (
    DBPostProcess, DetModel,
    img_nchw as det_image_nchw,
)
from torch_rec_infer import (
    CTCLabelConverter, RecModel,
    img_nchw as rec_image_nchw,
)


from PIL import Image, ImageDraw, ImageFont
def putText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)): # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


class TorchPPOCRv3Predictor(object):

    def __init__(
        self, 
        det_model_path="./weights/ppv3_db.pth",
        det_pp_params=AttrDict(thresh=0.3, box_thresh=0.4, max_candidates=1000, unclip_ratio=2),
        rec_model_path="./weights/ppv3_rec.pth",
        rec_dict_path="./weights/ppocr_keys_v1.txt",
        device="cpu",
    ):
        self.device = device = torch.device(device)

        # initialize det model
        db_config = AttrDict(
            in_channels=3,
            backbone=AttrDict(
                type='MobileNetV3', 
                model_name='large',
                scale=0.5,
                pretrained=True
            ),
            neck=AttrDict(
                type='RSEFPN', 
                out_channels=96
            ),
            head=AttrDict(
                type='DBHead'
            ),
        )
        det_model = DetModel(db_config)
        det_model.load_state_dict(torch.load(det_model_path))
        det_model = det_model.to(device)
        det_model.eval()
        post_process = DBPostProcess(**det_pp_params)
        self.det_model = det_model
        self.post_process = post_process

        # initialize rec model
        rec_config = AttrDict(
            in_channels=3,
            backbone=AttrDict(
                type='MobileNetV1Enhance', 
                scale=0.5, 
                last_conv_stride=[1, 2], 
                last_pool_type='avg',
            ),
            neck=AttrDict(type='None'),
            head=AttrDict(
                type='Multi', 
                head_list=AttrDict(
                    CTC=AttrDict(
                        Neck=AttrDict(
                            name="svtr", 
                            dims=64, 
                            depth=2, 
                            hidden_dims=120, 
                            use_guide=True
                        )
                    ),
                    # SARHead=AttrDict(enc_dim=512,max_text_length=70)
                ),
                n_class=6625,
            ),
        )
        rec_model = RecModel(rec_config)
        rec_model.load_state_dict(torch.load(rec_model_path))
        rec_model = rec_model.to(device)
        rec_model.eval()
        converter = CTCLabelConverter(rec_dict_path)
        self.rec_model = rec_model
        self.converter = converter

    @torch.no_grad()
    def __call__(self, image, size=640):
        output = dict()
        image0, box_list, score_list = self.predict_det(image, size)
        output["image"] = image0
        output["items"] = []
        for box, box_score in zip(box_list, score_list):
            crop, box = self.crop_image(image0, box)
            text, text_score = self.predict_rec(crop)
            output["items"].append(dict(
                box=box, 
                text=text, 
                box_score=box_score, 
                text_score=text_score,
            ))
        return output

    @torch.no_grad()
    def predict_det(self, image, size):
        image0, image_np_nchw = det_image_nchw(image, size)
        input_for_torch = torch.from_numpy(image_np_nchw)
        out = self.det_model(input_for_torch)  # torch model infer
        box_list, score_list = self.post_process(
            out, [image0.shape[:2]], is_output_polygon=False)
        box_list, score_list = box_list[0], score_list[0]
        if len(box_list) > 0:
            idx = [x.sum() > 0 for x in box_list]
            box_list = [box_list[i].astype(np.int).tolist() for i, v in enumerate(idx) if v]
            score_list = [score_list[i] for i, v in enumerate(idx) if v]
        else:
            box_list, score_list = [], []
        return image0, box_list, score_list

    @torch.no_grad()
    def predict_rec(self, image):
        image_np_nchw = rec_image_nchw(image)
        input_for_torch = torch.from_numpy(image_np_nchw)
        out = self.rec_model(input_for_torch).softmax(dim=2)
        text_score_list = self.converter.decode(out.detach().cpu().numpy())
        texts, scores = list(zip(*text_score_list))
        return texts[0], scores[0]
    
    def crop_image(self, image, box):
        h, w, c = image.shape
        x, y = list(zip(*box))
        y1, y2, x1, x2 = min(y), max(y), min(x), max(x)
        y1 = max(min(y1, h), 0)
        y2 = max(min(y2, h), 0)
        x1 = max(min(x1, w), 0)
        x2 = max(min(x2, w), 0)
        return image[y1: y2, x1: x2], (y1, y2, x1, x2)
    
    def draw(
        self, 
        image, 
        items, 
        box_color=(0, 0, 255), 
        box_thickness=1, 
        text_color=(0, 0, 255), 
        text_size=14,
    ):
        image = image.copy()
        for item in items:
            y1, y2, x1, x2 = item["box"]
            points = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]], dtype=np.int)
            cv2.polylines(image, [points], True, box_color, box_thickness)
            image = putText(image, item["text"], x1, y2, text_color, text_size)
        return image

if __name__ == "__main__":
    from torch_det_infer import draw_bbox as det_draw_bbox

    predictor = TorchPPOCRv3Predictor()
    image_dir = "./det_images/"
    image_files = os.listdir(image_dir)
    for image_file in image_files:
        image = cv2.imread(os.path.join(image_dir, image_file))
        # image = cv2.imread("det_images/2.png")
        results = predictor(image, 640)
        for item in results["items"]:
            print(item)
        cv2.imshow("draw", predictor.draw(**results)); cv2.waitKey()

        # det_draw_bbox(image0, )
    ...
    