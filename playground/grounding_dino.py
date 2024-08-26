"""
Either run this script or:
$ cd /path/to/GroundingDINO
$ CUDA_VISIBLE_DEVICES=0 python demo/inference_on_a_image.py -c groundingdino/config/GroundingDINO_SwinT_OGC.py
    -p ./weights/groundingdino_swint_ogc.pth
    -i "./VisualCamera1_Run_48/frame_000773.png"
    -o logs/1111 -t "bicycle . person . car ."
"""

from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2


if __name__ == '__main__':
    model = load_model(
            "/path/to/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "/path/to/GroundingDINO/GroundingDINO/weights/groundingdino_swint_ogc.pth")

    IMAGE_PATH = f""
    output_file = f""
    TEXT_PROMPT = "bicycle . person . car ."
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    image_source, image = load_image(IMAGE_PATH)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite(f"{output_file}/grounding_dino_annotated.png", annotated_frame)
