"""
Example of using AutoDistill for detection.
Either run this script or:
$ cd /home/privvyledge/GitRepos/AutoDistill
$ autodistill images --base="grounding_dino" --target="yolov8" --ontology '{"prompt": "label"}' --output="./dataset"

"""

import os
import cv2
import supervision as sv
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
# from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.utils import plot


if __name__ == '__main__':
    run_number = 48
    frame = 'frame_000773.png'
    IMAGE_PATH = f"/mnt/c/Users/boluo/OneDrive - Florida State University/Projects/Research/ISC/isc_ws/data/Validation Data/Run_{run_number}/frames/VisualCamera1_Run_48/{frame}"
    IMAGE_DIR_PATH = f"/mnt/c/Users/boluo/OneDrive - Florida State University/Projects/Research/ISC/isc_ws/data/Validation Data/Run_{run_number}/frames"
    IMAGE_DIR_PATH = f"{IMAGE_DIR_PATH}/VisualCamera1_Run_{run_number}"
    DATASET_DIR_PATH = f"/mnt/c/Users/boluo/OneDrive - Florida State University/Projects/Research/ISC/isc_ws/data/Validation Data/Run_{run_number}/dataset"
    output_file = f"/mnt/c/Users/boluo/OneDrive - Florida State University/Projects/Research/ISC/isc_ws/data/Validation Data/Run_{run_number}/annotations"

    run_number = 2
    frame = 'frame_000871.png'
    IMAGE_PATH = f"/mnt/c/Users/boluo/OneDrive - Florida State University/Projects/Research/ISC/isc_ws/data/Sample Data Runs/Sample Data Run {run_number}/frames/VisualCamera4_sample_data_run_{run_number}/{frame}"
    IMAGE_DIR_PATH = f"/mnt/c/Users/boluo/OneDrive - Florida State University/Projects/Research/ISC/isc_ws/data/Sample Data Runs/Sample Data Run {run_number}/frames"
    IMAGE_DIR_PATH = f"{IMAGE_DIR_PATH}/VisualCamera4_sample_data_run_{run_number}"

    frame = 'frame_000939.png'  # 'frame_000939.png', 'frame_000027.png'
    IMAGE_PATH = f"/mnt/c/Users/boluo/OneDrive - Florida State University/Projects/Research/ISC/isc_ws/data/Sample Data Runs/Sample Data Run {run_number}/frames/ThermalCamera2_sample_data_run_{run_number}/{frame}"
    IMAGE_DIR_PATH = f"/mnt/c/Users/boluo/OneDrive - Florida State University/Projects/Research/ISC/isc_ws/data/Sample Data Runs/Sample Data Run {run_number}/frames"
    IMAGE_DIR_PATH = f"{IMAGE_DIR_PATH}/ThermalCamera2_sample_data_run_{run_number}"

    DATASET_DIR_PATH = f"/mnt/c/Users/boluo/OneDrive - Florida State University/Projects/Research/ISC/isc_ws/data/Sample Data Runs/Sample Data Run {run_number}/dataset"
    output_file = f"/mnt/c/Users/boluo/OneDrive - Florida State University/Projects/Research/ISC/isc_ws/data/Sample Data Runs/Sample Data Run {run_number}/annotations"

    HOME = os.getcwd()
    print(HOME)

    SAMPLE_SIZE = 16
    SAMPLE_GRID_SIZE = (4, 4)
    SAMPLE_PLOT_SIZE = (16, 16)

    # define an ontology to map class names to our Grounded SAM 2 prompt
    # the ontology dictionary has the format {caption: class}
    # where caption is the prompt sent to the base model, and class is the label that will
    # be saved for that caption in the generated annotations
    # then, load the model
    ontology = CaptionOntology({
        "person on wheelchair": "VRU",
        "person on bike": "VRU",
        "person on motorcycle": "VRU",
        "person on bicycle": "VRU",
        "person on scooter": "VRU",
        "person on skateboard": "VRU",
        "person": "VRU",
        # "bicycle": "VRU",
        "motorcycle": "VRU",
        "car": "vehicle",
        "vehicle": "vehicle",
        "bus": "vehicle",
        "truck": "vehicle",
    })

    base_model = GroundedSAM(ontology=ontology)
    # base_model2 = GroundedSAM2(ontology=ontology)

    # run inference on a single image
    results = base_model.predict(IMAGE_PATH)
    annotated_image = plot(
            image=cv2.imread(IMAGE_PATH),
            classes=base_model.ontology.classes(),
            detections=results,
            raw=True
    )

    # save the image
    image_type = 'visual' if 'Visual' in IMAGE_PATH else 'thermal'
    cv2.imwrite(f"{output_file}/grounded_sam_annotated_{image_type}_{frame}.png", annotated_image)

    # label all images in a folder called IMAGE_DIR_PATH
    dataset = base_model.label(
            input_folder=IMAGE_DIR_PATH,
            extension=".png",
            output_folder=DATASET_DIR_PATH)

    ANNOTATIONS_DIRECTORY_PATH = f"{DATASET_DIR_PATH}/train/labels"
    IMAGES_DIRECTORY_PATH = f"{DATASET_DIR_PATH}/train/images"
    DATA_YAML_PATH = f"{DATASET_DIR_PATH}/data.yaml"

    dataset = sv.DetectionDataset.from_yolo(
            images_directory_path=IMAGES_DIRECTORY_PATH,
            annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH,
            data_yaml_path=DATA_YAML_PATH)

    len(dataset)

    image_names = list(dataset.images.keys())[:SAMPLE_SIZE]

    mask_annotator = sv.MaskAnnotator()
    box_annotator = sv.BoxAnnotator()

    images = []
    for image_name in image_names:
        image = dataset.images[image_name]
        annotations = dataset.annotations[image_name]
        labels = [
            dataset.classes[class_id]
            for class_id
            in annotations.class_id]
        annotates_image = mask_annotator.annotate(
                scene=image.copy(),
                detections=annotations)
        annotates_image = box_annotator.annotate(
                scene=annotates_image,
                detections=annotations,
                labels=labels)
        images.append(annotates_image)

    sv.plot_images_grid(
            images=images,
            titles=image_names,
            grid_size=SAMPLE_GRID_SIZE,
            size=SAMPLE_PLOT_SIZE)

    print("Done")
