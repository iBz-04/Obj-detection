
# Object Detection with TensorFlow SSD MobileNet V2

This project demonstrates object detection using TensorFlow's SSD MobileNet V2 model, a pre-trained model from the TensorFlow Object Detection API. The model is capable of detecting common objects in images with bounding boxes, class labels, and confidence scores(ranges from 0 = [lowest] to 1 = [highest] the closer the score is to 1 the better).


## Installation

Run locally

```bash
git clone https://github.com/iBz-04/Obj-detection
cd https://github.com/iBz-04/Obj-detection
```
    
## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- OpenCV (cv2)
- Pillow (PIL)
- Matplotlib


## Run Locally

Clone the project

```bash
  git clone https://github.com/iBz-04/Obj-detection
```

Go to the project directory

```bash
  cd https://github.com/iBz-04/Obj-detection
```

Install dependencies

```bash
  pip install tensorflow 
  pip install numpy 
  pip install opencv-python
  pip install pillow 
  pip install matplotlib
```

* Download the pre-trained SSD MobileNet V2 model from the TensorFlow Model Zoo and place it in the models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/ directory



## Example Output
When you run the script on an image, the output will look similar to this:

<img src="https://res.cloudinary.com/diekemzs9/image/upload/v1729453505/Screenshot_2024-10-20_220015_fsm189.png" alt="Original Image" width="800"/>

