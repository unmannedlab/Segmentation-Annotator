# Segmentation Annotator

This repository contains a segmentation annotator tool that allows you to annotate images for segmentation tasks. The tool was used to annotate [The Great Outdoors Dataset](http://www.unmannedlab.org/the-great-outdoors-dataset/).


https://github.com/user-attachments/assets/f53c23f6-0f35-46b2-ba8b-2936869aa97c


## Features

- Super Pixelation
- Auto "SAM" Masks
- Interactive SAM masks
- Rectangle SAM mask
- Coarse OFFseg segmentation
- Manual Polygon

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/kasiv008/Segmentation-annotator.git
    ```

2. Install the required dependencies: 

    ```bash
    pip install -r requirements.txt
    ```

3. Install SAM package

    ```bash
    cd segment-anything; pip install -e .
    ```
Download the [SAM checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and update the path in gui.py

## Usage

1. Run the annotator tool:

    ```bash
    python gui.py
    ```

2. Load an image and start annotating.

3. Save your annotations and export the annotated image.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
