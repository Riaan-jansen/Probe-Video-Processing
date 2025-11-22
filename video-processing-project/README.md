# Video Processing Project

## Overview
This project is designed for processing video files, specifically focusing on analyzing motion and calculating centers of rotation. It utilizes various Python libraries to perform video analysis, visualize results, and generate output videos.

## Project Structure
The project is organized as follows:

```
video-processing-project
├── data
│   ├── input          # Directory for input .mp4 files
│   │   └── README.md  # Placeholder for input .mp4 files
│   └── output         # Directory for output .mp4 files
│       └── README.md  # Placeholder for output .mp4 files
├── src
│   ├── video_processing.ipynb  # Jupyter notebook for video processing
│   ├── video_processing_tools.py # Utility functions for video processing
│   └── utils
│       └── __init__.py         # Marks the utils directory as a Python package
├── requirements.txt             # Lists project dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Specifies files to ignore by Git
```

## Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd video-processing-project
   ```

2. **Install Dependencies**
   Ensure you have Python installed, then install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Input Data**
   Place your input `.mp4` files in the `data/input` directory.

4. **Run the Analysis**
   Open the Jupyter notebook `src/video_processing.ipynb` and run the cells to process the videos.

5. **Output Data**
   The processed output videos will be saved in the `data/output` directory.

## Usage Guidelines
- Modify the parameters in the Jupyter notebook to suit your video analysis needs.
- Ensure that the input videos are in the correct format and resolution for optimal results.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.