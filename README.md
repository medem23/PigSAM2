# PigSAM2

## Description

App based interactive Pig tracking using SAM2 model. 


## Prerequisites

* **Git:** To clone the repository.
* **Conda:** Anaconda or Miniconda package manager installed.
* **NVIDIA GPU:** A CUDA-compatible NVIDIA GPU.
* **NVIDIA Driver:** A recent NVIDIA driver compatible with CUDA 12.8. 


## Installation

Follow these steps to set up the project environment and install dependencies:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/medem23/PigSAM2.git
    cd PigSAM2
    ```

2.  **Create Conda Environment:**
    This project uses Conda to manage dependencies, including Python, CUDA Toolkit, PyTorch, and other libraries. The environment is defined in the `environment.yml` file.
    ```bash
    conda env create -f env.yml
    ```
    This command will create a new Conda environment named `my_cuda_env` (or whatever name is specified inside `environment.yml`).

3.  **Activate Conda Environment:**
    ```bash
    conda activate my_cuda_env
    ```
4.  **Build Custom CUDA Extension:**
    This project includes custom CUDA code that needs to be compiled. Run the following command from the root directory of the repository (where `setup.py` is located):
    ```bash
    pip install -e .
    ```
    
## Run Fish Eye Compensation

To undistort a fisheye video, run the following command:  

```bash
python undistort.py <input_video_path> <output_video_path> --balance <balance_value>
```

## Running the Streamlit Application

Once the installation is complete and the Conda environment is activated:

1.  **Navigate to the project directory** (if you aren't already there).
2.  **Run the Streamlit app:**
    ```bash
    streamlit run streamlit.py --server.address 127.0.0.1 --server.port 8501
    ```
    * This command starts the Streamlit application using the `streamlit.py` script.
    * `--server.address` and `--server.port` are optional flags that specify the network address and port. By default, Streamlit often tries to open in your browser automatically.
    * You should be able to access the application in your web browser, typically at `http://127.0.0.1:8501` (or the address/port Streamlit indicates in the terminal output).


## Acknowledgements

* This project utilizes the **Segment Anything Model 2 (SAM2)** developed by **Meta AI Research**. Thanks for their great open-source contribution ❤️❤️.

