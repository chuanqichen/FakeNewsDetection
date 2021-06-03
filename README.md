## Setup: 

### 1: Install cuda driver like [CUDA Toolkit 10.2](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Linux&target_arch=x86_64&target_distro=CentOS&target_version=7&target_type=runfilelocal) 

```
Install instruction in Linux: 
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run

Update environment variable (you can add it into  ~/.bashrc): 
LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64"
PATH="$PATH:/usr/local/cuda-10.2/bin"
```

### 2: Install [pytorch](https://developer.nvidia.com/cuda-zone)  

On MAC or Linux, it's very simple: 

pip install torch==1.6.0 
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch

On Windows, please follow this link [pytorch](https://developer.nvidia.com/cuda-zone):   
No CUDA
To install PyTorch via pip, and do not have a CUDA-capable system or do not require CUDA, in the above selector, choose OS: Windows, Package: Pip and CUDA: None. Then, run the command that is presented to you.

With CUDA
To install PyTorch via pip, and do have a CUDA-capable system, in the above selector, choose OS: Windows, Package: Pip and the CUDA version suited to your machine. Often, the latest CUDA version is better. Then, run the command that is presented to you.

### 3: Install lxmert required libraries: 
https://github.com/chuanqichen/FakeNewsDetection/tree/main/lxmert

### 4: Install transformer
pip install transformers==3.0.0  --trusted-host  files.pythonhosted.org 

pip install transformers==3.5.1  --trusted-host  files.pythonhosted.org

pip install transformers==4.6.1  --trusted-host  files.pythonhosted.org

### 5: Other libraries: 
pip install matplotlib

pip install numpyencoder  --trusted-host files.pythonhosted.org

## Experiments Tips: 
### How to specify second GPU for python execution
CUDA_VISIBLE_DEVICES=1 to limit python code to use second GPU

CUDA_VISIBLE_DEVICES=0 python main.py >resnet_training_out_ephocs2.txt &


## Dataset: Fakeddit

Kai Nakamura, Sharon Levy, and William Yang Wang. 2020. r/Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection

Website: https://fakeddit.netlify.app/

Codalab Competition: https://competitions.codalab.org/competitions/25337

Paper: https://arxiv.org/abs/1911.03854

Our lab: http://nlp.cs.ucsb.edu/index.html


### Getting Started

Follow the instructions to download the dataset. You can download text, metadata, comment data, and image data.

Note that released test set is public. Private test set is used for leaderboard (coming soon).

Please read the `Usage` section. It is important.  

Please let us know if you encounter any problems by opening an issue or by directly contacting us.

### Installation

#### Download text and metadata
Please read the USAGE section before using or downloading. 
Download the v2.0 dataset from [here](https://drive.google.com/drive/folders/1jU7qgDqU1je9Y0PMKJ_f31yXRo5uWGFm?usp=sharing) 

#### Download image data 

**Option 1: (RECOMMENDED)**
Download the images [here](https://drive.google.com/file/d/1cjY6HsHaSZuLVHywIxD5xQqng33J5S2b/view?usp=sharing).

**Option 2:**
The `*.tsv` dataset files have an `image_url` column which contain the image urls. You can use the URLs to download the images.

For convenience, we have provided a script which will download the images for you. Please follow the instructions if you would like to use the attached script.

Fork or clone this repository and install required python libraries

```
$ git clone https://github.com/entitize/Fakeddit
$ cd Fakeddit
$ pip install -r requirements.txt
```
Copy `image_downloader.py` to the same directory/folder as where you downloaded the tsv files. 

Run `image_downloader.py`  in the new directory/folder

```
$ python image_downloader.py file_name
```

#### Download comment data
Download the comment data from [here](https://drive.google.com/drive/folders/150sL4SNi5zFK8nmllv5prWbn0LyvLzvo?usp=sharing)

### Usage

Please note that results in the paper are based on multimodal samples only (samples that have both text and image). In our paper, only samples that have both image and text were used for the baseline experiments and error analysis. Thus, if you would like to compare against the results in the paper, use the samples in the `multimodal_only_samples` folder. 

If there are `Unnamed`... columns, you can ignore or get rid of them. Use the `clean_title` column to get filtered text data. 

`comments.tsv` consists of comments made by Reddit users on submissions in the entire released dataset. Use the `submission_id` column to identify which submission the comment is associated with. Note that one submission can have zero, one, or multiple comments.

