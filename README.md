# Knee osteoarthritis detection and categorization with deep learning models 

This repository provides the implementation of the method described in our paper:<br />
Knee osteoarthritis detection and categorization with deep learning models<br />
<i>Gourab Roy, Arup Kumar Pal, Manish Raj, Jitesh Pradhan</i>

Conference: The International Symposium on Artificial Intelligence (ISAI), 2025<br />
[To appear in conference proceedings in the <b>Lecture Notes in Networks and Systems</b> Book Series]

The code includes:

This repository provides the implementation of our proposed deep learning framework for detecting and classifying knee osteoarthritis (OA) severity from X-ray images. The model combines a modified Double U-Net architecture for hierarchical feature extraction, and a Convolutional Block Attention Module (CBAM) to predict Knee OA grades based on Kellgren-Lawrence (KL) grading system.

## How to set up Docker?
Docker makes it easy to build, share, and run the application in a consistent and reproducible environment, regardless of the host system. By using Docker, you avoid issues like dependency conflicts or mismatched library versions because all the necessary packages, configurations, and the runtime environment are encapsulated in the container. It also simplifies deployment to different machines and ensures that the code runs exactly as intended, whether on your local machine, a server, or in the cloud.

Make sure you have Docker installed.
Before you do anything make sure that the variables in config.py are properly set.

Now make sure you are in the code directory. Then run the following command:

```bash
docker build -t myapp-image .
```

The above command creates a docker image. Once the image is created, you need to pass the _dataset path_ in the following command to create and run the container:

```bash
docker run --gpus all --name my_container -v /path/to/your/dataset:/app/dataset -it myapp-image
```

## Note:
If you want to run without GPU, drop `--gpus all`
<!-- You can use this repository to reproduce the experiments and results presented in the paper. See the instructions in README.md for setup and usage details. -->
