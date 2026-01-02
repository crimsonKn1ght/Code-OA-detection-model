# Knee osteoarthritis detection and categorization with deep learning models 

![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)
[![GitHub forks](https://img.shields.io/github/forks/crimsonKn1ght/Code-OA-detection-model.svg?style=social&label=Fork)](https://github.com/crimsonKn1ght/Code-OA-detection-model/network/members)
[![GitHub stars](https://img.shields.io/github/stars/crimsonKn1ght/Code-OA-detection-model.svg?style=social&label=★%20Star)](https://github.com/crimsonKn1ght/Code-OA-detection-model/stargazers)


This repository provides the implementation of the method described in our paper:<br />
Knee osteoarthritis detection and categorization with deep learning models<br />
<i>Gourab Roy, Arup Kumar Pal, Manish Raj, Jitesh Pradhan</i>

Conference: The International Symposium on Artificial Intelligence (ISAI), 2025.<br />

Published in conference proceedings in the <b>Lecture Notes in Networks and Systems</b> Book Series by **Springer Nature**

Our paper is available at: [https://doi.org/10.1007/978-981-96-9239-2_9](https://doi.org/10.1007/978-981-96-9239-2_9)

## Abstract
This paper deploys deep learning models to detect and categorize knee osteoarthritis disease. The mentioned disease affects the joint areas and is marked by degrading transformations in the surrounding bone structure and tissues accompanied by a progressive breakdown of articular cartilage. In such studies, the Kellgren–Lawrence (KL) grading system is mainly adopted to determine the extent of knee osteoarthritis (OA). Based on the training of previously graded X-ray images, deep learning models can simultaneously automate the prediction process and expedite the diagnosis process with reduced human error. Our proposed model integrates a modified version of the Double U-Net model, a hierarchical system, and CBAM, an attention module to accurately predict the KL class of the knee OA X-ray scans. The training of the proposed model stands at 98%, while the validation accuracy is 80%. Our proposed model is efficient and achieves the desirable outcome with the available datasets.

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

## Cite this paper
Roy, G., Pal, A.K., Raj, M., Pradhan, J. (2026). Knee Osteoarthritis Detection and Categorization with Deep Learning Models. In: Giri, D., Ekbal, A., Ray, S., Kouichi, S. (eds) Proceedings of Second International Symposium on Artificial Intelligence. ISAI 2025. Lecture Notes in Networks and Systems, vol 1536. Springer, Singapore. https://doi.org/10.1007/978-981-96-9239-2_9
