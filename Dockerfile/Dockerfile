FROM python:3.9-slim

WORKDIR /opt
RUN apt-get update && apt-get install -y git
RUN pip install --upgrade pip
RUN pip install pytest \
                flake8 \
                tqdm \
                numpy \
                pandas \
                scipy \
                matplotlib \
                seaborn \
                scikit-learn \
                torch \
                torchvision \
                torch_summary


WORKDIR /work