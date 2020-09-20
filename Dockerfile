FROM ubuntu:16.04

LABEL Name=chromosight Version=1.3.1

COPY * ./ /app/
WORKDIR /app

RUN apt-get update && apt-get install -y curl

RUN curl -LO https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda config --add channels bioconda

RUN conda install -c conda-forge -y cooler

RUN pip install -Ur requirements.txt

RUN pip install .

ENTRYPOINT [ "chromosight" ]
