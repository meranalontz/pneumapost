FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 python3.8-distutils python3-pip git \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

WORKDIR /opt/pneumapost
COPY requirements.txt ./
RUN python -m pip install --upgrade "pip==22.2.2" \
    && python -m pip install -r requirements.txt

COPY . .
RUN python -m pip install --no-deps -e .

ENTRYPOINT ["python", "-m", "pneumapost.wicket"]
CMD ["--help"]
