FROM tensorflow/tensorflow:1.15.5-py3

RUN apt-get update \
    && apt-get install -y \
        postgresql-client \
        libgl1-mesa-glx \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

RUN python3 -m pip install --upgrade pip

# Install Django and additional libraries
COPY ./requirements.txt .
RUN pip install -r requirements.txt && \
    pip install --upgrade git+https://github.com/seungjinhan/python_jimmy_util.git && \
    git clone https://github.com/thtrieu/darkflow.git && \
    cd /usr/src/app/darkflow && \
    pip install .


WORKDIR /usr/src/app

COPY . .

EXPOSE 8000
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]