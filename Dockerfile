FROM finsberg/pulse:latest

USER root

RUN apt-get -qq update && \
    apt-get -y upgrade && \
    apt-get -y install python3-scipy ipython3 python3-setuptools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
    
RUN pip3 install --upgrade pip
RUN pip3 install h5py --no-binary h5py
RUN pip3 install ldrb

RUN git clone https://github.com/ComputationalPhysiology/perspect.git
RUN cd perspect && python3 setup.py install && cd ..
