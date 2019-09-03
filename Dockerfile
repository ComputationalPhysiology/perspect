
FROM finsberg/pulse:latest

RUN sudo apt-get update && sudo apt-get -y install git python3-setuptools ipython3
RUN pip3 install --upgrade pip

RUN git clone https://github.com/ComputationalPhysiology/perspect.git
RUN cd perspect && python3 setup.py install && cd ..
