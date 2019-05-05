ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:19.04-py3
FROM ${FROM_IMAGE_NAME}
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update && apt-get install -y pbzip2 pv bzip2

RUN pip install toposort networkx pytest nltk tqdm html2text progressbar
RUN pip install gensim nltk pandas subword-nmt tqdm sklearn tensorflow-hub tables

WORKDIR /workspace/bert
RUN git clone https://github.com/openai/gradient-checkpointing.git gradientcheckpointing
RUN git clone https://github.com/attardi/wikiextractor.git
RUN git clone https://github.com/soskek/bookcorpus.git

RUN apt-get install -y openssh-server git
RUN mkdir /var/run/sshd
RUN echo 'root:testdocker' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

WORKDIR /workspace/bert
COPY . .

ENV PYTHONPATH=/workspace/bert
EXPOSE 22 8888
CMD ["bash", "-c", "/usr/sbin/sshd && jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"]
