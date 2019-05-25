FROM tensorflow/tensorflow:2.0.0a0-py3-jupyter

RUN apt-get update && apt-get install -y locales

# fix python encode/decode error in Ubuntu
# from https://stackoverflow.com/questions/27931668/encoding-problems-when-running-an-app-in-docker-python-java-ruby-with-u/27931669
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

WORKDIR /opt/rossmann-tf

EXPOSE 8888
RUN pip3 install pandas pyarrow jupyter_contrib_nbextensions jupyter_nbextensions_configurator sklearn tensorflow-addons
RUN python3 -m ipykernel.kernelspec
RUN jupyter nbextensions_configurator enable --user
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable toc2/main
COPY . .
CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/opt/rossmann-tf --ip 0.0.0.0 --no-browser --allow-root"]