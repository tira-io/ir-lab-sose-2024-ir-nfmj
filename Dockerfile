# A prepared image with python3.10, java 11, ir_datasets, tira, and PyTerrier installed 
FROM webis/ir-lab-wise-2023:0.0.4

# Update the tira command to use the latest version
RUN pip3 uninstall -y tira \
	&& pip3 install tira

RUN pip3 download https://github.com/explosion/spacy-models/releases/download/en_core_web_md-2.2.0/en_core_web_md-2.2.0.tar.gz
RUN pip3 install /en_core_web_md-2.1.0.tar.gz

ADD . /app

