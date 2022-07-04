# HyPIX
Tool for exploring time series of hyperspectral images for cold water coral stress response analysis


The h5 data file to be loaded by the tool can be downloaded at https://ani.cebitec.uni-bielefeld.de/hypix.h5


You can start the tool for example using gunicorn with gunicorn --workers 8 --bind 0.0.0.0:8080 hyperspectralViewer:server to serve the website on port 8080 with 8 workers. The tool is then accesible at localhost:8080.