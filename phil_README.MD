
# Building the Docker Container
1. First you need to set the default runtime, see [here](https://github.com/dusty-nv/jetson-containers#docker-default-runtime). Be sure to restart the docker service afterwards.
2. From the top level phil-jetson-inference directory, run
`./docker/phil_build.sh`



# Running cmake install (from within the phil-jetson-inference folder)
```
mkdir build && \
cd build && \
cmake ../ && \
make -j$(nproc) && \
make install 
```