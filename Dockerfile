# parameters
ARG REPO_NAME="template-ros-core"

# ==================================================>
# ==> Do not change this code
FROM light5551/ado_torch_template:arm64
# define repository path
ARG REPO_NAME
ARG REPO_PATH="${CATKIN_WS_DIR}/src/${REPO_NAME}"
WORKDIR "${REPO_PATH}"

# create repo directory
RUN mkdir -p "${REPO_PATH}"

# copy dependencies files only
RUN ls -la
COPY ./dependencies-apt.txt "${REPO_PATH}/"
COPY ./dependencies-py.txt "${REPO_PATH}/"

RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# install apt dependencies
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    $(awk -F: '/^[^#]/ { print $1 }' dependencies-apt.txt | uniq) \
  && rm -rf /var/lib/apt/lists/*

# install python dependencies
RUN pip install -r ${REPO_PATH}/dependencies-py.txt
#RUN pip install https://github.com/smorad/arm64-popular-wheels/blob/master/aarch64/ubuntu_20.04/ray-1.5.1-cp38-cp38-linux_aarch64.whl
RUN pip install pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-2.0.0.dev0-cp38-cp38-linux_aarch64.whl
# copy the source code
COPY . "${REPO_PATH}/"

RUN cp -r "${REPO_PATH}/packages/dt-core" "${CATKIN_WS_DIR}/src/"
RUN rm -r "${REPO_PATH}/packages/dt-core"

# build packages
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
  catkin build \
    --workspace ${CATKIN_WS_DIR}/

# define launch script
ENV LAUNCHFILE "${REPO_PATH}/launch.sh"

# define command
CMD ["bash", "-c", "${LAUNCHFILE}"]
# <== Do not change this code
# <==================================================




# maintainer
LABEL maintainer="Konstantin Chaika (pro100kot14@gmail.com)"
