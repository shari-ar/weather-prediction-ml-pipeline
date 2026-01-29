##
# Dockerfile for the weather prediction ML pipeline development container.
#
# This image is intentionally slim to reduce the size of the development
# environment. It installs only the dependencies declared in
# requirements.txt and copies the source tree into the container.
# Additional tools can be installed by extending this file.
##

FROM python:3.11-slim

# Set a working directory for the project. All relative paths in the dev
# container will be resolved from this directory.
WORKDIR /workspace

# Copy the dependency specification into the container. Keeping this
# separate from the source code allows Docker to cache the layer after
# dependencies are installed, avoiding reinstalling packages on every
# source code change.
COPY requirements.txt ./

# Install Python dependencies without caching to keep the image size small.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container. Use a separate COPY so that
# changes in requirements.txt do not invalidate this layer.
COPY src/ src/

# The default command simply starts a bash shell. When using VS Code
# Dev Containers, this will be overridden by the editor.
CMD ["bash"]