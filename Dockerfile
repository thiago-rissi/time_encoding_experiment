# Use python:3.10 as base image for both cases
FROM python:3.11

ARG PYTHON_VERSION=3.11

# Set the working directory
WORKDIR /usr/src/code/

# Copy your application's source code
COPY . /usr/src/code/

# Install pip and requirements based on USE_PROXY flag
RUN python -m pip install --upgrade pip

RUN python -m pip install -r requirements.txt

RUN python -m pip install -r requirements-torch.txt

# Expose the application on port 5000
EXPOSE 5000

# tail entry point
ENTRYPOINT ["tail", "-f", "/dev/null"]

# ENTRYPOINT ["./docker-entrypoint.sh"]