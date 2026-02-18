FROM python:3.14-slim

# Set working directory
WORKDIR /alzheimer_mri_cnn_app

# Copy python requirements in the container
COPY requirements.txt .

# Install libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code in the container
COPY src/ ./src/

# Copy input files
COPY inputs/ ./inputs/

# Copy notebooks
COPY notebooks/ ./notebooks/

# Create folder for dataset
RUN mkdir /alzheimer_mri_cnn_app/data

# Create folder for outputs
RUN mkdir /alzheimer_mri_cnn_app/outputs

# Disable python print buffering
ENV PYTHONUNBUFFERED=1

# Default comand when container starts
ENTRYPOINT ["python", "src/train_evaluate.py"]

# Pass default input file
CMD ["--input", "18_12_25_1923"]