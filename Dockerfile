# Use an official Python 3.11 runtime as the base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Copy application files to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for Flask
EXPOSE 8080

# Default environment variables (can be overridden during image creation or runtime)
ENV OPENSCALE_INSTANCEID="TESTdefault_instance_id"
ENV API_KEY="MtXf5AoqfpuYBfrOXa6St3VKrFFvipb6BzmGut5g0l-r"
ENV PROJECT_ID="9891cf2c-a599-48c5-9bd6-b4ed439d05bc"
ENV IAM_URL="https://iam.cloud.ibm.com"
ENV URL="https://us-south.ml.cloud.ibm.com"

# Run the application
CMD ["python", "app.py"]
