# Use an official lightweight Python image
FROM python:3.10-slim

# Set environment variables for non-interactive installs
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    PIP_NO_CACHE_DIR=1

# Set the working directory
WORKDIR /app

# Copy the QT package and dependencies
COPY qt/ /app/qt/
COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Set the default command
CMD ["python3"]
