
# Building Docker image using multi-stage strategy to reduce image size

# First stage: Install dependencies
FROM python:3.8-slim-buster AS builder

# Set working directory
WORKDIR /install

# Copy only requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Second stage: Create the final image
FROM python:3.8-slim-buster

# Set working directory
WORKDIR /app

# Copy installed dependencies from the builder stage
COPY --from=builder /install /usr/local

# Copy the rest of the application
COPY . .

# Cleanup unnecessary files if any
RUN rm -rf /var/lib/apt/lists/*

# Command to run the application
CMD ["flask", "run", "-h", "0.0.0.0", "-p", "5000"]
