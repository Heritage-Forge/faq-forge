# ---------- Builder Stage ----------
FROM ubuntu:20.04 AS builder
ARG DEBIAN_FRONTEND=noninteractive

# Install Python 3.9 and build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 python3.9-venv python3-pip build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and create virtual environment
COPY requirements.txt .
RUN python3.9 -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the image
COPY . .

# ---------- Runner Stage ----------
FROM ubuntu:20.04 AS runner
ARG DEBIAN_FRONTEND=noninteractive

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 python3.9-venv && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd --create-home myuser
WORKDIR /home/myuser/app

# Copy the built application from the builder stage
COPY --from=builder /app /home/myuser/app

# Switch to non-root user
USER myuser

# Activate the virtual environment and set environment variables
ENV PATH="/home/myuser/app/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Set the default command to show CLI help (adjust as needed)
CMD ["python3.9", "src/main.py", "--help"]
