FROM python:3.10-slim
 
WORKDIR /home
 
# Copy requirements first to leverage Docker's layer caching
# This step only re-runs if requirements.txt changes.
COPY requirements.txt .
 
# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
 
# Copy all project code and data into the container
COPY . .
 
# Environment variable for Ollama host (can be overridden at runtime)
ENV OLLAMA_HOST=http://host.docker.internal:11434
 
CMD ["python", "app/main.py"]