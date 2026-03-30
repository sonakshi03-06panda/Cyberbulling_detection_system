FROM python:3.10

WORKDIR /app

# Copy dependencies first for Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Hugging Face Spaces exposes port 7860 by default for Docker Spaces
EXPOSE 7860

# Run the app
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
