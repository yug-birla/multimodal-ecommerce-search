# Use a lightweight Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /code

# Copy over the requirements and install them
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy over your application code
COPY ./app /code/app

# ADD THIS NEW LINE: Copy your frontend UI into the cloud!
COPY ./static /code/static

# Hugging Face Spaces require port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]