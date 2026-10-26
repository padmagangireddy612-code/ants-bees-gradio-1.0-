
# 1. Select the base image
FROM python:3.11-slim

# 2. Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Copy the dependency list into the container
COPY requirements.txt .

# 5. Install all Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# 6. Copy the application code into the container
COPY app.py model_utils.py ./

# 7. Copy the saved model checkpoint into the container
COPY saved_model/ ./saved_model/

# 8. Create a non-root user for running the application
RUN useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /app

# 9. Switch to the non-root user
USER appuser

# 10. Document the application's default port
EXPOSE 7860

# 11. Start the Gradio application
CMD ["python", "app.py"]
