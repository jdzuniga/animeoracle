FROM python:3.13-slim

# Set working directory first
WORKDIR /app

RUN pip install --no-cache-dir streamlit==1.47.1 pandas==2.3.1  && pip uninstall -y pyarrow

# Copy app and config
COPY src/config.py /app/src/
COPY src/utils.py /app/src/
COPY webapp.py /app/
COPY .streamlit /app/.streamlit

# Copy predictions folder
ARG TODAY
ENV RUN_DATE=$TODAY
COPY predictions/${RUN_DATE} /app/predictions/${RUN_DATE}
COPY posters/${RUN_DATE} /app/posters/${RUN_DATE}

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
ENTRYPOINT ["streamlit", "run"]
CMD ["webapp.py", "--server.port=8501", "--server.headless=true"]