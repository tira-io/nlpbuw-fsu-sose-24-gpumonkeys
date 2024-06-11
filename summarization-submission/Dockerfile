FROM fschlatt/natural-language-processing-exercises:0.0.1

# Add requirements.txt and run.py to the container
ADD requirements.txt /code/requirements.txt
ADD run.py /code/run.py

# Install the required packages
RUN pip install -r /code/requirements.txt

# Define the entry point for the container
ENTRYPOINT ["python3", "/code/run.py"]
