FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD requirements.txt /code/requirements.txt
ADD run.py /code/run.py

RUN pip install -r /code/requirements.txt

ENTRYPOINT [ "python3", "/code/run.py" ]
