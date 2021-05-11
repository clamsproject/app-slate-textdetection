FROM clamsproject/clams-python-opencv4

COPY ./ ./app
WORKDIR ./app

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3"]
CMD ["app.py"]