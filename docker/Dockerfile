FROM gcr.io/deeplearning-platform-release/tf2-cpu

COPY . /cats_vs_dogs
ENV PYTHONPATH ${PYTHONPATH}:/cats_vs_dogs
ENTRYPOINT ["python3", "-m", "trainer.task"]
