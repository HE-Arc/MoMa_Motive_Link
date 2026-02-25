FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim
LABEL authors="antoine.lestrade"

# Set the working directory in the container to /app
WORKDIR /app

# Install dependencies
RUN apt-get update \
    && apt-get install -y bison build-essential \
    curl flex git libassimp-dev libbz2-dev libc6-dev \
    libffi-dev libgdbm-dev libncursesw5-dev libsqlite3-dev \
    libssl-dev libxml2-dev tk-dev zlib1g-dev

# Install any needed packages specified in requirements.txt
COPY pyproject.toml /app
RUN uv sync

EXPOSE 1510/udp 1511/udp 1520/udp

# Add the current directory contents into the container at /app
COPY /src /app
COPY /tests /app/tests

ENV PYTHONPATH="/app"

CMD ["uv", "run", "MoMaMotiveLink/core/MotiveLink.py"]
#CMD ["uv", "run", "tests/test_natnet.py"]

