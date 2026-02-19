FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim
LABEL authors="antoine.lestrade"

# Set the working directory in the container to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY pyproject.toml /app
RUN uv sync

EXPOSE 1510/udp 1511/udp

# Add the current directory contents into the container at /app
COPY /src /app

ENV PYTHONPATH="/app"

CMD ["uv", "run", "MoMaMotiveLink/core/MotiveLink.py"]

