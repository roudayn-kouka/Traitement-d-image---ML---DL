FROM node:22-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-venv libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY server/package*.json ./server/
RUN cd server && npm install

COPY server/requirements-inference.txt ./server/requirements-inference.txt
RUN python3 -m pip install --no-cache-dir --break-system-packages -r server/requirements-inference.txt

COPY server ./server
COPY plant_disease_app ./plant_disease_app
COPY data ./data

WORKDIR /app/server
RUN mkdir -p /app/server/cache

ENV PORT=5000
ENV CLIENT_URL=http://localhost:5173
ENV MONGODB_URI=mongodb://mongo:27017/plant-disease-playground
ENV PYTHON_EXECUTABLE=python3
ENV DATASET_ROOT=../data/dataset

EXPOSE 5000

CMD ["npm", "run", "start"]
