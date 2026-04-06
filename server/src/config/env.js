import dotenv from "dotenv";
import path from "path";

dotenv.config();

export const env = {
  port: Number(process.env.PORT || 5000),
  clientUrl: process.env.CLIENT_URL || "http://localhost:5173",
  mongoUri: process.env.MONGODB_URI || "mongodb://127.0.0.1:27017/plant-disease-playground",
  pythonExecutable: process.env.PYTHON_EXECUTABLE || path.join("..", "venv", "Scripts", "python.exe"),
  datasetRoot: process.env.DATASET_ROOT || "../data/dataset",
};
