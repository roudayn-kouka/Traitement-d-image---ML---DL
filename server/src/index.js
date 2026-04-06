import express from "express";
import mongoose from "mongoose";
import cors from "cors";
import morgan from "morgan";

import { env } from "./config/env.js";
import { analysisRouter } from "./routes/analysisRoutes.js";
import { errorHandler, notFoundHandler } from "./middleware/errorHandler.js";
import { setPersistenceMode } from "./services/analysisStore.js";
import { uploadsDirectory } from "./utils/paths.js";

const app = express();

app.use(
  cors({
    origin: env.clientUrl,
  }),
);
app.use(morgan("dev"));
app.use(express.json());
app.use("/uploads", express.static(uploadsDirectory));
app.use("/api", analysisRouter);
app.use(notFoundHandler);
app.use(errorHandler);

async function startServer() {
  let persistenceMode = "memory";

  try {
    await mongoose.connect(env.mongoUri);
  } catch (error) {
    console.warn(`MongoDB unavailable, starting with in-memory storage: ${error.message}`);
  }

  if (mongoose.connection.readyState === 1) {
    persistenceMode = "db";
  }

  setPersistenceMode(persistenceMode);

  app.listen(env.port, () => {
    console.log(`Server listening on http://localhost:${env.port} (${persistenceMode})`);
  });
}

startServer();
