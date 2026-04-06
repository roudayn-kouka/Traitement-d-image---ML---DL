import { Router } from "express";
import multer from "multer";
import path from "path";
import fs from "fs";

import {
  createAnalysis,
  deleteAnalysis,
  getAnalysisReport,
  getDatasetTrainingReport,
  getHealth,
  listAnalyses,
} from "../controllers/analysisController.js";
import { asyncHandler } from "../utils/asyncHandler.js";
import { uploadsDirectory } from "../utils/paths.js";
fs.mkdirSync(uploadsDirectory, { recursive: true });

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadsDirectory);
  },
  filename: (req, file, cb) => {
    const extension = path.extname(file.originalname).toLowerCase();
    const safeBaseName = path
      .basename(file.originalname, extension)
      .replace(/[^a-z0-9_-]+/gi, "-")
      .replace(/^-+|-+$/g, "")
      .toLowerCase();
    cb(null, `${Date.now()}-${safeBaseName || "leaf"}${extension}`);
  },
});

const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    const allowedExtensions = new Set([".jpg", ".jpeg", ".png", ".bmp", ".webp"]);
    const extension = path.extname(file.originalname).toLowerCase();

    if (!allowedExtensions.has(extension)) {
      return cb(new Error("Only image uploads are allowed."));
    }

    cb(null, true);
  },
  limits: {
    fileSize: 5 * 1024 * 1024,
  },
});

export const analysisRouter = Router();

analysisRouter.get("/health", asyncHandler(getHealth));
analysisRouter.get("/analyses", asyncHandler(listAnalyses));
analysisRouter.get("/analyses/:id/report", asyncHandler(getAnalysisReport));
analysisRouter.post("/analyses", upload.single("image"), asyncHandler(createAnalysis));
analysisRouter.delete("/analyses/:id", asyncHandler(deleteAnalysis));
analysisRouter.get("/training-report", asyncHandler(getDatasetTrainingReport));
