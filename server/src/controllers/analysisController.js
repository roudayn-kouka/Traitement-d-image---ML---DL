import fs from "fs";
import path from "path";

import { inferWithPython } from "../services/pythonInference.js";
import { getImageReport, getTrainingReport } from "../services/pythonReports.js";
import {
  createStoredAnalysis,
  deleteStoredAnalysisById,
  findStoredAnalysisById,
  getPersistenceMode,
  listStoredAnalyses,
} from "../services/analysisStore.js";
import { uploadsDirectory } from "../utils/paths.js";

export async function getHealth(req, res) {
  res.json({
    status: "ok",
    persistence: getPersistenceMode(),
  });
}

export async function listAnalyses(req, res) {
  const analyses = await listStoredAnalyses();
  res.json(analyses);
}

export async function createAnalysis(req, res) {
  if (!req.file) {
    return res.status(400).json({ message: "Image file is required." });
  }

  try {
    const category = (req.body.category || "").trim();
    const uploadPath = path.join(uploadsDirectory, req.file.filename);
    const result = await inferWithPython(uploadPath);

    const analysis = await createStoredAnalysis({
      originalName: req.file.originalname,
      storedName: req.file.filename,
      imageUrl: `/uploads/${req.file.filename}`,
      mimeType: req.file.mimetype,
      size: req.file.size,
      category: category || result.diagnosis,
      diagnosis: result.diagnosis,
      confidence: result.confidence,
      healthStatus: result.health_status,
      notes: result.notes,
      inferenceMode: result.mode,
      stats: result.stats,
      trainingSamples: result.training_samples,
      knownLabels: result.known_labels,
    });

    res.status(201).json(analysis);
  } catch (error) {
    if (req.file) {
      const uploadPath = path.join(uploadsDirectory, req.file.filename);
      if (fs.existsSync(uploadPath)) {
        fs.unlinkSync(uploadPath);
      }
    }
    throw error;
  }
}

export async function deleteAnalysis(req, res) {
  const analysis = await findStoredAnalysisById(req.params.id);

  if (!analysis) {
    return res.status(404).json({ message: "Analysis not found." });
  }

  const uploadPath = path.join(uploadsDirectory, analysis.storedName);
  if (fs.existsSync(uploadPath)) {
    fs.unlinkSync(uploadPath);
  }

  await deleteStoredAnalysisById(req.params.id);
  res.status(204).send();
}

export async function getAnalysisReport(req, res) {
  const analysis = await findStoredAnalysisById(req.params.id);

  if (!analysis) {
    return res.status(404).json({ message: "Analysis not found." });
  }

  const uploadPath = path.join(uploadsDirectory, analysis.storedName);
  const report = await getImageReport(uploadPath);
  res.json({
    analysis,
    report,
  });
}

export async function getDatasetTrainingReport(req, res) {
  const report = await getTrainingReport();
  res.json(report);
}
