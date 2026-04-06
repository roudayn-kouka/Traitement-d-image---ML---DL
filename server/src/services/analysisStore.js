import crypto from "crypto";

import { Analysis } from "../models/Analysis.js";

let persistenceMode = "memory";
const memoryAnalyses = [];

export function setPersistenceMode(mode) {
  persistenceMode = mode === "db" ? "db" : "memory";
}

export function getPersistenceMode() {
  return persistenceMode;
}

export async function listStoredAnalyses() {
  if (persistenceMode === "db") {
    return Analysis.find().sort({ createdAt: -1 }).lean();
  }

  return [...memoryAnalyses].sort((left, right) => {
    return new Date(right.createdAt).getTime() - new Date(left.createdAt).getTime();
  });
}

export async function createStoredAnalysis(payload) {
  if (persistenceMode === "db") {
    return Analysis.create(payload);
  }

  const timestamp = new Date().toISOString();
  const analysis = {
    _id: crypto.randomUUID(),
    createdAt: timestamp,
    updatedAt: timestamp,
    ...payload,
  };

  memoryAnalyses.unshift(analysis);
  return analysis;
}

export async function findStoredAnalysisById(id) {
  if (persistenceMode === "db") {
    return Analysis.findById(id).lean();
  }

  return memoryAnalyses.find((analysis) => analysis._id === id) || null;
}

export async function deleteStoredAnalysisById(id) {
  if (persistenceMode === "db") {
    return Analysis.findByIdAndDelete(id).lean();
  }

  const index = memoryAnalyses.findIndex((analysis) => analysis._id === id);
  if (index === -1) {
    return null;
  }

  const [deleted] = memoryAnalyses.splice(index, 1);
  return deleted;
}
