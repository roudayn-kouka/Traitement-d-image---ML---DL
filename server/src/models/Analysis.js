import mongoose from "mongoose";

const analysisSchema = new mongoose.Schema(
  {
    originalName: {
      type: String,
      required: true,
      trim: true,
    },
    storedName: {
      type: String,
      required: true,
      trim: true,
    },
    imageUrl: {
      type: String,
      required: true,
      trim: true,
    },
    mimeType: {
      type: String,
      default: "",
    },
    size: {
      type: Number,
      default: 0,
    },
    category: {
      type: String,
      default: "unclassified",
      trim: true,
    },
    diagnosis: {
      type: String,
      required: true,
      trim: true,
    },
    confidence: {
      type: Number,
      required: true,
      min: 0,
      max: 1,
    },
    healthStatus: {
      type: String,
      enum: ["healthy", "warning", "critical"],
      required: true,
    },
    notes: {
      type: [String],
      default: [],
    },
    inferenceMode: {
      type: String,
      enum: ["ml", "heuristic"],
      default: "heuristic",
    },
    stats: {
      type: Object,
      default: {},
    },
    trainingSamples: {
      type: Number,
      default: 0,
    },
    knownLabels: {
      type: [String],
      default: [],
    },
  },
  {
    timestamps: true,
  },
);

export const Analysis = mongoose.model("Analysis", analysisSchema);
