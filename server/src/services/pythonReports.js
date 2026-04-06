import { spawn } from "child_process";
import path from "path";

import { env } from "../config/env.js";
import { projectRoot } from "../utils/paths.js";

function resolvePythonExecutable() {
  return path.isAbsolute(env.pythonExecutable) || env.pythonExecutable.includes(path.sep)
    ? path.resolve(projectRoot, env.pythonExecutable)
    : env.pythonExecutable;
}

function runPythonScript(scriptName, args) {
  return new Promise((resolve, reject) => {
    const workspaceRoot = path.resolve(projectRoot, "..");
    const scriptPath = path.join(projectRoot, "python", scriptName);
    const child = spawn(resolvePythonExecutable(), [scriptPath, ...args], {
      cwd: workspaceRoot,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("error", (error) => {
      reject(new Error(`Python process failed to start: ${error.message}`));
    });

    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(stderr || stdout || "Python report failed."));
        return;
      }

      try {
        const payload = JSON.parse(stdout);
        if (payload.error) {
          reject(new Error(payload.error));
          return;
        }
        resolve(payload);
      } catch (error) {
        reject(new Error(`Invalid Python response: ${stdout || error.message}`));
      }
    });
  });
}

export function getImageReport(imagePath) {
  return runPythonScript("report_image.py", ["--image", imagePath]);
}

export function getTrainingReport() {
  const datasetRoot = path.resolve(projectRoot, env.datasetRoot);
  return runPythonScript("report_training.py", ["--dataset-root", datasetRoot]);
}
