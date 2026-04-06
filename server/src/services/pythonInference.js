import { spawn } from "child_process";
import path from "path";

import { env } from "../config/env.js";
import { projectRoot } from "../utils/paths.js";

export function inferWithPython(imagePath) {
  return new Promise((resolve, reject) => {
    const workspaceRoot = path.resolve(projectRoot, "..");
    const scriptPath = path.join(projectRoot, "python", "infer.py");
    const datasetRoot = path.resolve(projectRoot, env.datasetRoot);
    const pythonExecutable =
      path.isAbsolute(env.pythonExecutable) || env.pythonExecutable.includes(path.sep)
        ? path.resolve(projectRoot, env.pythonExecutable)
        : env.pythonExecutable;
    const args = [scriptPath, "--image", imagePath, "--dataset-root", datasetRoot];

    const child = spawn(pythonExecutable, args, {
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
        reject(new Error(stderr || stdout || "Python inference failed."));
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
