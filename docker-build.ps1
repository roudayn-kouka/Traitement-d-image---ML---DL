$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$stageRoot = Join-Path $projectRoot ".docker-context"

if (Test-Path -LiteralPath $stageRoot) {
    $resolvedStageRoot = (Resolve-Path -LiteralPath $stageRoot).Path
    if (-not $resolvedStageRoot.StartsWith($projectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove unexpected staging path: $resolvedStageRoot"
    }

    Remove-Item -LiteralPath $stageRoot -Recurse -Force
}

New-Item -ItemType Directory -Path $stageRoot | Out-Null

$itemsToStage = @(
    ".dockerignore",
    "docker-compose.yml",
    "Dockerfile.client",
    "Dockerfile.server",
    "client",
    "server",
    "plant_disease_app",
    "data"
)

foreach ($item in $itemsToStage) {
    Copy-Item -LiteralPath (Join-Path $projectRoot $item) -Destination (Join-Path $stageRoot $item) -Recurse -Force
}

@(
    (Join-Path $stageRoot "client\\Dockerfile"),
    (Join-Path $stageRoot "server\\Dockerfile"),
    (Join-Path $stageRoot "client\\client.out.log"),
    (Join-Path $stageRoot "client\\client.err.log"),
    (Join-Path $stageRoot "server\\server.out.log"),
    (Join-Path $stageRoot "server\\server.err.log")
) | ForEach-Object {
    if (Test-Path -LiteralPath $_) {
        Remove-Item -LiteralPath $_ -Force
    }
}

Push-Location $stageRoot
try {
    docker compose up --build -d
}
finally {
    Pop-Location
}
