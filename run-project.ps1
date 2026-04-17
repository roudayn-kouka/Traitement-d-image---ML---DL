$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Test-PortOpen {
  param([int]$Port)

  $client = New-Object System.Net.Sockets.TcpClient
  try {
    $async = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
    $connected = $async.AsyncWaitHandle.WaitOne(500)
    if (-not $connected) {
      return $false
    }

    $client.EndConnect($async)
    return $true
  } catch {
    return $false
  } finally {
    $client.Dispose()
  }
}

if (-not (Test-PortOpen -Port 5000)) {
  Start-Process -FilePath "cmd.exe" -ArgumentList "/c start ""plant-server"" cmd /c ""cd /d """"$projectRoot\server"""" && npm.cmd run dev""" -WorkingDirectory $projectRoot
  Start-Sleep -Seconds 2
} else {
  Write-Host "Server already running on http://localhost:5000"
}

if (-not (Test-PortOpen -Port 5173)) {
  Start-Process -FilePath "cmd.exe" -ArgumentList "/c start ""plant-client"" cmd /c ""cd /d """"$projectRoot\client"""" && npm.cmd run dev -- --host 127.0.0.1""" -WorkingDirectory $projectRoot
} else {
  Write-Host "Client already running on http://127.0.0.1:5173"
}

Write-Host "Server: http://localhost:5000"
Write-Host "Client: http://127.0.0.1:5173"
