Write-Output "=== SYSTEM INFO ==="
$cs = Get-CimInstance Win32_ComputerSystem
Write-Output "Model: $($cs.Manufacturer) $($cs.Model)"
Write-Output "Total RAM: $([math]::Round($cs.TotalPhysicalMemory/1GB,1)) GB"

Write-Output ""
Write-Output "=== MEMORY NOW ==="
$os = Get-CimInstance Win32_OperatingSystem
$freeGB = [math]::Round($os.FreePhysicalMemory/1MB,1)
$totalGB = [math]::Round($cs.TotalPhysicalMemory/1GB,1)
$usedGB = [math]::Round($totalGB - $freeGB, 1)
Write-Output "Used: $usedGB GB / $totalGB GB"
Write-Output "Free: $freeGB GB"

Write-Output ""
Write-Output "=== TOP MEMORY PROCESSES ==="
Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 10 Name, @{N='MB';E={[int]($_.WorkingSet64/1MB)}} | Format-Table -AutoSize
