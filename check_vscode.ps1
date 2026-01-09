$procs = Get-Process Code -ErrorAction SilentlyContinue
$count = ($procs | Measure-Object).Count
$totalMB = [int](($procs | Measure-Object WorkingSet64 -Sum).Sum / 1MB)

Write-Output "VS Code: $count processes, $totalMB MB total"
Write-Output ""
Write-Output "Detail:"
$procs | Sort-Object WorkingSet64 -Descending | ForEach-Object {
    $mb = [int]($_.WorkingSet64/1MB)
    $title = if ($_.MainWindowTitle) { $_.MainWindowTitle } else { "(background)" }
    Write-Output "  $mb MB - $title"
}
