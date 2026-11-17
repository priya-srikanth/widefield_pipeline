param([string]$Deck, [string]$OutDir, [string]$Slides)
# Export chosen slides of a deck to PNG via PowerPoint COM (read-only open, no window). For visual QC of the swap pass.
if ($Deck -like "*presentations*") { throw "export a scratch COPY, never the live deck (COM attaches to an already-open copy and Close() closes it)" }
New-Item -ItemType Directory -Force $OutDir | Out-Null
$app = New-Object -ComObject PowerPoint.Application
$pres = $app.Presentations.Open($Deck, $true, $false, $false)
foreach ($n in $Slides.Split(",")) {
    $i = [int]$n
    $pres.Slides.Item($i).Export((Join-Path $OutDir ("s{0:D3}.png" -f $i)), "PNG", 1600, 900)
}
$pres.Close()
