$root = Split-Path -Parent $MyInvocation.MyCommand.Path

foreach ($split in @("train","validation","test")) {
    $dir = Join-Path $root $split
    $parts = Get-ChildItem -Path $dir -Filter ("{0}_*.jsonl" -f $split) | Sort-Object Name
    if (-not $parts) { throw "No shards found for $split" }

    $out = Join-Path $root ("{0}.jsonl" -f $split)
    $writer = [System.IO.StreamWriter]::new($out, $false, [System.Text.UTF8Encoding]::new($false))
    try {
        foreach ($part in $parts) {
            $writer.Write([System.IO.File]::ReadAllText($part.FullName, [System.Text.UTF8Encoding]::new($false)))
        }
    }
    finally { $writer.Dispose() }

    Write-Host ("{0} : {1} shards -> {0}.jsonl" -f $split, $parts.Count)
}
