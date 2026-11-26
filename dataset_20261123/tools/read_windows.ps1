param([int]$Worker=0,[int]$Workers=1)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.Ocr.OcrEngine,Windows.Foundation,ContentType=WindowsRuntime]
$null = [Windows.Storage.StorageFile,Windows.Storage,ContentType=WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder,Windows.Foundation,ContentType=WindowsRuntime]
$null = [Windows.Globalization.Language,Windows.Globalization,ContentType=WindowsRuntime]
$asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and
    $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
} | Select-Object -First 1

function Wait-WinRt($Operation, [type]$Type) {
    $asTask.MakeGenericMethod($Type).Invoke($null,@($Operation)).GetAwaiter().GetResult()
}

$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage((New-Object Windows.Globalization.Language('ko-KR')))
if (!$engine) { throw 'Korean Windows OCR is unavailable.' }
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$dir = Join-Path $root 'audit/windows_ocr'
[IO.Directory]::CreateDirectory($dir) | Out-Null
$rows = @(Import-Csv -LiteralPath (Join-Path $root 'audit/ocr_queue.csv') -Encoding UTF8)
$done = 0
for ($i = $Worker; $i -lt $rows.Count; $i += $Workers) {
    $row = $rows[$i]
    $target = Join-Path $dir ($row.receipt_id + '.json')
    if ([IO.File]::Exists($target)) { continue }
    $path = [IO.Path]::GetFullPath((Join-Path $root $row.image))
    $file = Wait-WinRt ([Windows.Storage.StorageFile]::GetFileFromPathAsync($path)) ([Windows.Storage.StorageFile])
    $stream = Wait-WinRt ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    try {
        $decoder = Wait-WinRt ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
        $scale = [Math]::Min(1, [Windows.Media.Ocr.OcrEngine]::MaxImageDimension / [Math]::Max([double]$row.width, [double]$row.height))
        $transform = New-Object Windows.Graphics.Imaging.BitmapTransform
        $transform.ScaledWidth = [Math]::Round([double]$row.width * $scale)
        $transform.ScaledHeight = [Math]::Round([double]$row.height * $scale)
        $rotation = [double]0
        $result = $null
        for ($attempt = 0; $attempt -lt 4; $attempt++) {
            $degrees = @(0,180,90,270)[$attempt]
            if ($degrees) {
                $transform.Rotation = [Windows.Graphics.Imaging.BitmapRotation]("Clockwise$degrees")
            }
            $operation = $decoder.GetSoftwareBitmapAsync(
                [Windows.Graphics.Imaging.BitmapPixelFormat]::Bgra8,
                [Windows.Graphics.Imaging.BitmapAlphaMode]::Ignore, $transform,
                [Windows.Graphics.Imaging.ExifOrientationMode]::IgnoreExifOrientation,
                [Windows.Graphics.Imaging.ColorManagementMode]::DoNotColorManage)
            $bitmap = Wait-WinRt $operation ([Windows.Graphics.Imaging.SoftwareBitmap])
            try { $candidate = Wait-WinRt ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult]) } finally { $bitmap.Dispose() }
            if (!$result -or $candidate.Text.Length -gt $result.Text.Length) {
                $result = $candidate
                $rotation = $degrees
            }
            if ($attempt -eq 0 -and $result.Text.Length -ge 55 -and [Math]::Abs([double]$result.TextAngle) -lt 25) { break }
        }
        $lines = @(foreach ($line in $result.Lines) {
            $words = @(foreach ($word in $line.Words) {
                $box = $word.BoundingRect
                [ordered]@{ text = $word.Text; box = @($box.X, $box.Y, ($box.X + $box.Width), ($box.Y + $box.Height)) }
            })
            [ordered]@{ text = $line.Text; words = $words }
        })
        $record = [ordered]@{
            receipt_id = $row.receipt_id
            width = $transform.ScaledWidth
            height = $transform.ScaledHeight
            rotation = $rotation
            angle = [double]$result.TextAngle
            lines = $lines
        }
        $temporary = "$target.tmp"
        [IO.File]::WriteAllText($temporary,(ConvertTo-Json -InputObject $record -Depth 9 -Compress),(New-Object Text.UTF8Encoding($false)))
        [IO.File]::Move($temporary,$target)
        $done++
        if ($done % 25 -eq 0) { Write-Output "Worker $Worker prepared $done" }
    } finally { $stream.Dispose() }
}
Write-Output "Worker $Worker completed $done"
