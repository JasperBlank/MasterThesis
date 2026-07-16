$ErrorActionPreference = "Stop"

$Root = "C:\Users\jjbla\OneDrive\Desktop\Masterproject\aprilgrid_blender"
$Blender = "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
$RenderScript = Join-Path $Root "render_aprilgrid_dataset.py"
$DetectScript = Join-Path $Root "detect_rendered_aprilgrid.py"

if (-not (Test-Path -LiteralPath $Blender)) {
    throw "Blender executable not found at $Blender"
}

& $Blender --background --python $RenderScript
python $DetectScript
