$ErrorActionPreference = "Stop"

$Root = "C:\Users\jjbla\OneDrive\Desktop\Masterproject\aprilgrid_blender"
$Blender = "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"

if (-not (Test-Path -LiteralPath $Blender)) {
    throw "Blender executable not found at $Blender"
}

python (Join-Path $Root "generate_degradation_textures.py")
& $Blender --background --python (Join-Path $Root "render_degradation_matrix.py")
python (Join-Path $Root "postprocess_degradation_matrix.py")
python (Join-Path $Root "detect_degradation_matrix.py")
