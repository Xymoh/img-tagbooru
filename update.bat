@echo off
cd /d "%~dp0"
echo ============================================
echo   Img-Tagbooru - Update from GitHub Releases
echo ============================================
echo.
echo Checking for latest release...

powershell -Command ^
  "$release = Invoke-RestMethod -Uri 'https://api.github.com/repos/Xymoh/img-tagbooru/releases/latest' -ErrorAction Stop; ^
   $asset = $release.assets | Where-Object { $_.name -eq 'img-tagger.exe' }; ^
   if ($asset) { ^
     Write-Host 'Latest release:' $release.tag_name; ^
     Write-Host 'Downloading' $asset.name '(' ([math]::Round($asset.size/1MB, 1)) 'MB)...'; ^
     Invoke-WebRequest -Uri $asset.browser_download_url -OutFile 'img-tagger.exe' -ErrorAction Stop; ^
     Write-Host ''; ^
     Write-Host '[OK] Updated to' $release.tag_name ^
   } else { ^
     Write-Host '[ERROR] No img-tagger.exe found in the latest release' ^
   }"

echo.
pause
