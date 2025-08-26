@echo off
set output=MH.txt
del %output% 2>nul

python .\psnr.py .\GT_00.png .\00_MH_0.9.png %output%

echo All PSNR results saved to %output%
pause
