@echo off
set output=random_results.txt
del %output% 2>nul

python .\psnr.py .\GT_00.png .\00_baseline.png %output%
python .\psnr.py .\GT_00.png .\00_1.0.png %output%
python .\psnr.py .\GT_00.png .\00_0.9.png %output%
python .\psnr.py .\GT_00.png .\00_0.8.png %output%

python .\psnr.py .\GT_12.png .\12_baseline.png %output%
python .\psnr.py .\GT_12.png .\12_1.0.png %output%
python .\psnr.py .\GT_12.png .\12_0.9.png %output%
python .\psnr.py .\GT_12.png .\12_0.8.png %output%

echo All PSNR results saved to %output%
pause
