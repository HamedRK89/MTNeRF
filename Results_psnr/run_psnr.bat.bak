@echo off
set output=random_results.txt
del %output% 2>nul

python .\psnr.py .\GT_00.png .\random_00.png %output%
python .\psnr.py .\GT_00.png .\100_00.png %output%
python .\psnr.py .\GT_00.png .\150_00.png %output%
python .\psnr.py .\GT_12.png .\random_12.png %output%
python .\psnr.py .\GT_12.png .\100_12.png %output%
python .\psnr.py .\GT_12.png .\150_12.png %output%

echo All PSNR results saved to %output%
pause
