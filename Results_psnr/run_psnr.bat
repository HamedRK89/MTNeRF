@echo off
set output=main_2_results.txt
del %output% 2>nul

python .\psnr.py .\GT_00.png .\000.png %output%


echo All PSNR results saved to %output%
pause
