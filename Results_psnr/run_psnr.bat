@echo off
set output=all_results.txt
del %output% 2>nul

python .\psnr.py .\GT_00.png .\prop_scaled_loss.png %output%

echo All PSNR results saved to %output%
pause
