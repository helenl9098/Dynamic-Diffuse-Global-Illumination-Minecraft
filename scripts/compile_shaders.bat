@echo off
for %%i in (*.vert *.frag *.comp) do "glslangValidator.exe" -V "%%~i" -o "%%~i.spv" -I"."
