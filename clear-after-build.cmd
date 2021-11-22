rem rd ".vs" /S/Q

for /f "usebackq" %%f in (`"dir /ad/b/s obj"`) do rd "%%f" /S/Q
for /f "usebackq" %%f in (`"dir /ad/b/s bin"`) do rd "%%f" /S/Q

rem pause