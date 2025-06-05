## Before compilation: compile cpp_mstar code

- cd into the od_mstar3 folder.
- python3 setup.py build_ext (may need --inplace as extra argument).
- copy so object from build/lib.*/ at the root of the od_mstar3 folder.
- Check by going back to the root of the git folder,
running python3 and "import cpp_mstar"