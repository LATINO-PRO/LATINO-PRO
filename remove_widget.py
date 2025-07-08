# strip_widgets.py
import nbformat as nbf, sys, pathlib

nb_path = pathlib.Path(sys.argv[1])
nb = nbf.read(nb_path, as_version=nbf.NO_CONVERT)

if "widgets" in nb.metadata:
    del nb.metadata["widgets"]
    nbf.write(nb, nb_path)
    print(f"Removed widgets metadata from {nb_path}")
else:
    print("No widgets metadata found; nothing to do.")
