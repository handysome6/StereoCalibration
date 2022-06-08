from pathlib import Path
import os

p = Path('.')
files = [f for f in p.glob('**/*.jpg')]

os.chdir("scenes/")
for i in range(len(files)):
    f = files[i].name
    os.rename(f, f"sbs_{str(i+1).zfill(2)}.jpg")