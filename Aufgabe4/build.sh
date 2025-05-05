
## Alles in einem Rutsch (bash inside Container)

```bash
# 1. Anaconda-Image holen
docker pull continuumio/anaconda3

# 2. Container starten und Projekt einbinden
docker run -it \
-v "/c/Users/Robin/Workspace/HTWG/2D/2d-cv/Aufgabe4:/home/project" \
continuumio/anaconda3 bash

# --- Ab jetzt bist du im Container ---
# 3. In dein Projektverzeichnis wechseln
cd /home/project

# 4. System-Abhängigkeiten installieren
apt-get update
apt-get install -y build-essential python3-dev libeigen3-dev

# 5. Python-Bibliothek für PyBind11
pip install pybind11

# 6. C++-Extension kompilieren
c++ -O3 -Wall -shared -std=c++11 -fPIC \
  $(python3 -m pybind11 --includes) \
  sobel_demo.cpp \
  -o sobel_demo$(python3-config --extension-suffix)

# 7. Fertiges Modul überprüfen
ls sobel_demo*.so

# 8. Container verlassen (Module liegt nun im Windows-Ordner)
exit
