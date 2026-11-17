import threading

# Mehrere Sitzungen laden Modelle sonst gleichzeitig; transformers/accelerate patchen dabei global und erzeugen "meta tensor"-Fehler.
MODEL_LOAD_LOCK = threading.RLock()
