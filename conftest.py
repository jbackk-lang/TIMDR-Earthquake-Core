"""
conftest.py — pusty celowo. Jego jedyna rola to zagwarantowanie, że
pytest wstawi katalog główny repo (ten, w którym leży ten plik) na
sys.path, niezależnie od tego, jak pytest zostanie uruchomiony (`pytest`,
`python -m pytest`, z innego CWD) - tak, żeby `tests/*.py`'s `from
core.X import Y` zawsze się rozwiązywało, bez polegania na przypadkowym
zachowaniu `-m` (które dodaje CWD do sys.path samo, ale zwykły `pytest`
tego nie robi).
"""
