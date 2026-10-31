import os
import sys

# Make the repository root importable so that `import nn4pdes_advdiff` works
# when pytest is run from any directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
