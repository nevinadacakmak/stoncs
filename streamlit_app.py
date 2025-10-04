"""
Stable Streamlit entrypoint wrapper for Stoncs.
Use this file as the Streamlit command / entrypoint on hosting:

    streamlit run streamlit_app.py

It imports the `app` module and invokes `main()` in a way that works whether
`app` is available as a top-level module or as `stoncs.app` inside a package.
"""

import importlib
import sys

def load_and_run():
    # Try to import top-level app module first
    try:
        mod = importlib.import_module('app')
    except Exception:
        # Fall back to package import
        try:
            mod = importlib.import_module('stoncs.app')
        except Exception as e:
            print('Failed to import app module:', e)
            raise
    if not hasattr(mod, 'main'):
        raise RuntimeError('Imported app module has no main()')
    return mod.main()

if __name__ == '__main__':
    load_and_run()
