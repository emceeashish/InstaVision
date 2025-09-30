#!/usr/bin/env python
import os
import sys
from pathlib import Path


def main():
    # Ensure project root is on sys.path so we can import `core` package
    current_file = Path(__file__).resolve()
    backend_dir = current_file.parent
    project_root = backend_dir.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and available on your PYTHONPATH environment variable?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
