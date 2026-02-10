"""
WSGI config for chronobiotic project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/wsgi/
"""
"""
WSGI config for chronobiotic project.

Used for traditional HTTP servers (gunicorn/uwsgi).
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "chronobiotic.settings")

application = get_wsgi_application()