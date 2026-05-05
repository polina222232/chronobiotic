# chronobioticagent/chronobiotic/wsgi.py
"""
WSGI config for ChronobioticAgent project.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chronobiotic.settings')

application = get_wsgi_application()