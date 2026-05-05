# chronobioticagent/chronobiotic/settings_dev.py
"""
Development settings for ChronobioticAgent
"""

from .settings import *

DEBUG = True
SECRET_KEY = 'django-insecure-dev-key-do-not-use-in-production'

ALLOWED_HOSTS = ['*']

# Database for development
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('DB_NAME', 'chronobiotic_dev'),
        'USER': os.environ.get('DB_USER', 'postgres'),
        'PASSWORD': os.environ.get('DB_PASSWORD', 'postgres'),
        'HOST': os.environ.get('DB_HOST', 'localhost'),
        'PORT': os.environ.get('DB_PORT', '5432'),
    }
}

# Cache for development
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': os.environ.get('REDIS_URL', 'redis://localhost:6379/1'),
    }
}

# Logging for development
LOGGING['root']['level'] = 'DEBUG'
LOGGING['loggers']['main']['level'] = 'DEBUG'

# Email for development
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'

# Debug toolbar
if DEBUG:
    INSTALLED_APPS.append('debug_toolbar')
    MIDDLEWARE.insert(0, 'debug_toolbar.middleware.DebugToolbarMiddleware')
    
    INTERNAL_IPS = [
        '127.0.0.1',
    ]

# CORS for development
CORS_ALLOW_ALL_ORIGINS = True

# REST Framework for development
REST_FRAMEWORK['DEFAULT_THROTTLE_RATES'] = {
    'anon': '1000/day',
    'user': '10000/day',
}

# Agent config for development
AGENT_CONFIG['enable_monitoring'] = True
AGENT_CONFIG['enable_tracing'] = True

# LLM config for development - use smaller models
LLM_CONFIG['default_model'] = 'gpt-3.5-turbo'
LLM_CONFIG['cache_enabled'] = False
