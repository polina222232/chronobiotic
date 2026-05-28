"""
Pytest configuration and fixtures for Chronobiotic Agent tests
"""

import os
import pytest
from pathlib import Path

# Configure Django settings before importing Django modules
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chronobiotic.settings_test')

import django
from django.conf import settings

# Early configuration for pytest-django
if not settings.configured:
    django.setup()


@pytest.fixture(scope='session')
def django_db_setup():
    """Configure database for testing"""
    pass


@pytest.fixture
def sample_message():
    """Sample chat message for testing"""
    return {
        'message': 'What are chronobiotics?',
        'context': {}
    }


@pytest.fixture
def sample_chemical_data():
    """Sample chemical data for testing"""
    return {
        'name': 'Melatonin',
        'smiles': 'COc1ccc(CCN)cc1',
        'pubchem_cid': 896
    }


@pytest.fixture
def api_client():
    """Create a test API client"""
    from rest_framework.test import APIClient
    return APIClient()


@pytest.fixture
def test_user():
    """Create a test user"""
    from django.contrib.auth.models import User
    user, _ = User.objects.get_or_create(
        username='testuser',
        defaults={
            'email': 'test@example.com',
            'is_active': True
        }
    )
    return user
