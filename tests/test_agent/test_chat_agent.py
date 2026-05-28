"""
Tests for Chat Agent API endpoints
"""

import pytest
from rest_framework import status


@pytest.mark.django_db
class TestChatAPI:
    """Test cases for chat API endpoints"""
    
    def test_chat_basic_request(self, api_client, sample_message):
        """Test basic chat request"""
        response = api_client.post('/api/v1/chat/', sample_message, format='json')
        
        assert response.status_code == status.HTTP_200_OK
        assert 'response' in response.data
        assert 'placeholder' in response.data.get('status', '')
    
    def test_chat_empty_message(self, api_client):
        """Test chat with empty message"""
        response = api_client.post('/api/v1/chat/', {'message': ''}, format='json')
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert 'error' in response.data
    
    def test_chat_missing_message(self, api_client):
        """Test chat with missing message field"""
        response = api_client.post('/api/v1/chat/', {}, format='json')
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_chat_stream_basic(self, api_client, sample_message):
        """Test streaming chat endpoint"""
        response = api_client.post('/api/v1/chat/stream/', sample_message, format='json')
        
        # Streaming response should return 200
        assert response.status_code == status.HTTP_200_OK
        assert response['Content-Type'] == 'text/event-stream'
    
    def test_chat_with_context(self, api_client):
        """Test chat request with context"""
        data = {
            'message': 'Explain chronobiology',
            'context': {
                'user_level': 'beginner',
                'language': 'en'
            }
        }
        
        response = api_client.post('/api/v1/chat/', data, format='json')
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.django_db
class TestChatIntegration:
    """Integration tests for chat functionality"""
    
    def test_chat_response_format(self, api_client, sample_message):
        """Test that chat response has correct format"""
        response = api_client.post('/api/v1/chat/', sample_message, format='json')
        
        assert response.status_code == status.HTTP_200_OK
        data = response.data
        
        # Check required fields
        assert 'response' in data
        assert 'citations' in data
        assert 'confidence' in data
        
        # Check types
        assert isinstance(data['response'], str)
        assert isinstance(data['citations'], list)
        assert isinstance(data['confidence'], (int, float))

