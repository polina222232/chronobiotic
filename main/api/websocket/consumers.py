import json

from channels.generic.websocket import AsyncWebsocketConsumer

from main.agent.chronobiotics_agent import ChronobioticsAgent

agent = ChronobioticsAgent()


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = f'chat_{self.room_name}'
        
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.accept()
    
    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )
    
    async def receive(self, text_data):
        data = json.loads(text_data)
        message = data.get('message', '')
        
        response = agent.process_message(message)
        
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'message': response.get('text', ''),
                'citations': response.get('citations', [])
            }
        )
    
    async def chat_message(self, event):
        await self.send(text_data=json.dumps({
            'message': event['message'],
            'citations': event['citations']
        }))


class VoiceConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        self.group_name = f'voice_{self.session_id}'
        
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        await self.accept()
    
    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )
    
    async def receive(self, text_data):
        data = json.loads(text_data)
        
        if data.get('type') == 'transcription':
            await self.channel_layer.group_send(
                self.group_name,
                {
                    'type': 'voice_transcription',
                    'text': data.get('text', '')
                }
            )
    
    async def voice_transcription(self, event):
        await self.send(text_data=json.dumps({
            'type': 'transcription',
            'text': event['text']
        }))
