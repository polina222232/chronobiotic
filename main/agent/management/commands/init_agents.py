# chronobioticagent/main/management/commands/init_agents.py
"""
Initialize the agent system
"""

import logging

from django.core.management.base import BaseCommand

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Initialize the Chronobiotic Agent System'
    
    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Initializing Chronobiotic Agent System...'))
        
        try:
            # Simple initialization without async
            from main.agent.agent_manager import AgentManager
            
            manager = AgentManager()
            
            self.stdout.write(self.style.SUCCESS('✓ Agent Manager initialized'))
            self.stdout.write(f'✓ Agent Manager ready with {len(manager._agents)} agents')
            
            self.stdout.write(self.style.SUCCESS('\nAgent system initialized successfully!'))
        
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Failed to initialize agent system: {e}'))
            import traceback
            traceback.print_exc()
