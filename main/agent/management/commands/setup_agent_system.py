# main/management/commands/setup_agent_system.py
# Команда для инициализации агентной системы

from django.core.management.base import BaseCommand

from main.agent import agent_system


class Command(BaseCommand):
    help = 'Setup agent system and initialize all agents'
    
    def handle(self, *args, **options):
        self.stdout.write("Initializing agent system...")
        
        try:
            agent_system.initialize()
            status = agent_system.get_status()
            
            self.stdout.write(
                self.style.SUCCESS(
                    f"Agent system initialized successfully!\n"
                    f"Total agents: {status.get('total_agents', 0)}\n"
                    f"Active workers: {status.get('active_workers', 0)}"
                )
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"Failed to initialize agent system: {e}")
            )
