import os
from helpers.config import get_settings, Settings

class BaseController:
    def __init__(self):
        
        self.app_settings = get_settings()

        # Get project root directory (two levels up)
        self.base_dir = os.path.dirname(os.path.dirname(__file__))

        # Define audios directory
        self.audios_dir = os.path.join(self.base_dir, "assets", "audios")
        os.makedirs(self.audios_dir, exist_ok=True)
        
        # Define transcript directory
        self.transcripts_dir = os.path.join(self.base_dir, "assets", "transcripts")
        os.makedirs(self.audios_dir, exist_ok=True)
        
        # Define database directory
        self.database_dir = os.path.join(self.base_dir, "assets", "database")
        os.makedirs(self.audios_dir, exist_ok=True)
        
    def get_database_path(self, db_name: str):

        database_path = os.path.join(
            self.database_dir, db_name
        )

        if not os.path.exists(database_path):
            os.makedirs(database_path)

        return database_path