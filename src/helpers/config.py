from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    
    MONGODB_URL: str
    MONGODB_DATABASE: str
    
    MAX_VIDEO_SIZE_MB: int
    
    class Config:
        env_file = ".env"
        

def get_settings():
    return Settings()