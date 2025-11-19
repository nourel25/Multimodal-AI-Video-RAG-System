from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    
    MONGODB_URL: str
    MONGODB_DATABASE: str
    
    MAX_VIDEO_SIZE_MB: int
    
    VECTOR_DB_BACKEND : str
    VECTOR_DB_PATH : str
    VECTOR_DB_DISTANCE_METHOD: str = None
    
    class Config:
        env_file = ".env"
        

def get_settings():
    return Settings()