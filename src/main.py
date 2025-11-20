from fastapi import FastAPI
from routes import base, ingest
from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings
from contextlib import asynccontextmanager
from stores.vectordb.VectorDPBroviderFactory import VectorDBProviderFactory


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL)
    app.db_client = app.mongo_conn[settings.MONGODB_DATABASE]
    
    vectordb_provider_factory = VectorDBProviderFactory(settings)
    
    app.vector_client = vectordb_provider_factory.create(
        provider=settings.VECTOR_DB_BACKEND
    )
    app.vector_client.connect()
    
    yield
    
    app.mongo_conn.close()
    app.vector_client.disconnect()

app = FastAPI(lifespan=lifespan)

app.include_router(base.base_router)
app.include_router(ingest.ingest_router)