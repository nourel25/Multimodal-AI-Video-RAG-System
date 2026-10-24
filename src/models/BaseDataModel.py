from models.db_schemas.base import utc_now


class BaseDataModel:
    collection_name: str

    def __init__(self, db_client):
        self.db_client = db_client
        self.collection = db_client[self.collection_name]

    @staticmethod
    def now():
        return utc_now()
