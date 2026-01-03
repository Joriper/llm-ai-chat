from pydantic import BaseModel
from enum import Enum

class AskingType(str,Enum):
    advantage="advantage/benifits"
    definition="definition"
    
class Queryhandler(BaseModel):
    query:str
    asking_type:AskingType = AskingType.definition
    