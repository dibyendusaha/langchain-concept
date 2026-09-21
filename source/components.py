from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from utils import Utils


class ClassificationOutputModel(BaseModel):
    call_type: str = Field(description="Type of Cutomer Call")
    confidence_score: float = Field(description="Confidence Score between 0 and 1")

class Components(Utils):

    def __init__(self):
        super().__init__(temperature=0.2)


    def _classification(self, labels):
        llm = self._load_llm()
        parser = PydanticOutputParser(ClassificationOutputModel)

        prompt = PromptTemplate(
            """
                You're a call classification assistant.

                Classify the following customer support transcript into one of these categories:
                {labels}

                Transcript:
                {transcript}

                {format_instructions}
            """,
            input_variables=["transcript"],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
                "labels": labels
            }
        )

        df = self._load_file()