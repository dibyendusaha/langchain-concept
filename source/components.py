import pandas as pd

from tqdm import tqdm
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_core.output_parsers import PydanticOutputParser

from utils import Utils


class ClassificationOutputModel(BaseModel):
    call_type: str = Field(description="Type of Cutomer Call")
    confidence_score: float = Field(description="Confidence Score between 0 and 1")


class EvaluationOutputModel(BaseModel):
    score: float = Field(description="Score between 1 and 5")
    reasoning: str = Field(description="Explaination of the score")


class FinalReportOutputModel(BaseModel):
    summary: str = Field(description="Overall evaluation summary")
    recommendations: list[str] = Field(description="List of actionable improvements")


class Components(Utils):

    def __init__(self):
        super().__init__(temperature=0.2)


    def _classification(self, labels: list[str]) -> pd.DataFrame:
        llm = self._load_llm()
        parser = PydanticOutputParser(pydantic_object=ClassificationOutputModel)

        prompt = PromptTemplate(
            template="""
                You're a call classification assistant.

                Classify the following customer support transcript into one of these categories:
                {labels}

                Transcript:
                {transcript}

                {format_instructions}
            """,
            input_variables=["transcript"],
            partial_variables={
                "labels": labels,
                "format_instructions": parser.get_format_instructions()
            }
        )

        df = self._load_file()

        results = []

        for i, row in tqdm(iterable=df.iterrows(), total=len(df), desc="Running Classification"):
            try:
                prompt_chain = prompt | llm | parser

                result = prompt_chain.invoke({
                    "transcript": row["transcript"]
                })

                results.append({
                    "call_id": row["call_id"],
                    "predicted_call_type": result.call_type,
                    "confidence_score": result.confidence_score
                })

            except Exception as e:
                print(f"Error at row[{i}] -> {e}")

                results.append({
                    "call_id": row["call_id"],
                    "confidence_score": None,
                    "predicted_call_type": None
                })

        results_df = pd.DataFrame(data=results)
        df = df.merge(results_df, on="call_id")

        return df


    def _evaluation_plan(self, call_type: str) -> list[str]:
        call_type = call_type.lower()
        
        if call_type == "billing":
            return ["knowledge_accuracy", "resolution_quality"]

        elif call_type == "claims":
            return ["knowledge_accuracy", "resolution_quality"]

        elif call_type == "complaint":
            return ["tone_empathy", "resolution_quality"]

        elif call_type == "general_query":
            return ["knowledge_accuracy"]

        else:
            return ["knowledge_accuracy"]


    def _evaluation(self, df: pd.DataFrame) -> pd.DataFrame:
        df["evaluation_plan"] = df["predicted_call_type"].apply(self._evaluation_plan)

        llm = self._load_llm()
        parser = PydanticOutputParser(pydantic_object=EvaluationOutputModel)

        tone_prompt = PromptTemplate(
            template="""
                You are a QA evaluator for customer support calls.

                Evaluate the agent's tone and empathy in the following transcript.

                Consider:
                - Did the agent acknowledge the customer's issue?
                - Was the tone polite and professional?
                - Did the agent show empathy?

                Transcript:
                {transcript}

                {format_instructions}
            """,
            input_variables=["transcript"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            }
        )
        tone_chain = tone_prompt | llm | parser

        knowledge_prompt = PromptTemplate(
            template="""
                You are a QA evaluator for customer support calls.

                Evaluate the agent's knowledge accuracy and clarity.

                Consider:
                - Did the agent provide correct and relevant information?
                - Was the explanation clear and easy to understand?
                - Did the agent avoid vague or misleading statements?

                IMPORTANT:
                - If the transcript does not contain enough information, give a moderate score (2 or 3) and explain why.

                Transcript:
                {transcript}

                {format_instructions}
            """,
            input_variables=["transcript"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            }
        )
        knowledge_chain = knowledge_prompt | llm | parser

        resolution_prompt = PromptTemplate(
            template="""
                You are a QA evaluator for customer support calls.

                Evaluate the resolution quality of the agent.

                Consider:
                - Did the agent fully resolve the customer's issue?
                - Were next steps clearly communicated?
                - Did the agent confirm resolution before ending?

                Transcript:
                {transcript}

                {format_instructions}
            """,
            input_variables=["transcript"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            }
        )
        resolution_chain = resolution_prompt | llm | parser


        results = []

        for i, row in tqdm(iterable=df.iterrows(), total=len(df), desc="Running Evaluation"):
            eval_results = {}

            for eval_plan in row["evaluation_plan"]:
                try:
                    output = RunnableBranch(
                        (lambda _: "tone_empathy" in eval_plan, tone_chain),
                        (lambda _: "knowledge_accuracy" in eval_plan, knowledge_chain),
                        (lambda _: "resolution_quality" in eval_plan, resolution_chain),
                        knowledge_chain
                    )
                    eval_results[eval_plan] = output.model_dump()

                except Exception as e:
                    eval_results[eval_plan] = {"error": str(e)}

            results.append({
                "call_id": row["call_id"],
                "evaluation": eval_results
            })

        results_df = pd.DataFrame(results)
        df = df.merge(results_df, on="call_id")

        return df


    def _report_generation(self, df: pd.DataFrame) -> pd.DataFrame:
        llm = self._load_llm()
        parser = PydanticOutputParser(pydantic_object=FinalReportOutputModel)

        final_report_prompt = PromptTemplate(
            template="""
            """,
            input_variables=[],
            partial_variables={}
        )