import time
import pandas as pd

from tqdm import tqdm
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_core.output_parsers import PydanticOutputParser

from .utils import Utils


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

    def __init__(self, provider="openai"):
        super().__init__(provider=provider, temperature=0.2)
        print("🚀 Components has been initialised successfully")


    def _classification(self, df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
        start_time = time.time()

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

        results = []

        for i, row in tqdm(iterable=df.iterrows(), total=len(df), desc="Generating Classification"):
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
                print(f"⚠️ Error at row[{i}] -> {e}")

                results.append({
                    "call_id": row["call_id"],
                    "confidence_score": None,
                    "predicted_call_type": None
                })

        results_df = pd.DataFrame(data=results)
        df = df.merge(results_df, on="call_id")

        duration = (time.time() - start_time) * 1000
        print(f"⌛ Time taken to run the calssification job: {duration} ms")

        print("🎉 Generated classification for the provided labels")
        return df


    def __evaluation_plan(self, call_type: str) -> list[str]:
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
        start_time = time.time()

        df["evaluation_plan"] = df["predicted_call_type"].apply(self.__evaluation_plan)

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

        for i, row in tqdm(iterable=df.iterrows(), total=len(df), desc="Generating Evaluation"):
            eval_results = {}

            for eval_plan in row["evaluation_plan"]:
                try:
                    eval_response = RunnableBranch(
                        (lambda _: "tone_empathy" in eval_plan, tone_chain),
                        (lambda _: "knowledge_accuracy" in eval_plan, knowledge_chain),
                        (lambda _: "resolution_quality" in eval_plan, resolution_chain),
                        knowledge_chain
                    ).invoke({
                        "transcript": row["transcript"]
                    })
                    eval_results[eval_plan] = eval_response.model_dump()

                except Exception as e:
                    print(f"⚠️ Error at row[{i}] -> {e}")

                    eval_results[eval_plan] = {"error": str(e)}

            results.append({
                "call_id": row["call_id"],
                "evaluation": eval_results
            })

        results_df = pd.DataFrame(results)
        df = df.merge(results_df, on="call_id")

        duration = (time.time() - start_time) * 1000
        print(f"⌛ Time taken to run the evaluation plan: {duration} ms")

        print("🎉 Generated evaluation plan for the provided transcripts")
        return df


    def _report_generation(self, df: pd.DataFrame) -> pd.DataFrame:
        start_time = time.time()

        llm = self._load_llm()
        parser = PydanticOutputParser(pydantic_object=FinalReportOutputModel)

        final_report_prompt = PromptTemplate(
            template="""
                You are a QA manager reviewing customer support calls.

                Based on the evaluation results below, generate:

                1. A concise summary of the agent's performance
                2. A list of actionable recommendations for improvement

                Evaluation Data:
                {evaluation}

                IMPORTANT:
                - Be specific and practicals
                - Do not repeat scores
                - Focus on improvement

                {format_instructions}
            """,
            input_variables=["evaluation"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            }
        )
        final_report_chain = final_report_prompt | llm | parser


        results = []

        for i, row in tqdm(iterable=df.iterrows(), total=len(df), desc="Generating Final report"):
            try:
                response = final_report_chain.invoke({
                    "evaluation": row["evaluation"]
                })

                result = response.model_dump()

                results.append({
                    "call_id": row["call_id"],
                    "summary": result.get("summary"),
                    "recommendations": result.get("recommendations")
                })

            except Exception as e:
                print(f"⚠️ Error at row[{i}] -> {e}")

                results.append({
                    "call_id": row["call_id"],
                    "summary": None,
                    "recommendations": None
                })

        results_df = pd.DataFrame(results)
        df = df.merge(results_df, on="call_id")

        duration = (time.time() - start_time) * 1000
        print(f"⌛ Time taken to run the final reporting: {duration} ms")

        print("🎉 Generated final report with recommendations and summary")
        return df