import os
import time
import pandas as pd

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


class Utils:

    def __init__(self, provider="openai", temperature=0):
        load_dotenv()
        self.__provider = provider.lower()
        self.__temperature = float(temperature)


    def _load_file(self, path="./data/transcripts.xlsx") -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File does not exists for the path -> {path}")

        return pd.read_excel(path)


    def _save_file(self, df: pd.DataFrame, path="./data/output.xlsx") -> None:
        df.to_excel(path, index=False)


    def _load_llm(self) -> ChatOpenAI | ChatGoogleGenerativeAI:
        if self.__provider == "openai":
            return ChatOpenAI(
                model="gpt-4o-mini",
                temperature=self.__temperature
            )
        elif self.__provider == "google" or self.__provider == "gemini":
            return ChatGoogleGenerativeAI(
                model="gemini-3.5-flash",
                temperature=self.__temperature
            )
        else:
            raise ValueError(f"Invalid LLM Provider -> {self.__provider}")