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


    def _load_file(self, path=None) -> pd.DataFrame:
        start_time = time.time()
        
        if path is None:
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "transcripts.xlsx")

        if not os.path.exists(path):
            raise FileNotFoundError(f"File does not exists for the path -> {path}")

        duration = (time.time() - start_time) * 1000
        print(f"⌛ Time taken to load the file: {duration} ms")

        return pd.read_excel(path)


    def _save_file(self, df: pd.DataFrame, path="./data/output.xlsx") -> None:
        start_time = time.time()

        df.to_excel(path, index=False)

        duration = (time.time() - start_time) * 1000
        print(f"⌛ Time taken to save the file: {duration} ms")


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