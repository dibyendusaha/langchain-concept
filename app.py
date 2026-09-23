import os
import time
import uuid
import pandas as pd

from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from source.components import Components

os.makedirs("data", exist_ok=True)

app = FastAPI(
    title="Customer Support QA Evaluator",
    version="0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

@app.get("/")
def health_check():
    return {"message": "QA Evuation API is running"}


@app.post("/evaluate")
async def evaluate(file: UploadFile = File(..., description=".xlsx/.csv file which will contain call_id, agent_name, transcript, expected_call_type")) -> FileResponse:
    start_time = time.time()

    allowed_extensions = [".csv", ".xlsx"]

    filename, extension = os.path.splitext(file.filename)

    if extension not in allowed_extensions:
        return {"error": "Please upload either a .csv or .xlsx file"}

    temp_file_name = uuid.uuid4()
    input_path = f"data/{temp_file_name}{extension}"
    output_path = f"data/{filename}_{temp_file_name}.xlsx"

    with open(input_path, mode="wb") as f:
        content = await file.read()
        f.write(content)

    if extension == '.csv':
        df = pd.read_csv(input_path)

    elif extension == ".xlsx":
        df = pd.read_excel(input_path)


    labels = ["billing", "claims", "complaint", "general_query"]

    x = Components()

    cdf = x._classification(df=df, labels=labels)

    edf = x._evaluation(df=cdf)

    final_df = x._report_generation(edf)

    final_df.to_excel(output_path, index=False)

    duration = (time.time() - start_time) * 1000
    print(f"🕛 Time taken to generate the final report: {duration} ms")

    return FileResponse(
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="evaluate_qa.xlsx",
        path=output_path
    )