import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from structured_output.classes import NanozymeExperiment, SeltoxExperiment, MagneticExperiment, CytotoxicityExperiment, SynergyExperiment
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional
import click

pd.set_option("display.max_columns", None)

dataset_map = {
    "nanozymes": NanozymeExperiment,
    "seltox": SeltoxExperiment,
    "magnetic": MagneticExperiment,
    "synergy": SynergyExperiment,
    "cytotoxicity": CytotoxicityExperiment
}

load_dotenv(override=True)

@click.command()
@click.argument("agent_answers_dir", type=click.Path())
@click.argument("dataset", type=click.Choice(["nanozymes", "seltox", "magnetic", "synergy", "cytotoxicity"]))
def main(
    agent_answers_dir: str, dataset: str
):
    
    experiment_class = dataset_map[dataset]

    class Response(BaseModel):
        experiments: Optional[list[experiment_class]]

    assistant_df_list = []
    failed_md_list = []
    folder_path = agent_answers_dir
    for article_name in tqdm(os.listdir(folder_path)):
        with open(
            os.path.join(folder_path, article_name),
            "r",
            encoding="utf-8",
        ) as file:
            content = file.read()

        llm = ChatOpenAI(
            model="gpt-4.1",
            streaming=True,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )    

        structured_llm = llm.with_structured_output(Response)

        response = structured_llm.invoke(f"Text:\n{content}")

        response_experiments = response.experiments if response.experiments else []

        data = [experiment.dict() for experiment in response_experiments]
        if len(data) > 0:
            article_df = pd.DataFrame(data)
            article_df["pdf"] = article_name[:-3]+".pdf"
            assistant_df_list.append(article_df)

    assistant_df = pd.concat(assistant_df_list).reset_index(drop=True)

    assistant_df.to_csv(f"{dataset}.csv", index=False)


if __name__ == "__main__":
    main()