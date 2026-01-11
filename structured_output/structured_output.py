import os
import pandas as pd
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from structured_output.classes import NanozymeExperiment, SeltoxExperiment, MagneticExperiment, CytotoxicityExperiment, SynergyExperiment
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)

dataset_map = {
    "nanozymes": NanozymeExperiment,
    "seltox": SeltoxExperiment,
    "magnetic": MagneticExperiment,
    "synergy": SynergyExperiment,
    "cytotoxicity": CytotoxicityExperiment
}

load_dotenv(override=True)

DEFAULT_MODEL = "openai/gpt-5.2"


def main() -> None:
    results_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

    llm = ChatOpenAI(
        model=os.getenv("STRUCTURED_OUTPUT_MODEL", DEFAULT_MODEL),
        openai_api_key=os.getenv("OPENROUTER_KEY"),
        openai_api_base=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )

    for dataset, experiment_class in dataset_map.items():
        folder_path = os.path.join(results_root, dataset)
        if not os.path.isdir(folder_path):
            continue

        print(f"{dataset} Processing {len(os.listdir(folder_path))} articles")

        class Response(BaseModel):
            experiments: Optional[list[experiment_class]]

        assistant_df_list = []

        md_files = [
            f for f in sorted(os.listdir(folder_path))
            if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".md")
        ]

        def process_article(article_name: str):
            file_path = os.path.join(folder_path, article_name)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()

                structured_llm = llm.with_structured_output(Response)

                # Retry up to 3 times for connection-related errors
                response = None
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        response = structured_llm.invoke(f"Text:\n{content}")
                        break
                    except Exception as e:
                        print(f"{dataset} API Error on {article_name} (attempt {attempt + 1}/{max_attempts}): {e}")
                        if attempt == max_attempts - 1:
                            # After exhausting retries, return empty-row DataFrame
                            field_names = list(getattr(experiment_class, "model_fields", {}).keys()) or list(getattr(experiment_class, "__fields__", {}).keys())
                            empty_row = {field_name: None for field_name in field_names}
                            article_df = pd.DataFrame([empty_row])
                            base_name, _ = os.path.splitext(article_name)
                            article_df["pdf"] = f"{base_name}.pdf"
                            return article_df

                response_experiments = response.experiments if response and response.experiments else []
                data = [experiment.dict() for experiment in response_experiments]
                if len(data) == 0:
                    print(f"{dataset} No experiments found for {article_name}")
                    field_names = list(getattr(experiment_class, "model_fields", {}).keys()) or list(getattr(experiment_class, "__fields__", {}).keys())
                    empty_row = {field_name: None for field_name in field_names}
                    data = [empty_row]

                article_df = pd.DataFrame(data)
                base_name, _ = os.path.splitext(article_name)
                article_df["pdf"] = f"{base_name}.pdf"
                return article_df
            except Exception as e:
                print(f"{dataset} Error processing {article_name}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_article, name) for name in md_files]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {dataset}", leave=False):
                result_df = future.result()
                if result_df is not None:
                    assistant_df_list.append(result_df)

        if len(assistant_df_list) == 0:
            continue

        assistant_df = pd.concat(assistant_df_list).reset_index(drop=True)
        output_csv = os.path.join(results_root, f"{dataset}_result.csv")
        assistant_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()