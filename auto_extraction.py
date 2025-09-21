import glob
import json
import os
from pathlib import Path
from time import time

import click
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, tool
from langchain_openai import ChatOpenAI
from data_preprocessing.pdf2txt import extract_text_from_pdf
from tqdm import tqdm

from logger import LOGGER
from vision_api_client import VisionAPIClient
from ner_api_client import NERAPIClient

load_dotenv(override=True)


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _map_dataset_for_api(dataset: str) -> str:
    # Vision and NER APIs use "cytox" instead of "cytotoxicity"
    return "cytox" if dataset == "cytotoxicity" else dataset


def get_files_with_extension(directory, extension):
    return glob.glob(os.path.join(directory, f"*{extension}"))


@click.command()
@click.argument("pdf_articles_dir", type=click.Path())
@click.argument("pdf_supplements_dir", type=click.Path())
@click.argument("results_dir", type=click.Path())
@click.argument("dataset", type=click.Choice(["nanozymes", "seltox", "magnetic", "synergy", "cytotoxicity"]))
def main(
    pdf_articles_dir: str, pdf_supplements_dir: str, results_dir: str, dataset: str
):
    
    # Load prompt from prompts/<dataset>.txt
    prompt_path = PROMPTS_DIR / f"{dataset}.txt"
    if not prompt_path.exists():
        LOGGER.error(f"Prompt file '{prompt_path}' not found for dataset '{dataset}'.")
        return
    try:
        with open(prompt_path, "r", encoding="utf-8") as pf:
            prompt = pf.read()
    except Exception as e:
        LOGGER.error(f"Failed to read prompt for dataset '{dataset}': {e}")
        return

    directory = str(Path(pdf_articles_dir))
    os.makedirs(results_dir, exist_ok=True)

    # Initialize API clients (override via env if provided)
    vision_client = VisionAPIClient(base_url=os.getenv("VISION_API_URL", "http://77.234.216.102:17628"))
    ner_client = NERAPIClient(base_url=os.getenv("NER_API_URL", "http://77.234.216.102:17629"))
    api_dataset = _map_dataset_for_api(dataset)

    extension = ".pdf"
    articles_files = get_files_with_extension(directory, extension)
    LOGGER.info(f"Files count: {len(articles_files)}")

    for article_file in tqdm(articles_files):
        try:
            start_time = time()
            LOGGER.info(
                f"Agent initialization start: {article_file[len(directory)+1:]}"
            )

            text_dict = {}

            text_dict["article_text"] = extract_text_from_pdf(article_file)

            supplement_file = None
            si_file_path = f"{pdf_supplements_dir}/{article_file[len(directory)+1:]}"
            if os.path.isfile(si_file_path):
                supplement_file = si_file_path
            text_dict["supplement_text"] = (
                extract_text_from_pdf(supplement_file) if supplement_file else None
            )

            @tool("get_full_text")
            def get_full_text(query: str) -> str:
                "Returns full text of the article and supplement information if provided. You can pass any query, it doesn't matter. You must use this tool every time."

                full_text_dict = {}
                full_text_dict["article_text"] = text_dict["article_text"]
                full_text_dict["supplement_text"] = text_dict["supplement_text"]

                return "```json\n" + str(full_text_dict) + "\n```"
            
            @tool("analyze_images")
            def analyze_images(file_name: str) -> str:
                "Extracts data from pages with figures/tables using the Vision API (Cmin/Cmax, tables, etc.). Returns JSON with 'article' and optional 'supplement' results."
                
                LOGGER.info(f"ANALYZE IMAGES TOOL article_file: {article_file}")
                LOGGER.info(f"ANALYZE IMAGES TOOL supplement_file: {supplement_file}")

                results_dict = {}
                try:
                    results_dict["article"] = vision_client.extract_file(
                        file_path=article_file,
                        dataset_type=api_dataset,
                        use_vlm=True,
                        max_pages=0,
                    )
                except Exception as e:
                    LOGGER.error(f"Vision API error (article): {e}")
                    results_dict["article"] = {"error": str(e)}

                if supplement_file is not None:
                    try:
                        results_dict["supplement"] = vision_client.extract_file(
                            file_path=supplement_file,
                            dataset_type=api_dataset,
                            use_vlm=True,
                            max_pages=0,
                        )
                    except Exception as e:
                        LOGGER.error(f"Vision API error (supplement): {e}")
                        results_dict["supplement"] = {"error": str(e)}

                LOGGER.info("ANALYZE IMAGES TOOL FINISHED")
                
                return "```json\n" + json.dumps(results_dict, ensure_ascii=False) + "\n```"
            
            @tool("find_parameters")
            def find_parameters(file_name: str) -> str:
                "Extracts parameters from the article using the NER API. Returns final task results JSON."

                LOGGER.info("FIND PARAMETERS TOOL start (NER API)")
                try:
                    initial_result = ner_client.extract_entities(
                        file_path=article_file,
                        extraction_type=api_dataset,
                        model_names=None,
                        max_pages=0,
                    )
                except Exception as e:
                    LOGGER.error(f"NER API initial request error: {e}")
                    return "```json\n" + json.dumps({"error": str(e)}, ensure_ascii=False) + "\n```"

                task_id = None
                if isinstance(initial_result, dict):
                    task_id = initial_result.get("task_id") or initial_result.get("id")

                final_result = initial_result
                if task_id:
                    try:
                        final_result = ner_client.wait_for_completion(task_id, check_interval=45, max_wait=5400)
                    except Exception as e:
                        LOGGER.error(f"NER API wait error: {e}")
                        final_result = {"error": str(e), "task_id": task_id}

                LOGGER.info("FIND PARAMETERS TOOL END")
                
                return "```json\n" + json.dumps(final_result, ensure_ascii=False) + "\n```"

            tools = [get_full_text, analyze_images, find_parameters]

            agent_llm = ChatOpenAI(
                temperature=0,
                model="gpt-4.1",
                streaming=True,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
        
            agent = initialize_agent(
                tools,
                agent_llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                agent_kwargs={"prefix": prompt, "seed": 42},
            )

            LOGGER.info(
                f"Successful Agent initialization: {article_file[len(directory)+1:]}"
            )

            user_prompt = (
                "get all parameters, use all available tools"
            )
            for i in range(5):
                try:
                    response = agent.run(user_prompt)
                    with open(
                        f"{results_dir}/{article_file[len(directory)+1:-4]}.md",
                        "w",
                    ) as f:
                        f.write(response)
                    break
                except Exception as e:
                    LOGGER.error(e)
                    with open(
                        f"{results_dir}/{article_file[len(directory)+1:-4]}.md",
                        "w",
                    ) as f:
                        f.write(f"Error: \n{str(e)}")

            end_time = time()

            LOGGER.info(
                f"Article {article_file[len(directory)+1:]} was processed in {end_time-start_time:.2f}s"
            )
            LOGGER.info("")
        except Exception as e:
            LOGGER.error(e)
            with open(
                f"{results_dir}/{article_file[len(directory)+1:-4]}.md", "w"
            ) as f:
                f.write(f"Error: \n{str(e)}")


if __name__ == "__main__":
    main()
