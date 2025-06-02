import glob
import json
import os
from pathlib import Path
from time import time

import click
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, tool
from langchain_openai import ChatOpenAI
from data_preproccessing.pdf2txt import extract_text_from_pdf
from tqdm import tqdm
from graph_processing.image_extracting import pdf_analysis

from logger import LOGGER

load_dotenv(override=True)


def get_files_with_extension(directory, extension):
    return glob.glob(os.path.join(directory, f"*{extension}"))


@click.command()
@click.argument("pdf_articles_dir", type=click.Path())
@click.argument("pdf_supplements_dir", type=click.Path())
@click.argument("ner_json_dir", type=click.Path())
@click.argument("results_dir", type=click.Path())
@click.argument("dataset", type=click.Choice(["nanozymes", "seltox", "magnetic", "synergy", "cytotoxicity"]))
def main(
    pdf_articles_dir: str, pdf_supplements_dir: str, ner_json_dir: str, results_dir: str, dataset: str
):
    
    prompt = os.getenv(dataset.upper() + "PROMPT")
    directory = str(Path(pdf_articles_dir))

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
                "Extracts the minimum (Cmin) and maximum (Cmax) substrate concentrations from kinetic data on pages including pictures and graphs. Also extracts tables in markdown format. It uses GPT-4.1 to analyze images. Returns Cmin and Cmax for each page of the article and supplementary if provided, along with any tables found. You can pass any filename, it doesn't matter. You must use this tool every time."
                
                LOGGER.info(f"ANALYZE IMAGES TOOL article_file: {article_file}")
                LOGGER.info(f"ANALYZE IMAGES TOOL supplement_file: {supplement_file}")

                results_dict = {}

                results_dict["article"] = pdf_analysis(article_file)
                if supplement_file is not None:
                    results_dict["supplement"] = pdf_analysis(supplement_file)

                LOGGER.info("ANALYZE IMAGES TOOL FINISHED")
                
                return "```json\n" + str(results_dict) + "\n```"
            
            @tool("find_parameters")
            def find_parameters(file_name: str) -> str:
                "Extracts various parameters from the article using NER agent. You can pass any filename, it doesn't matter. You must use this tool every time. This tool may make mistakes, so you should use it only as an auxiliary tool that can help draw attention to the relevant values and experiments."

                json_file_path = os.path.join(ner_json_dir, f"{article_file.split("/")[-1][:-4]}.json")
                LOGGER.info(f"FIND PARAMETERS TOOL json_file_path: {json_file_path}")
                
                if os.path.exists(json_file_path):
                    with open(json_file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                else:
                    LOGGER.info(f"JSON file not found: {json_file_path}")
                    data = {}

                LOGGER.info("FIND PARAMETERS TOOL END")
                
                return "```json\n" + str(data) + "\n```"

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
