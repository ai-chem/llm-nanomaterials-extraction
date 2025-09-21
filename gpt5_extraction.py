import glob
import os
from pathlib import Path
from time import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from logger import LOGGER
from data_preprocessing.pdf2txt import extract_text_from_pdf


load_dotenv(override=True)


def get_files_with_extension(directory, extension):
    return glob.glob(os.path.join(directory, f"*{extension}"))


DATASET_TO_DIR = {
    "nanozymes": "test_data/nanozymes_test_pdf",
    "seltox": "test_data/seltox_test_pdf",
    "magnetic": "test_data/magnetic_test_pdf",
    "synergy": "test_data/synergy_test_pdf",
    "cytotoxicity": "test_data/cytox_test_pdf",
}

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def main():
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        # base_url=os.getenv("OPENAI_BASE_URL"),
    )

    for dataset, pdf_articles_dir in DATASET_TO_DIR.items():
        prompt_path = PROMPTS_DIR / f"{dataset}.txt"
        if not prompt_path.exists():
            LOGGER.warning(
                f"Prompt file '{prompt_path}' for dataset '{dataset}' not found. Skipping."
            )
            continue
        try:
            with open(prompt_path, "r", encoding="utf-8") as pf:
                prompt = pf.read()
        except Exception as e:
            LOGGER.error(f"Failed to read prompt for dataset '{dataset}': {e}")
            continue

        directory = str(Path(pdf_articles_dir))

        results_dir = os.path.join("results", dataset)
        os.makedirs(results_dir, exist_ok=True)

        extension = ".pdf"
        articles_files = get_files_with_extension(directory, extension)
        LOGGER.info(f"Dataset '{dataset}' — files count: {len(articles_files)}")

        def process_article(article_file: str):
            try:
                start_time = time()
                LOGGER.info(
                    f"GPT-5 extraction start: {article_file[len(directory)+1:]} (dataset: {dataset})"
                )

                article_text = extract_text_from_pdf(article_file)

                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": article_text},
                ]

                response_text = None
                for _ in range(5):
                    try:
                        completion = client.chat.completions.create(
                            model="gpt-5",
                            messages=messages,
                            # reasoning_effort="high",
                        )
                        response_text = completion.choices[0].message.content or ""
                        break
                    except Exception as e:
                        LOGGER.error(e)

                out_path = f"{results_dir}/{article_file[len(directory)+1:-4]}.md"
                with open(out_path, "w") as f:
                    f.write(response_text if response_text is not None else "Error: see logs")

                end_time = time()
                LOGGER.info(
                    f"Article {article_file[len(directory)+1:]} was processed in {end_time-start_time:.2f}s"
                )
                LOGGER.info("")
            except Exception as e:
                LOGGER.error(e)
                out_path = f"{results_dir}/{article_file[len(directory)+1:-4]}.md"
                with open(out_path, "w") as f:
                    f.write(f"Error: \n{str(e)}")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_article, f) for f in articles_files]
            for _ in tqdm(as_completed(futures), total=len(futures)):
                pass


if __name__ == "__main__":
    main()


