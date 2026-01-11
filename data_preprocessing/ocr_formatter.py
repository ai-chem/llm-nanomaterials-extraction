"""Format enriched OCR output to clean markdown text."""

import re


def format_to_markdown(enriched_ocr: dict) -> str:
    """
    Convert enriched OCR result to clean markdown text.
    
    Replaces image markers with captions and adds table captions.
    Groups images with common figure captions to avoid repetition.
    
    Args:
        enriched_ocr: OCR result enriched with captions
        
    Returns:
        Clean markdown text
    """
    pages = []
    
    for page in enriched_ocr.get("pages", []):
        page_num = page.get("index", 0) + 1
        markdown = page.get("markdown", "")
        images = page.get("images", [])
        tables = page.get("tables", [])
        
        pages.append(f"---\n# Page {page_num}\n---\n")
        
        printed_captions = set()
        
        for img in images:
            img_id = img.get("id", "")
            caption = img.get("caption", "")
            figure_caption = img.get("figure_caption", "")
            marker = f"![{img_id}]({img_id})"
            
            if figure_caption and caption:
                if figure_caption not in printed_captions:
                    replacement = f"\n**{figure_caption}**\n- {caption}\n"
                    printed_captions.add(figure_caption)
                else:
                    replacement = f"- {caption}\n"
            elif figure_caption:
                if figure_caption not in printed_captions:
                    replacement = f"\n**{figure_caption}**\n"
                    printed_captions.add(figure_caption)
                else:
                    replacement = ""
            elif caption:
                replacement = f"\n**{caption}**\n"
            else:
                replacement = ""
            
            markdown = markdown.replace(marker, replacement)
        
        for tbl in tables:
            tbl_caption = tbl.get("caption", "")
            tbl_markdown = tbl.get("markdown", "")
            
            if tbl_caption and tbl_markdown in markdown:
                markdown = markdown.replace(
                    tbl_markdown,
                    f"\n**{tbl_caption}**\n{tbl_markdown}"
                )
        
        pages.append(markdown)
    
    return "\n\n".join(pages)


def remove_references(text: str) -> str:
    """Remove everything after References section."""
    return re.sub(r"(?i)references.*", "References", text, flags=re.DOTALL)
