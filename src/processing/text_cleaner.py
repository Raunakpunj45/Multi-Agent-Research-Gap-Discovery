import re


class TextCleaner:
    def clean(self, text: str) -> str:
        if not text:
            return ""

        # Replace line breaks and tabs with spaces
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")

        # Remove repeated whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()