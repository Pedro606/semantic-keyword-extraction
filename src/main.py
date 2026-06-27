from pathlib import Path

from src.services.keyword_extractor import (
    KeywordExtractor
)

from src.config.finance_terms import (
    TERMS_PRIORITARIOS
)


def main():

    script = Path(
        "data/sample_script.txt"
    ).read_text(
        encoding="utf-8"
    )

    extractor = (
        KeywordExtractor()
    )

    keywords = (
        extractor.extract_keywords(
            text=script,
            candidate_terms=(
                TERMS_PRIORITARIOS
            ),
            top_n=5
        )
    )

    print(
        "\nTop Keywords:\n"
    )

    for keyword in keywords:

        print(
            f"- {keyword}"
        )


if __name__ == "__main__":
    main()