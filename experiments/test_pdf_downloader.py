from src.processing.pdf_downloader import PDFDownloader


downloader = PDFDownloader()

pdf_url = "https://arxiv.org/pdf/2504.05496"

output_path = downloader.download(
    pdf_url=pdf_url,
    paper_id="test_paper",
)

print(f"\nPDF downloaded successfully:")
print(output_path)