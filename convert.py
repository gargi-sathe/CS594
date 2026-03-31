with open("pdf_content.txt", "r", encoding="utf-16le") as f_in:
    with open("pdf_content_utf8.txt", "w", encoding="utf-8") as f_out:
        f_out.write(f_in.read())
