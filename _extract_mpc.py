import pypdfium2 as pdfium

doc = pdfium.PdfDocument("mpc.pdf")
n = len(doc)
print("PAGES", n)
txt = []
for i in range(n):
    txt.append("\n\n===== PAGE %d =====\n" % (i + 1) + doc[i].get_textpage().get_text_range())
full = "".join(txt)
open("mpc_text.txt", "w").write(full)
print("CHARS", len(full))
