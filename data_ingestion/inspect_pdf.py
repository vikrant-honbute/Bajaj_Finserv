"""Get full text of a sample fund page (page 16 - Large Cap Fund) and page 46 (Performance)."""
import fitz
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

doc = fitz.open("Factsheet_July-2026.pdf")

# Page 16 - Bajaj Finserv Large Cap Fund (detailed fund page)
print("=== PAGE 16: Bajaj Finserv Large Cap Fund ===")
print(doc[15].get_text())

print("\n\n=== PAGE 46: Performance Data ===")
print(doc[45].get_text())

print("\n\n=== PAGE 44: Nifty 50 Index Fund ===")
print(doc[43].get_text())
