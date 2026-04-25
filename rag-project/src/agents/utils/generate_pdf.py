from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# Output PDF file name
file_path = "ecommerce_support_kb.pdf"

doc = SimpleDocTemplate(file_path, pagesize=A4)

styles = getSampleStyleSheet()

content = []

# Title
content.append(Paragraph("E-Commerce Customer Support Knowledge Base", styles["Title"]))
content.append(Spacer(1, 12))

# Sections of knowledge base
sections = {
    "Orders": """
Customers can place orders through the website or mobile app.
Order confirmation is sent via email and SMS.
Customers can track orders using the order ID.
""",

    "Payments": """
We support UPI, debit/credit cards, and Cash on Delivery (COD).
Failed transactions are refunded within 5–7 business days.
""",

    "Shipping & Delivery": """
Delivery usually takes 3–7 business days depending on location.
A tracking link is provided after the order is shipped.
Delays may occur due to weather or logistics issues.
""",

    "Returns & Refunds": """
Products can be returned within 7 days if unused and in original packaging.
Refunds are processed within 5–10 business days after approval.
Certain items like hygiene products are non-returnable.
""",

    "Cancellations": """
Orders can be cancelled before shipping.
Once shipped, cancellation is not allowed and return process must be followed.
""",

    "Account Issues": """
Users can reset passwords using 'Forgot Password'.
If email access is lost, support assistance is required.
""",

    "FAQs": """
Q: How do I track my order?
A: Use the tracking link sent after shipment.

Q: What if I receive a damaged product?
A: Report within 24 hours for replacement or refund.

Q: How to contact support?
A: Use website chat or customer support email.
"""
}

# Build PDF content
for title, text in sections.items():
    content.append(Paragraph(f"<b>{title}</b>", styles["BodyText"]))
    content.append(Spacer(1, 6))
    content.append(Paragraph(text, styles["BodyText"]))
    content.append(Spacer(1, 12))

# Generate PDF
doc.build(content)

print("PDF created successfully: ecommerce_support_kb.pdf")