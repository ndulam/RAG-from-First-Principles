import fitz  # PyMuPDF library, used to process PDF files
import matplotlib.patches as patches  # used to draw polygons on the image
import matplotlib.pyplot as plt  # Matplotlib library, used for plotting
from PIL import Image  # used for image processing

def render_pdf_page(file_path, doc_list, page_number):
    """
    Render the given PDF page and draw bounding boxes for each segment category.

    Parameters:
    - file_path: str, the PDF file path.
    - doc_list: list, a list of documents containing segment info, where each element is a
      dict containing the segment's metadata.
    - page_number: int, the page number to render (counting from 1).
    """
    # Open the PDF file and load the given page
    pdf_page = fitz.open(file_path).load_page(page_number - 1)
    segments = [doc.metadata for doc in doc_list if doc.metadata.get("page_number") == page_number]

    # Render the PDF page as a bitmap image
    pix = pdf_page.get_pixmap()
    pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Set up the plotting environment
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(pil_image)

    # Define the category-to-color mapping
    category_to_color = {"Title": "orchid", "Image": "forestgreen", "Table": "tomato"}
    categories = set()

    # Draw the bounding box for each segment
    for segment in segments:
        points = segment["coordinates"]["points"]
        layout_width = segment["coordinates"]["layout_width"]
        layout_height = segment["coordinates"]["layout_height"]
        scaled_points = [
            (x * pix.width / layout_width, y * pix.height / layout_height) for x, y in points
        ]
        box_color = category_to_color.get(segment["category"], "deepskyblue")
        categories.add(segment["category"])
        rect = patches.Polygon(scaled_points, linewidth=1, edgecolor=box_color, facecolor="none")
        ax.add_patch(rect)

    # Add the legend
    legend_handles = [patches.Patch(color="deepskyblue", label="Text")]
    for category, color in category_to_color.items():
        if category in categories:
            legend_handles.append(patches.Patch(color=color, label=category))
    ax.axis("off")
    ax.legend(handles=legend_handles, loc="upper right")
    plt.tight_layout()
