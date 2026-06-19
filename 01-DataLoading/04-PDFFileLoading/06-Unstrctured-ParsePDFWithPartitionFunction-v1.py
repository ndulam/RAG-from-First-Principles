from unstructured.partition.auto import partition
filename = "../../99-EN/black-myth-wukong/black_myth_wukong_slides.pdf"
elements = partition(filename=filename, 
                     content_type="application/pdf"
                    )
print("\n\n".join([str(el) for el in elements][:10]))

