from unstructured.partition.auto import partition
filename = "90-文档-Data/黑Wukong/black mythWukong.pdf"
elements = partition(filename=filename, 
                     content_type="application/pdf"
                    )
print("\n\n".join([str(el) for el in elements][:10]))

