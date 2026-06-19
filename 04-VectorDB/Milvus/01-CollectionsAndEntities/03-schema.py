# Install dependency: pip install pymilvus
from pymilvus import MilvusClient, DataType

# ——————————————
# 0. Connect to Milvus
# ——————————————
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)
print("✓ Connected to the Milvus API")

# ——————————————
# 1. Create a basic Schema
# ——————————————
schema = MilvusClient.create_schema()
print("✓ Created an empty Schema")

# ——————————————
# 2. Add the primary key field (Primary Field)
# ——————————————
# 2.1 INT64 primary key (manually specified ID)
schema.add_field(
    field_name="id",
    datatype=DataType.INT64,
    is_primary=True,  # set as the primary key
    auto_id=False     # do not auto-generate the ID
)

# 2.2 VARCHAR primary key (auto-generated ID)
# schema.add_field(
#     field_name="doc_id",
#     datatype=DataType.VARCHAR,
#     is_primary=True,  # set as the primary key
#     auto_id=True,     # auto-generate the ID
#     max_length=100    # VARCHAR requires a max length
# )
print("✓ Added the primary key field")

# ——————————————
# 3. Add vector fields (Vector Field)
# ——————————————
# 3.1 Dense Vector (floating-point vector)
schema.add_field(
    field_name="text_vector",
    datatype=DataType.FLOAT_VECTOR,  # 32-bit float vector
    dim=768                          # vector dimension
)

# 3.2 Binary Vector
schema.add_field(
    field_name="image_vector",
    datatype=DataType.BINARY_VECTOR,  # binary vector
    dim=256                           # dimension must be a multiple of 8
)
print("✓ Added the vector fields")

# ——————————————
# 4. Add scalar fields (Scalar Field)
# ——————————————
# 4.1 String field
schema.add_field(
    field_name="title",
    datatype=DataType.VARCHAR,
    max_length=200,
    # nullable, with a default value
    is_nullable=True,
    default_value="untitled"
)

# 4.2 Numeric field
schema.add_field(
    field_name="age",
    datatype=DataType.INT32,
    is_nullable=False  # cannot be null
)

# 4.3 Boolean field
schema.add_field(
    field_name="is_active",
    datatype=DataType.BOOL,
    default_value=True  # default value is True
)

# 4.4 JSON field
schema.add_field(
    field_name="metadata",
    datatype=DataType.JSON
)

# 4.5 Array field
schema.add_field(
    field_name="tags",
    datatype=DataType.ARRAY,
    element_type=DataType.VARCHAR,  # array element type
    max_capacity=10,                # max array capacity
    max_length=50                   # max length of each element
)
print("✓ Added the scalar fields")

# ——————————————
# 5. Add a dynamic field (Dynamic Field)
# ——————————————
# schema.add_field(
#     field_name="dynamic_field",
#     datatype=DataType.VARCHAR,
#     is_dynamic=True,    # set as a dynamic field
#     max_length=500
# )
print("✓ Added the dynamic field")

# ——————————————
# 6. Create a Collection using the Schema
# ——————————————
collection_name = "document_store10"
client.create_collection(
    collection_name=collection_name,
    schema=schema
)
print(f"✓ Created collection {collection_name}")

# ——————————————
# 7. Modify Collection fields
# ——————————————
# Add a new field
# client.alter_collection_field(
#     collection_name=collection_name,
#     field_name="tags",
#     field_params={
#         "max_capacity": 64
#     }
# )
# print("✓ Added the new field")

# ——————————————
# 8. View Collection details
# ——————————————
info = client.describe_collection(collection_name=collection_name)
print("Collection details:", info)

# ——————————————
# 9. Cleanup
# ——————————————
client.drop_collection(collection_name=collection_name)
print("✓ Deleted the test collection")
