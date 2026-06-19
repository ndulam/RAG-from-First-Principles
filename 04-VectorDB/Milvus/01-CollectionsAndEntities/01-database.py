# Install dependency: pip install pymilvus
# pip show pymilvus  # check the current SDK version
'''
# Install the Milvus server

wget https://github.com/milvus-io/milvus/releases/download/v2.5.10/milvus-standalone-docker-compose.yml -O docker-compose.yml

sudo docker compose up -d

Creating milvus-etcd  ... done
Creating milvus-minio ... done
Creating milvus-standalone ... done

'''

from pymilvus import MilvusClient, exceptions

# ——————————————
# 1. Connect to Milvus Standalone
# ——————————————
# uri: protocol+address+port, defaults to http://localhost:19530
# token: "username:password", defaults to root:Milvus
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

# ——————————————
# 2. Create database my_database_1 (no extra properties)
# ——————————————
try:
    client.create_database(db_name="my_database_1")
    print("✓ my_database_1 created successfully")
except exceptions.AlreadyExistError:
    print("ℹ my_database_1 already exists")

# ——————————————
# 3. Create database my_database_2 (set replica count to 3)
# ——————————————
client.create_database(
    db_name="my_database_2",
    properties={"database.replica.number": 3}
)
print("✓ my_database_2 created successfully, replicas=3")

# ——————————————
# 4. List all databases
# ——————————————
db_list = client.list_databases()
print("All current databases:", db_list)

# ——————————————
# 5. View details of the default database
# ——————————————
default_info = client.describe_database(db_name="default")
print("Default database details:", default_info)

# ——————————————
# 6. Modify my_database_1 properties: limit max collections to 10
# ——————————————
client.alter_database_properties(
    db_name="my_database_1",
    properties={"database.max.collections": 10}
)
print("✓ Limited my_database_1 to a maximum of 10 collections")

# ——————————————
# 7. Remove the max.collections limit on my_database_1
# ——————————————
client.drop_database_properties(
    db_name="my_database_1",
    property_keys=["database.max.collections"]
)
print("✓ Removed the max collections limit on my_database_1")

# ——————————————
# 8. Switch to my_database_2 (all subsequent operations apply to this database)
# ——————————————
client.use_database(db_name="my_database_2")
print("✓ Switched the current database to my_database_2")

# ——————————————
# 9. Delete database my_database_2
#    (note: if the database has collections, drop them first with client.drop_collection())
# ——————————————
client.drop_database(db_name="my_database_2")
print("✓ my_database_2 deleted")

# ——————————————
# 10. Delete database my_database_1
# ——————————————
client.drop_database(db_name="my_database_1")
print("✓ my_database_1 deleted")
