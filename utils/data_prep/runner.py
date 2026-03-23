from .related_tables import update_table_relations
from .metadata import update_metadata
from utils.database import Database

def run(db_id: str = "reef_db"):
    print("Starting data prep pipeline...")

    print("--- generating(updating) table relations...")
    # generate and update table relation data
    schema, change = update_table_relations(db_id)

    # generate and update per-table metadata
    print(f"----- generating(updating) metadata...")
    update_metadata(db_id, schema, change)
        
    print("------- Data prep pipeline complete!")

if __name__ == "__main__":
    database = Database()
    databases = database.list_databases() 

    print(f"============Running data prep for database: reef_db=============")
    run("reef_db")

    # for db_id in databases:
    #     if db_id != "bird" and db_id != "finance":
    #         print(f"============Running data prep for database: {db_id}=============")
    #         run(db_id)
