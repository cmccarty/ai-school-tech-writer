from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
import psycopg2
import psycopg2.extras
from typing import Any, List


class PostgresLoader(BaseLoader):
    """Load info from Postgres tables/query"""
    def __init__(self, db_url: str, debug: bool = False):
        self.db_url = db_url
        self.debug = debug

    def load(self, **kwargs: Any) -> List[Document]:
        """
        :param kwargs:
        query: str, required, the query to run
        content_columns=[]
        metadata_columns=["id"],
        :return:
        """

        # Validate query presence
        query = kwargs.get("query", None)
        if query is None:
            raise ValueError("Query is required to load from Postgres")

        content_columns = kwargs.get("content_columns", None)
        metadata_columns = kwargs.get("metadata_columns", None)

        # Create a Read-Only connection to the DB (for extra safety)
        conn = psycopg2.connect(self.db_url)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY;")

            # Run the query, load everything up as documents
            print(f"Loading from Postgres with query: {query}")

            cursor.execute(query)
            results = cursor.fetchall()
            print(f"Loaded {len(results)} rows from Postgres")
            documents = [self._process_row(row, content_columns=content_columns, metadata_columns=metadata_columns) for row in results]

        return documents


    def _process_row(self, row, content_columns=None, metadata_columns=None) -> Document:
        # Create content string block
        if content_columns is None:
            # use all columns returned in the content block
            content_lines = [f"{self._humanize_column_name(col)}: {val}" for col, val in row.items()]
        else:
            # use only the specified columns
            if len(content_columns) == 1:
                # just 1 column, no need to add a prefix to it
                content_lines = [row[content_columns[0]]]
            else:
                # prefix each column with its name
                content_lines = [f"{self._humanize_column_name(col)}: {row[col]}" for col in content_columns]


        # Build metadata
        if metadata_columns is None:
            metadata = {
                "id": row.get("id"),
            }

            if "title" in row:
                metadata["title"] = row["title"]

            # Timestamp info
            if "updated_at" in row:
                metadata["when"] = row["updated_at"]
            elif "created_at" in row:
                metadata["when"] = row["created_at"]
        else:
            metadata = {col: row[col] for col in metadata_columns}

        # Strip trailing whitespace from metadata if needed. Strip empty values
        metadata = {col: value.strip() if isinstance(value, str) else value for col, value in metadata.items()}
        metadata = {k: v for k, v in metadata.items() if v not in [None, '']}

        # Create Document
        return Document(
            page_content="\n".join(content_lines),
            metadata=metadata,
        )


    def _humanize_column_name(self, column_name: str) -> str:
        """Convert a snake_case column name to a human-readable format."""
        return column_name.replace('_', ' ').capitalize()

# =============================================================================
def test():
    loader = PostgresLoader(db_url="postgres://kimkim:@localhost:5432/kimkim_dev")
    docs = loader.load(
        query="SELECT academy_contents.*, academy_contents.updated_at AS when, academy_content_categories.name as category FROM academy_contents LEFT JOIN academy_content_categories ON academy_content_categories.id = category_id  WHERE status = 1 LIMIT 5",
        content_columns=['content'],
        metadata_columns=['id', 'slug', 'title', 'category', 'when']
    )
    print(docs[0])

if __name__ == '__main__':
    test()