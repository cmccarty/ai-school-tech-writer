from constants import *

from upload_confluence import ConfluenceRAG
from upload_academy import AcademyRAG

def seed_all(index):
    print("Seeding all spaces...")

    # Seed the Engineering Space
    ConfluenceRAG(pinecone_index=index, space_key="ENGINEERIN").seed()

    # Seed the Product Space
    ConfluenceRAG(pinecone_index=index, space_key="PM").seed()

    # Load the Academy Articles
    AcademyRAG(pinecone_index=index).seed()

    print("DONE")


if __name__ == '__main__':
    seed_all(PINECONE_INDEX)