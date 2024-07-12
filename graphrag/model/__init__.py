# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
kiku: https://microsoft.github.io/graphrag/posts/index/1-default_dataflow/
The GraphRAG Knowledge Model is a specification for data outputs that 
conform to our data-model definition.

This model is designed to be an abstraction over the underlying data storage technology, 
and to provide a common interface for the GraphRAG system to interact with. 

In normal use-cases the outputs of the GraphRAG Indexer would be loaded into a database system, 
and the GraphRAG's Query Engine would interact with the database using the knowledge model 
data-store types.

The following entity types are provided. The fields here represent the fields that are text-embedded by default.
- Document - An input document into the system. These either represent individual rows in a CSV or individual .txt file.
- TextUnit - A chunk of text to analyze. The size of these chunks, their overlap, and whether they adhere to any data boundaries may be configured below. A common use case is to set CHUNK_BY_COLUMNS to id so that there is a 1-to-many relationship between documents and TextUnits instead of a many-to-many.
- Entity - An entity extracted from a TextUnit. These represent people, places, events, or some other entity-model that you provide.
- Relationship - A relationship between two entities. These are generated from the covariates.
- Covariate - Extracted claim information, which contains statements about entities which may be time-bound.
- Community Report - Once entities are generated, we perform hierarchical community detection on them and generate reports for each community in this hierarchy.
- Node - This table contains layout information for rendered graph-views of the Entities and Documents which have been embedded and clustered.
"""

"""
GraphRAG knowledge model package root.

The GraphRAG knowledge model contains a set of classes that represent the target datamodels for our pipelines and analytics tools.
These models can be augmented and integrated into your own data infrastructure to suit your needs.
"""

from .community import Community
from .community_report import CommunityReport
from .covariate import Covariate
from .document import Document
from .entity import Entity
from .identified import Identified
from .named import Named
from .relationship import Relationship
from .text_unit import TextUnit

__all__ = [
    "Community",
    "CommunityReport",
    "Covariate",
    "Document",
    "Entity",
    "Identified",
    "Named",
    "Relationship",
    "TextUnit",
]
