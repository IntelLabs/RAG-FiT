from ...step import LocalStep
from haystack import Pipeline
from haystack.utils import Secret
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack.components.embedders.sentence_transformers_text_embedder import SentenceTransformersTextEmbedder

class HaystackRetriever(LocalStep):
    """
    Class for document retrieval using Haystack v2 pipelines.
    """
    def __init__(self, pipeline_or_yaml_path=None, docs_key="positive_passages", query_key="query", **kwargs):
        super().__init__(**kwargs)
        
        document_store = QdrantDocumentStore(
            url="",   #add your url from qdrant cloud,like xxxx.gcp.cloud.qdrant.io
            api_key=Secret.from_token(""), #add your api_key copy from qdrant cloud
            https=True,
            port=6333,
            index="wikipedia",
            embedding_dim=768,
            similarity="dot_product",
            write_batch_size=50,
            prefer_grpc=False
        )
        
        retriever = QdrantEmbeddingRetriever(
            document_store=document_store,
            top_k=10
        )
        
        text_embedder = SentenceTransformersTextEmbedder(
            model="BAAI/llm-embedder",
            prefix="Represent this query for retrieving relevant documents: ",
            batch_size=64
        )
        
        self.pipe = Pipeline()
        self.pipe.add_component("text_embedder", text_embedder)
        self.pipe.add_component("retriever", retriever)
        self.pipe.connect("text_embedder.embedding", "retriever.query_embedding")
            
        self.docs_key = docs_key
        self.query_key = query_key

    def default_query_function(self, query):
        """
        Create the default querying of the pipeline, by inserting the input query into all mandatory fields.
        """
        pipe_inputs = self.pipe.inputs()
        query_dict = {}
        for inp_node_name, inp_node_params in pipe_inputs.items():
            for param_name, param_values in inp_node_params.items():
                if param_values["is_mandatory"]:
                    if inp_node_name not in query_dict:
                        query_dict[inp_node_name] = {}
                    query_dict[inp_node_name][param_name] = query
        return query_dict   

    def query(self, query, structure=None):
        """
        Haystack v2 pipelines can have multiple inputs; structure specify how to call `pipe.run`.
        """
        if structure is None:
            structure = self.default_query_function(query)
        else:
            for key, value in structure.items():
                structure[key] = {k: query for k in value.keys()}
        response = self.pipe.run(structure)
        all_documents = []
        for v in response.values():
            if "documents" in v:
                all_documents += v["documents"]
        all_documents = [
            {"content": d.content, "title": d.meta.get("title")} for d in all_documents
        ]
        return all_documents

    def process_item(self, item, index, datasets, **kwargs):
        """
        Query the `query_key` in the item and store the results in the `docs_key`.
        """
        item[self.docs_key] = self.query(item[self.query_key])
        return item