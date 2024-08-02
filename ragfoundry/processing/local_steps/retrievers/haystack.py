from haystack import Pipeline

from ...step import LocalStep


class HaystackRetriever(LocalStep):
    """
    Class for document retrieval using Haystack v2 pipelines.
    """

    def __init__(self, pipeline_or_yaml_path, docs_key, query_key, **kwargs):
        super().__init__(**kwargs)
        if isinstance(pipeline_or_yaml_path, str):
            self.pipe = Pipeline.load(open(pipeline_or_yaml_path))
        else:
            self.pipe = pipeline_or_yaml_path

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

        For example, structure could look like this:
        {
            "Retriever": {"query": "query",},
            "Reranker": {"query": "query"},
        }
        and we replace the **value** of each key with the query.
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
                # has documents, add to list
                all_documents += v["documents"]

        all_documents = [
            {"content": d.content, "title": d.meta.get("title")} for d in all_documents
        ]

        return all_documents

    def process_item(self, item, index, datasets, **kwargs):
        """
        Query the `query_key` in the item and store the results in the `docs_key`.
        Retrieved documents are stored as a list of dictionaries with keys `content` and `title`.
        """
        item[self.docs_key] = self.query(item[self.query_key])
        return item
